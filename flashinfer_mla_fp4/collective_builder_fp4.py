# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Collective builder for per-token-scaled FP4 MLA decode."""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.typing import Int64

from flashinfer.cute_dsl.attention.mainloop_spec import MLAMainloopSpec
from flashinfer.cute_dsl.attention.mla_warp_schedule import MLAWarpScheduleFP8


def _make_paged_tiled_tma_atom(
    tma_load_op: cpasync.CopyBulkTensorTileG2SOp,
    gmem: cute.Tensor,
    smem_layout: cute.Layout,
    mma_tiler,
    tiled_mma: cute.TiledMma,
    page_size: int,
    is_k_load: bool,
    internal_type=None,
):
    """Create paged TMA, optionally unpacking sub-byte data into SMEM."""

    identity = cute.make_identity_layout(gmem.shape)
    g_tile = cute.composition(identity, mma_tiler)
    cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
    cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
    cta_v_map = cute.select(cta_v_map, mode=[0, 2])
    page_tile_size = (
        min(page_size, cta_mn) if is_k_load else min(page_size, mma_tiler[1])
    )
    cta_v_map = cute.zipped_divide(
        cta_v_map,
        ((page_tile_size, mma_tiler[1]) if is_k_load else (cta_mn, page_tile_size)),
    )
    cta_v_map = cute.select(cta_v_map, mode=[0])

    from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

    tma_format = None
    if internal_type is not None:
        use_unpack = internal_type.width == 8 and gmem.element_type.width < 8
        internal_mlir_type = (
            gmem.element_type.mlir_type if use_unpack else internal_type.mlir_type
        )
        tma_format = _cute_nvgpu_ir.TmaDataFormat(
            _cute_nvgpu_ir.get_default_tma_format(
                internal_mlir_type,
                use_unpack,
            )
        )
    result = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
        gmem.value,
        smem_layout.value,
        cta_v_map,
        tma_load_op._to_ir(),
        num_multicast=1,
        tma_format=tma_format,
    )
    return cute.CopyAtom(
        tma_load_op,
        cpasync.CopyBulkTensorTileG2SNonExecTrait(result[0]),
    ), result[1]


def build_mla_fp4_launch_params(
    mainloop: MLAMainloopSpec,
    schedule: MLAWarpScheduleFP8,
    q_latent: cute.Tensor,
    q_rope: cute.Tensor,
    c_latent: cute.Tensor,
    c_rope: cute.Tensor,
    c_latent_transpose: cute.Tensor,
    q_dtype,
    k_dtype,
    v_dtype,
) -> SimpleNamespace:
    """Build mixed E4M3/E2M1 MMA, TMA, and shared-storage launch parameters.

    The FP4 path keeps the split-loader topology and specializes:
    - Separate KC-latent, KC-rope, and VC SMEM buffers (no aliasing)
    - KC-latent stages use logical_divide for iterations_qk_latent
    - VC stages use nested logical_divide for iterations_pv_k * iterations_pv_n
    - No page-table SMEM buffer or load_pt barriers
    - Separate tma_copy_kc_bytes and tma_copy_vc_bytes
    """
    config = mainloop.config

    cta_group = tcgen05.CtaGroup.TWO
    q_major_mode = OperandMajorMode.K
    k_major_mode = OperandMajorMode.K
    v_major_mode = OperandMajorMode.MN
    p_major_mode = OperandMajorMode.K
    # F8F6F4 consumes an unpacked low-precision SMEM representation. TMA
    # expands each packed E2M1 latent nibble into one byte. RoPE remains E4M3
    # so the independent 64-d E4M3 MMA stays legal and numerically complete.
    kv_smem_dtype = cutlass.Uint8
    rope_smem_dtype = q_dtype

    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        q_dtype,
        k_dtype,
        q_major_mode,
        k_major_mode,
        config.acc_dtype,
        cta_group,
        config.mma_qk_tiler[:2],
    )
    qk_rope_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        q_dtype,
        q_dtype,
        q_major_mode,
        k_major_mode,
        config.acc_dtype,
        cta_group,
        config.mma_qk_tiler[:2],
    )
    pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        q_dtype,
        v_dtype,
        p_major_mode,
        v_major_mode,
        config.acc_dtype,
        cta_group,
        config.mma_pv_tiler[:2],
    )

    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(config.cluster_shape_mnk),
        (qk_tiled_mma.thr_id.shape,),
    )
    # --- Q SMEM layouts (same structure as FP16) ---
    q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma,
        config.mma_qk_tiler,
        q_dtype,
        (config.iterations_qk_latent * config.load_q_stage),
    )
    q_latent_smem_layout_staged = cute.logical_divide(
        q_latent_smem_layout_staged,
        (None, None, None, config.iterations_qk_latent),
    )
    q_rope_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_rope_tiled_mma,
        config.mma_qk_rope_tiler,
        q_dtype,
        config.load_q_stage,
    )

    # --- KC-latent SMEM: separate buffer with logical_divide for latent iterations ---
    kc_latent_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma,
        config.mma_qk_tiler,
        kv_smem_dtype,
        (config.iterations_qk_latent * config.load_k_stage),
    )
    kc_page_tile_size = min(
        config.page_size,
        qk_tiled_mma.op.shape_mnk[0] // qk_tiled_mma.thr_id.shape,
    )
    kc_latent_smem_layout_staged = cute.logical_divide(
        kc_latent_smem_layout_staged,
        (None, None, None, config.iterations_qk_latent),
    )

    kc_latent_smem_layout_for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.K,
        (config.mma_qk_tiler[0] // qk_tiled_mma.thr_id.shape, config.mma_qk_tiler[2]),
        kv_smem_dtype,
        (config.iterations_qk_latent * config.load_k_stage),
    )
    kc_latent_smem_layout_for_tma = cute.tiled_divide(
        kc_latent_smem_layout_for_tma,
        (kc_page_tile_size, config.mma_qk_tiler[2]),
    )
    kc_latent_smem_layout_for_tma = cute.logical_divide(
        kc_latent_smem_layout_for_tma,
        (None, None, None, config.iterations_qk_latent),
    )

    # --- KC-rope SMEM: separate buffer ---
    kc_rope_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_rope_tiled_mma,
        config.mma_qk_rope_tiler,
        rope_smem_dtype,
        config.load_k_stage,
    )
    kc_rope_smem_layout_for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.K,
        (
            config.mma_qk_rope_tiler[0] // qk_tiled_mma.thr_id.shape,
            config.mma_qk_rope_tiler[2],
        ),
        rope_smem_dtype,
        config.load_k_stage,
    )
    kc_rope_smem_layout_for_tma = cute.tiled_divide(
        kc_rope_smem_layout_for_tma,
        (kc_page_tile_size, config.mma_qk_rope_tiler[2]),
    )

    # --- P SMEM layout ---
    p_smem_layout_staged = sm100_utils.make_smem_layout_a(
        pv_tiled_mma,
        config.mma_pv_tiler,
        q_dtype,
        (config.iterations_pv_k * config.p_mma_stage),
    )
    p_smem_layout_staged = cute.logical_divide(
        p_smem_layout_staged,
        (None, None, None, config.iterations_pv_k),
    )

    # --- VC SMEM: separate buffer with nested logical_divide ---
    vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma,
        config.mma_pv_tiler,
        kv_smem_dtype,
        (config.iterations_pv_k * config.iterations_pv_n * config.load_v_stage),
    )
    vc_smem_layout_staged = cute.logical_divide(
        cute.logical_divide(
            vc_smem_layout_staged,
            (None, None, None, config.iterations_pv_k * config.iterations_pv_n),
        ),
        (None, None, None, (config.iterations_pv_n, None)),
    )
    vc_page_tile_size = min(config.page_size, config.mma_pv_tiler[2])
    vc_smem_layout_for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.MN,
        (config.mma_pv_tiler[1] // pv_tiled_mma.thr_id.shape, config.mma_pv_tiler[2]),
        kv_smem_dtype,
        (config.iterations_pv_k * config.iterations_pv_n * config.load_v_stage),
    )
    vc_smem_layout_for_tma = cute.tiled_divide(
        vc_smem_layout_for_tma,
        (
            pv_tiled_mma.op.shape_mnk[1] // pv_tiled_mma.thr_id.shape,
            vc_page_tile_size,
        ),
    )
    vc_smem_layout_for_tma = cute.logical_divide(
        cute.logical_divide(
            vc_smem_layout_for_tma,
            (None, None, None, config.iterations_pv_k * config.iterations_pv_n),
        ),
        (None, None, None, (config.iterations_pv_n, None)),
    )

    # --- TMA atoms ---
    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

    q_smem_layout = cute.select(q_latent_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        q_latent,
        q_smem_layout,
        config.mma_qk_tiler,
        qk_tiled_mma,
        cta_layout_vmnk.shape,
    )
    q_rope_smem_layout = cute.select(q_rope_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q_rope, tma_tensor_q_rope = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        q_rope,
        q_rope_smem_layout,
        config.mma_qk_rope_tiler,
        qk_rope_tiled_mma,
        cta_layout_vmnk.shape,
    )

    # Paged TMA unpacks the E2M1 latent payload into MMA-compatible SMEM.
    # With Float4E2M1FN tensors the descriptor lowers to the sub-byte 16U4
    # tensor-map format and writes one byte container per E2M1 nibble into the
    # MMA-compatible SMEM layout.
    kc_smem_layout = cute.select(kc_latent_smem_layout_for_tma, mode=[0])
    tma_atom_c_latent, tma_tensor_c_latent = _make_paged_tiled_tma_atom(
        tma_load_op,
        c_latent,
        kc_smem_layout,
        (config.mma_qk_tiler[1], config.mma_qk_tiler[2]),
        qk_tiled_mma,
        config.page_size,
        is_k_load=True,
        internal_type=kv_smem_dtype,
    )
    kc_rope_smem_layout = cute.select(kc_rope_smem_layout_for_tma, mode=[0])
    tma_atom_c_rope, tma_tensor_c_rope = _make_paged_tiled_tma_atom(
        tma_load_op,
        c_rope,
        kc_rope_smem_layout,
        (config.mma_qk_rope_tiler[1], config.mma_qk_rope_tiler[2]),
        qk_rope_tiled_mma,
        config.page_size,
        is_k_load=True,
    )

    vc_smem_layout = cute.select(vc_smem_layout_for_tma, mode=[0])
    tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose = (
        _make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent_transpose,
            vc_smem_layout,
            (config.mma_pv_tiler[1], config.mma_pv_tiler[2]),
            pv_tiled_mma,
            config.page_size,
            is_k_load=False,
            internal_type=kv_smem_dtype,
        )
    )

    # --- Copy sizes ---
    q_latent_copy_size = (
        cute.size_in_bytes(q_dtype, q_smem_layout)
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_latent
    )
    q_rope_copy_size = (
        cute.size_in_bytes(q_dtype, q_rope_smem_layout)
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_rope
    )
    tma_copy_q_bytes = q_latent_copy_size + q_rope_copy_size

    kc_latent_copy_size = (
        cute.size_in_bytes(
            k_dtype,
            cute.select(kc_latent_smem_layout_staged, mode=[0, 1, 2]),
        )
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_latent
    )
    kc_rope_copy_size = (
        cute.size_in_bytes(
            rope_smem_dtype,
            cute.select(kc_rope_smem_layout_staged, mode=[0, 1, 2]),
        )
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_rope
    )
    tma_copy_kc_bytes = kc_latent_copy_size + kc_rope_copy_size

    tma_copy_vc_bytes = (
        cute.size_in_bytes(
            v_dtype,
            cute.select(vc_smem_layout_staged, mode=[0, 1, 2]),
        )
        * cute.size(pv_tiled_mma.thr_id.shape)
        * config.iterations_pv_n
        * config.iterations_pv_k
    )

    # --- SharedStorage struct (no page-table buffer) ---
    align = mainloop.buffer_align_bytes
    threads_per_warp = schedule.threads_per_warp
    num_compute_warps = config.num_compute_warps
    token_scale_stages = config.mma_s_stage
    token_scale_elements = (
        config.mma_qk_tiler[1] * config.num_scale_groups * token_scale_stages
    )

    @cute.struct
    class FP4SplitKVKernelSharedStorage:
        load_q_mbar_ptr: cute.struct.MemRange[Int64, config.load_q_stage * 2]
        load_k_mbar_ptr: cute.struct.MemRange[Int64, config.load_k_stage * 2]
        load_v_mbar_ptr: cute.struct.MemRange[Int64, config.load_v_stage * 2]
        mma_s_mbar_ptr: cute.struct.MemRange[Int64, config.mma_s_stage * 2]
        p_mma_mbar_ptr: cute.struct.MemRange[Int64, config.p_mma_stage * 2]
        p_cor_mbar_ptr: cute.struct.MemRange[Int64, config.p_cor_stage * 2]
        mma_o_mbar_ptr: cute.struct.MemRange[Int64, config.mma_o_stage * 2]
        load_scale_mbar_ptr: cute.struct.MemRange[Int64, token_scale_stages * 2]

        smem_p: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(p_smem_layout_staged)],
            align,
        ]
        smem_kc_latent: cute.struct.Align[
            cute.struct.MemRange[
                kv_smem_dtype, cute.cosize(kc_latent_smem_layout_staged)
            ],
            align,
        ]
        smem_kc_rope: cute.struct.Align[
            cute.struct.MemRange[
                rope_smem_dtype, cute.cosize(kc_rope_smem_layout_staged)
            ],
            align,
        ]
        smem_q_latent: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(q_latent_smem_layout_staged)],
            align,
        ]
        smem_q_rope: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(q_rope_smem_layout_staged)],
            align,
        ]
        smem_vc: cute.struct.Align[
            cute.struct.MemRange[kv_smem_dtype, cute.cosize(vc_smem_layout_staged)],
            align,
        ]
        smem_token_scale: cute.struct.MemRange[config.acc_dtype, token_scale_elements]
        softmax_smem_exchange: cute.struct.MemRange[
            config.acc_dtype, num_compute_warps * threads_per_warp
        ]
        epilogue_smem_exchange: cute.struct.MemRange[
            config.acc_dtype, num_compute_warps * threads_per_warp
        ]
        tmem_dealloc_mbar_ptr: Int64
        tmem_holding_buf: cutlass.Int32

        @classmethod
        def size_in_bytes(cls) -> int: ...  # noqa: F811

    return SimpleNamespace(
        qk_tiled_mma=qk_tiled_mma,
        pv_tiled_mma=pv_tiled_mma,
        q_latent_smem_layout_staged=q_latent_smem_layout_staged,
        q_rope_smem_layout_staged=q_rope_smem_layout_staged,
        kc_latent_smem_layout_staged=kc_latent_smem_layout_staged,
        kc_rope_smem_layout_staged=kc_rope_smem_layout_staged,
        p_smem_layout_staged=p_smem_layout_staged,
        vc_smem_layout_staged=vc_smem_layout_staged,
        kc_latent_smem_layout_for_tma=kc_latent_smem_layout_for_tma,
        kc_rope_smem_layout_for_tma=kc_rope_smem_layout_for_tma,
        vc_smem_layout_for_tma=vc_smem_layout_for_tma,
        tma_atom_q_latent=tma_atom_q_latent,
        tma_tensor_q_latent=tma_tensor_q_latent,
        tma_atom_q_rope=tma_atom_q_rope,
        tma_tensor_q_rope=tma_tensor_q_rope,
        tma_atom_c_latent=tma_atom_c_latent,
        tma_tensor_c_latent=tma_tensor_c_latent,
        tma_atom_c_rope=tma_atom_c_rope,
        tma_tensor_c_rope=tma_tensor_c_rope,
        tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
        tma_tensor_c_latent_transpose=tma_tensor_c_latent_transpose,
        tma_copy_q_bytes=tma_copy_q_bytes,
        tma_copy_kc_bytes=tma_copy_kc_bytes,
        tma_copy_vc_bytes=tma_copy_vc_bytes,
        SharedStorage=FP4SplitKVKernelSharedStorage,
        cta_layout_vmnk=cta_layout_vmnk,
    )


__all__ = ["build_mla_fp4_launch_params"]
