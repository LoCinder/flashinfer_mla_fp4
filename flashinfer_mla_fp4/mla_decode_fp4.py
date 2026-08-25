# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-token-scaled FP4 MLA decode kernel for Blackwell."""

from types import SimpleNamespace
from typing import Optional, Type, cast

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from flashinfer.cute_dsl.attention.compat import (
    get_max_tmem_alloc_cols as _get_max_tmem_alloc_cols,
    setmaxregister_decrease as _setmaxregister_decrease,
    setmaxregister_increase as _setmaxregister_increase,
)
from flashinfer.cute_dsl.attention.config import AttentionFusion
from flashinfer.cute_dsl.attention.mla_decode_fp8 import (
    BlackwellMultiLatentAttentionForwardFP8,
)
from flashinfer.cute_dsl.attention.mla_warp_schedule import MLAWarpScheduleFP8
from flashinfer.cute_dsl.attention.scheduler.mla_persistent import (
    LOG2_E,
    MAX_SPLITS,
    MLAStaticTileSchedulerParams,
)

from .collective_builder_fp4 import build_mla_fp4_launch_params
from .mla_config_fp4 import (
    MLAConfigFP4,
    MLA_DECODE_FP4_SCHEDULE,
    make_mla_fp4_mainloop_spec,
)
from .roles.mla_compute_fp4 import MLAComputeFP4Role
from .roles.mla_correction_fp4 import MLACorrectionFP4Role
from .roles.mla_loader_fp4 import MLAFP4LoaderKRole, MLAFP4LoaderVRole
from .roles.mla_mma_fp4 import MLAMmaFP4Role


class BlackwellMultiLatentAttentionForwardFP4(BlackwellMultiLatentAttentionForwardFP8):
    """Mixed E4M3-query/E2M1-latent MLA with one FP32 scale per token."""

    def __init__(
        self,
        config: MLAConfigFP4,
        fusion: AttentionFusion | None = None,
        schedule: MLAWarpScheduleFP8 | None = None,
    ):
        selected_schedule = (
            schedule if schedule is not None else MLA_DECODE_FP4_SCHEDULE
        )
        super().__init__(config, fusion=fusion, schedule=selected_schedule)
        self.mainloop = make_mla_fp4_mainloop_spec(config, selected_schedule)

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        c_latent_sf: Optional[cute.Tensor],
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        params_in: Optional[cute.Tensor],
        stream,
    ):
        self.q_dtype: Type[cutlass.Numeric] = q_latent.element_type
        self.k_storage_dtype: Type[cutlass.Numeric] = c_latent.element_type
        self.k_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN
        self.v_dtype: Type[cutlass.Numeric] = self.k_dtype
        self.o_dtype: Type[cutlass.Numeric] = o.element_type

        if cutlass.const_expr(self.q_dtype != cutlass.Float8E4M3FN):
            raise TypeError("Per-token-scaled FP4 MLA requires an E4M3 query")
        if cutlass.const_expr(self.k_storage_dtype != cutlass.Uint8):
            raise TypeError("Per-token-scaled FP4 MLA requires byte-packed latent data")
        if cutlass.const_expr(c_rope.element_type != cutlass.Float8E4M3FN):
            raise TypeError(
                "Per-token-scaled FP4 MLA requires independent E4M3 K-RoPE"
            )
        if cutlass.const_expr(
            c_latent_sf is None or c_latent_sf.element_type != cutlass.Float32
        ):
            raise TypeError("Per-token-scaled FP4 MLA requires FP32 scales")

        def _reinterpret_4d(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[2], t.shape[3], t.shape[1], t.shape[0]),
                    stride=(t.stride[2], t.stride[3], t.stride[1], t.stride[0]),
                ),
            )

        q_latent = _reinterpret_4d(q_latent)
        q_rope = _reinterpret_4d(q_rope)
        o = _reinterpret_4d(o)

        def _reinterpret_3d_kv(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[1], t.shape[2], t.shape[0]),
                    stride=(t.stride[1], t.stride[2], t.stride[0]),
                ),
            )

        def _reinterpret_3d_fp4(t):
            fp4_ptr = cute.recast_ptr(t.iterator, dtype=cutlass.Float4E2M1FN)
            return cute.make_tensor(
                fp4_ptr,
                cute.make_layout(
                    (t.shape[1], t.shape[2] * 2, t.shape[0]),
                    stride=(
                        t.stride[1] * 2,
                        1,
                        t.stride[0] * 2,
                    ),
                ),
            )

        c_latent = _reinterpret_3d_fp4(c_latent)
        c_rope = _reinterpret_3d_kv(c_rope)
        c_latent_sf = cute.make_tensor(
            c_latent_sf.iterator,
            cute.make_layout(
                (
                    c_latent_sf.shape[1],
                    c_latent_sf.shape[2],
                    c_latent_sf.shape[0],
                ),
                stride=(
                    c_latent_sf.stride[1],
                    c_latent_sf.stride[2],
                    c_latent_sf.stride[0],
                ),
            ),
        )

        page_table = cute.make_tensor(
            page_table.iterator,
            cute.make_layout(
                (page_table.shape[1], page_table.shape[0]),
                stride=(page_table.stride[1], page_table.stride[0]),
            ),
        )

        lse = cute.make_tensor(
            lse.iterator,
            cute.make_layout(
                (lse.shape[2], lse.shape[1], lse.shape[0]),
                stride=(lse.stride[2], lse.stride[1], lse.stride[0]),
            ),
        )

        acc_o, acc_lse = self.initialize_workspace(
            q_latent.shape[0],
            q_latent.shape[1],
            q_latent.shape[2],
            q_latent.shape[3],
            split_kv,
            self.config.acc_dtype,
            workspace,
        )

        c_latent_transpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_transpose_layout
        )

        self.mainloop = self.mainloop.resolve(self.q_dtype.width)

        params = (
            cute.make_tensor(
                params_in.iterator,
                cute.make_layout(
                    self.fusion.params_shape,
                    stride=self.fusion.params_strides,
                ),
            )
            if cutlass.const_expr(self.fusion.has_params)
            else None
        )

        self.loader_k_role = MLAFP4LoaderKRole(self.config)
        self.loader_v_role = MLAFP4LoaderVRole(self.config)
        self.mma_role = MLAMmaFP4Role(self.config, self.mainloop)
        self.mma_role.set_dtypes(self.q_dtype, self.k_dtype, self.v_dtype)
        self.compute_role = MLAComputeFP4Role(self.config, fusion=self.fusion)
        self.compute_role.set_dtypes(self.q_dtype)
        self.compute_role.set_barriers(self.softmax_exchange_sync_bar)
        self.correction_role = MLACorrectionFP4Role(
            self.config,
            fusion=self.fusion,
            p_dtype=self.q_dtype,
            v_dtype=self.v_dtype,
            o_dtype=self.o_dtype,
        )
        self.correction_role.set_barriers(self.epilogue_exchange_sync_bar)

        lp = build_mla_fp4_launch_params(
            self.mainloop,
            self.schedule,
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            c_latent_transpose,
            self.q_dtype,
            self.k_dtype,
            self.v_dtype,
        )
        self.tma_copy_q_bytes = lp.tma_copy_q_bytes
        self.tma_copy_kc_bytes = lp.tma_copy_kc_bytes
        self.tma_copy_vc_bytes = lp.tma_copy_vc_bytes

        tile_sched_params, grid = self._compute_grid(
            o,
            split_kv,
            self.config.cluster_shape_mnk,
            self.config.max_active_clusters,
            self.config.is_persistent,
        )

        softmax_scale_log2 = softmax_scale * LOG2_E
        self.split_kv_kernel(
            lp.qk_tiled_mma,
            lp.pv_tiled_mma,
            lp.tma_atom_q_latent,
            lp.tma_tensor_q_latent,
            lp.tma_atom_q_rope,
            lp.tma_tensor_q_rope,
            lp.tma_atom_c_latent,
            lp.tma_tensor_c_latent,
            lp.tma_atom_c_rope,
            lp.tma_tensor_c_rope,
            lp.tma_atom_c_latent_transpose,
            lp.tma_tensor_c_latent_transpose,
            c_latent_sf,
            page_table,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            lp.q_latent_smem_layout_staged,
            lp.q_rope_smem_layout_staged,
            lp.kc_latent_smem_layout_staged,
            lp.kc_rope_smem_layout_staged,
            lp.p_smem_layout_staged,
            lp.vc_smem_layout_staged,
            lp.kc_latent_smem_layout_for_tma,
            lp.kc_rope_smem_layout_for_tma,
            lp.vc_smem_layout_for_tma,
            lp.cta_layout_vmnk,
            tile_sched_params,
            lp.SharedStorage,
            params,
        ).launch(
            grid=grid,
            block=[self.schedule.threads_per_cta, 1, 1],
            cluster=self.config.cluster_shape_mnk,
            smem=lp.SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.config.enable_pdl,
        )
        if cutlass.const_expr(acc_o is not None):
            self.reduction_kernel(
                o,
                lse,
                acc_o,
                acc_lse,
                split_kv,
                cache_seqs,
                block_split_kvs,
            ).launch(
                grid=(q_latent.shape[0], q_latent.shape[2], q_latent.shape[3]),
                block=[
                    self.schedule.threads_per_warp * self.config.num_compute_warps,
                    1,
                    1,
                ],
                smem=MAX_SPLITS * self.config.acc_dtype.width // 8,
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=self.config.enable_pdl,
            )

    @cute.kernel
    def split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_q_rope: Optional[cute.CopyAtom],
        mQR: cute.Tensor,
        tma_atom_c_latent: Optional[cute.CopyAtom],
        mCL: cute.Tensor,
        tma_atom_c_rope: Optional[cute.CopyAtom],
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: Optional[cute.CopyAtom],
        mCLT: cute.Tensor,
        mCLSF: Optional[cute.Tensor],
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        q_latent_smem_layout_staged: cute.ComposedLayout,
        q_rope_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_staged: cute.ComposedLayout,
        kc_rope_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_for_tma: cute.ComposedLayout,
        kc_rope_smem_layout_for_tma: cute.ComposedLayout,
        vc_smem_layout_for_tma: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: MLAStaticTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
        params: Optional[cute.Tensor] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        if warp_idx == self.schedule.mma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_q_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent)
            cpasync.prefetch_descriptor(tma_atom_c_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_bar,
            allocator_warp_id=self.schedule.mma_warp_id,
            is_two_cta=self.config.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        pipes = self._create_pipelines(storage, cta_layout_vmnk)
        load_q_prod, load_q_cons = pipes["load_q"]
        load_k_prod, load_k_cons = pipes["load_k"]
        load_v_prod, load_v_cons = pipes["load_v"]
        mma_s_prod, mma_s_cons = pipes["mma_s"]
        p_mma_prod, p_mma_cons = pipes["p_mma"]
        p_cor_prod, p_cor_cons = pipes["p_cor"]
        mma_o_prod, mma_o_cons = pipes["mma_o"]
        load_scale_prod, load_scale_cons = pipes["load_scale"]

        pipeline_init_arrive(
            cluster_shape_mn=self.config.cluster_shape_mnk,
            is_relaxed=True,
        )

        # SMEM tensor views
        sQ = storage.smem_q_latent.get_tensor(
            q_latent_smem_layout_staged.outer,
            swizzle=q_latent_smem_layout_staged.inner,
        )
        sQ_rope = storage.smem_q_rope.get_tensor(
            q_rope_smem_layout_staged.outer,
            swizzle=q_rope_smem_layout_staged.inner,
        )
        sKC = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_staged.outer,
            swizzle=kc_latent_smem_layout_staged.inner,
        )
        sKC_rope = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_staged.outer,
            swizzle=kc_rope_smem_layout_staged.inner,
        )
        sKC_for_tma = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_for_tma.outer,
            swizzle=kc_latent_smem_layout_for_tma.inner,
        )
        sKC_rope_for_tma = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_for_tma.outer,
            swizzle=kc_rope_smem_layout_for_tma.inner,
        )
        sVC = storage.smem_vc.get_tensor(
            vc_smem_layout_staged.outer,
            swizzle=vc_smem_layout_staged.inner,
        )
        sVC_for_tma = storage.smem_vc.get_tensor(
            vc_smem_layout_for_tma.outer,
            swizzle=vc_smem_layout_for_tma.inner,
        )
        sP = storage.smem_p.get_tensor(
            p_smem_layout_staged.outer,
            swizzle=p_smem_layout_staged.inner,
        )
        token_scale_stages = self.config.mma_s_stage
        token_scale_shape = (
            self.config.mma_qk_tiler[1],
            self.config.num_scale_groups,
            token_scale_stages,
        )
        token_scale_stride = (
            self.config.num_scale_groups,
            1,
            self.config.mma_qk_tiler[1] * self.config.num_scale_groups,
        )
        sTokenScale = storage.smem_token_scale.get_tensor(
            cute.make_layout(token_scale_shape, stride=token_scale_stride)
        )
        softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
            cute.make_layout(
                self.config.num_compute_warps * self.schedule.threads_per_warp
            )
        )
        epilogue_smem_exchange = storage.epilogue_smem_exchange.get_tensor(
            cute.make_layout(
                self.config.num_compute_warps * self.schedule.threads_per_warp
            )
        )

        pipeline_init_wait(cluster_shape_mn=self.config.cluster_shape_mnk)

        if cutlass.const_expr(self.config.enable_pdl):
            cute.arch.griddepcontrol_wait()

        # /////////////////////////////////////////////////////////////////////
        #  Empty warps
        # /////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(len(self.schedule.empty_warp_ids) > 0):
            if (
                warp_idx >= self.schedule.empty_warp_ids[0]
                and warp_idx <= self.schedule.empty_warp_ids[-1]
            ):
                _setmaxregister_decrease(self.schedule.other_reg_num)

        # /////////////////////////////////////////////////////////////////////
        #  Dense-TMA loader topology moves sub-byte Float4 payloads directly
        #  into SMEM for the FP4 path.
        # /////////////////////////////////////////////////////////////////////
        fp8_schedule = cast(MLAWarpScheduleFP8, self.schedule)
        if warp_idx == fp8_schedule.load_tma_k_warp_id:
            _setmaxregister_decrease(self.schedule.other_reg_num)
            tma_common_params = SimpleNamespace(mPT=mPT)
            tma_qk_params = SimpleNamespace(
                tiled_mma_qk=tiled_mma_qk,
                tma_atom_q_latent=tma_atom_q_latent,
                tma_atom_q_rope=tma_atom_q_rope,
                tma_atom_c_latent=tma_atom_c_latent,
                tma_atom_c_rope=tma_atom_c_rope,
                mQL=mQL,
                mQR=mQR,
                mCL=mCL,
                mKR=mKR,
                sQ=sQ,
                sQ_rope=sQ_rope,
                sKC=sKC_for_tma,
                sKC_rope=sKC_rope_for_tma,
                mTokenSF=mCLSF,
                sTokenScale=sTokenScale,
            )
            self.loader_k_role.run(
                tma_common_params,
                tma_qk_params,
                split_kv,
                cache_seqs,
                block_split_kvs,
                load_q_prod,
                load_k_prod,
                load_scale_prod,
                tile_sched_params,
            )

        if warp_idx == fp8_schedule.load_tma_v_warp_id:
            _setmaxregister_decrease(self.schedule.other_reg_num)
            tma_common_params = SimpleNamespace(mPT=mPT)
            tma_v_params = SimpleNamespace(
                tiled_mma_pv=tiled_mma_pv,
                tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                mCLT=mCLT,
                sVC=sVC_for_tma,
            )
            self.loader_v_role.run(
                tma_common_params,
                tma_v_params,
                split_kv,
                cache_seqs,
                block_split_kvs,
                load_v_prod,
                tile_sched_params,
            )

        # /////////////////////////////////////////////////////////////////////
        #  MMA warp
        # /////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.mma_warp_id:
            _setmaxregister_decrease(self.schedule.other_reg_num)
            tmem.allocate(_get_max_tmem_alloc_cols(self.arch))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.config.acc_dtype)

            self.mma_role.run(
                tiled_mma_qk,
                tiled_mma_pv,
                load_q_cons,
                load_k_cons,
                load_v_cons,
                mma_s_prod,
                p_mma_cons,
                mma_o_prod,
                split_kv,
                cache_seqs,
                block_split_kvs,
                tile_sched_params,
                sQ,
                sQ_rope,
                sKC,
                sKC_rope,
                sP,
                sVC,
                tmem_ptr,
                is_leader_cta,
                mCL.shape[1],
            )

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
            if cutlass.const_expr(self.config.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        # /////////////////////////////////////////////////////////////////////
        #  Compute (softmax) warps
        # /////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.schedule.compute_warp_ids[0]
            and warp_idx <= self.schedule.compute_warp_ids[-1]
        ):
            # Fresh TiledMma avoids SSA dominance conflict with the MMA warp's
            # .set(ACCUMULATE) mutations on the same tiled_mma_qk variable.
            compute_tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
                self.q_dtype,
                self.k_dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                self.config.acc_dtype,
                tcgen05.CtaGroup.TWO,
                self.config.mma_qk_tiler[:2],
            )
            self.compute_role.run(
                split_kv,
                cache_seqs,
                block_split_kvs,
                tile_sched_params,
                tmem_ptr=None,
                mma_s_consumer=mma_s_cons,
                load_scale_consumer=load_scale_cons,
                p_mma_producer=p_mma_prod,
                p_cor_producer=p_cor_prod,
                softmax_smem_exchange=softmax_smem_exchange,
                mAccO=mAccO,
                mO=mO,
                mCL=mCL,
                sTokenScale=sTokenScale,
                K=None,
                L=mCL.shape[1],
                tiled_mma_qk=compute_tiled_mma_qk,
                sP=sP,
                softmax_scale_log2=softmax_scale_log2,
                tmem=tmem,
                params=params,
            )

        # /////////////////////////////////////////////////////////////////////
        #  Correction (rescale + epilogue) warps
        # /////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.schedule.correction_warp_ids[0]
            and warp_idx <= self.schedule.correction_warp_ids[-1]
        ):
            _setmaxregister_increase(self.schedule.correction_reg_num)
            tmem.wait_for_alloc()
            tmem_ptr_corr = tmem.retrieve_ptr(self.config.acc_dtype)

            cta_m_offset = (bidx % cute.size(tiled_mma_qk.thr_id.shape)) * (
                self.config.mma_qk_tiler[0] // self.config.cluster_shape_mnk[0]
            )
            corr_common_params = SimpleNamespace(
                smem_exchange=epilogue_smem_exchange,
                mAccO=mAccO,
                mO=mO,
                L=mCL.shape[1],
                H=mQL.shape[0],
                cta_m_offset=cta_m_offset,
            )
            corr_epilogue_params = SimpleNamespace(
                output_scale=output_scale,
                softmax_scale_log2=softmax_scale_log2,
                mAccLSE=mAccLSE,
                mLSE=mLSE,
            )
            self.correction_role.run(
                split_kv,
                cache_seqs,
                block_split_kvs,
                tile_sched_params,
                tmem_ptr_corr,
                p_cor_consumer=p_cor_cons,
                mma_o_consumer=mma_o_cons,
                compute_common_params=corr_common_params,
                epilogue_params=corr_epilogue_params,
                params=params,
            )

        return


__all__ = ["BlackwellMultiLatentAttentionForwardFP4"]
