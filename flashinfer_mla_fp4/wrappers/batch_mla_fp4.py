# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch-facing wrapper for per-token-scaled FP4 MLA decode."""

import functools
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from flashinfer.api_logging import flashinfer_api
from flashinfer.cute_dsl.attention.config import AttentionFusion
from flashinfer.cute_dsl.attention.fusion.variant import (
    AttentionVariant,
    StandardAttention,
)
from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
    BatchMLADecodeCuteDSLWrapper,
    _check_can_implement,
    _get_split_kv_and_workspace_size,
    _make_mla_fake_tensors,
)
from flashinfer.cute_dsl.utils import (
    get_max_active_clusters,
    get_num_sm,
    require_cute_dsl_arch as _require_dsl_arch,
    torch_to_cutlass_dtype,
)
from flashinfer.utils import device_support_pdl

from ..mla_config_fp4 import MLAConfigFP4
from ..mla_decode_fp4 import BlackwellMultiLatentAttentionForwardFP4


def _make_mla_fp4_fake_tensors(
    cutlass_out_dtype,
    is_workspace_size_zero: bool,
    is_var_split_kv: bool,
    kv_lora_rank: int,
):
    """Create fake tensors matching the mixed packed-cache ABI."""

    (
        q_latent,
        q_rope,
        dense_c_latent,
        c_rope,
        page_table,
        output,
        lse,
        workspace,
        cache_seqs,
        block_split_kvs,
    ) = _make_mla_fake_tensors(
        cutlass.Float8E4M3FN,
        cutlass_out_dtype,
        is_workspace_size_zero,
        is_var_split_kv,
    )
    sym_kv_batch, sym_seq_kv, _ = dense_c_latent.shape
    c_latent = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (sym_kv_batch, sym_seq_kv, kv_lora_rank // 2),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    c_latent_sf = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (sym_kv_batch, sym_seq_kv, 1),
        stride=(cute.sym_int(), 1, 1),
        assumed_align=16,
    )
    return (
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        c_latent_sf,
        page_table,
        output,
        lse,
        workspace,
        cache_seqs,
        block_split_kvs,
    )


@functools.cache
def _compile_mla_fp4_kernel(
    torch_out_dtype: torch.dtype,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    skip_correction_threshold: float = 0.0,
    is_workspace_size_zero: bool = False,
    enable_pdl: bool = False,
    variant: Optional[AttentionVariant] = None,
    params_shape: Optional[tuple] = None,
) -> Callable:
    """Compile and cache one per-token-scaled FP4 MLA specialization."""

    if variant is None:
        variant = StandardAttention()
    fusion = AttentionFusion(variant=variant)
    cluster_shape_mnk = (2, 1, 1)
    config = MLAConfigFP4(
        latent_dim=kv_lora_rank,
        rope_dim=qk_rope_head_dim,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        max_active_clusters=get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        ),
        page_size=page_size,
        skip_correction_threshold=skip_correction_threshold,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        enable_pdl=enable_pdl,
    )
    kernel = BlackwellMultiLatentAttentionForwardFP4(config, fusion=fusion)
    (
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        c_latent_sf,
        page_table,
        output,
        lse,
        workspace,
        cache_seqs,
        block_split_kvs,
    ) = _make_mla_fp4_fake_tensors(
        torch_to_cutlass_dtype(torch_out_dtype),
        is_workspace_size_zero,
        is_var_split_kv,
        kv_lora_rank,
    )

    params = None
    if params_shape is not None:
        ndim = len(params_shape)
        params = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            params_shape,
            stride_order=tuple(range(ndim - 1, -1, -1)),
            assumed_align=16,
        )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel,
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        c_latent_sf,
        page_table,
        output,
        lse,
        workspace,
        Int32(1),
        cache_seqs,
        block_split_kvs,
        Float32(1.0),
        Float32(1.0),
        params,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


class BatchMLADecodeCuteDSLFP4Wrapper(BatchMLADecodeCuteDSLWrapper):
    """Stateful per-token-scaled FP4 MLA decode wrapper.

    The query and independent RoPE cache are E4M3. The 512-value latent cache
    is packed E2M1 and has one effective FP32 scale per token.
    """

    @flashinfer_api
    def plan(
        self,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        num_heads: int = 128,
        page_size: int = 1,
        q_dtype: torch.dtype = torch.float8_e4m3fn,
        out_dtype: Optional[torch.dtype] = None,
        is_var_seq: bool = True,
        enable_pdl: Optional[bool] = None,
        variant: Optional[AttentionVariant] = None,
        kv_dtype: torch.dtype = torch.uint8,
    ) -> None:
        """Compile the FP4 MLA kernel for the requested page layout."""

        _require_dsl_arch(self._device)
        if kv_lora_rank != 512:
            raise ValueError(
                "Per-token-scaled FP4 MLA requires kv_lora_rank=512, "
                f"got {kv_lora_rank}"
            )
        if qk_rope_head_dim != 64:
            raise ValueError(
                "Per-token-scaled FP4 MLA requires qk_rope_head_dim=64, "
                f"got {qk_rope_head_dim}"
            )
        if q_dtype != torch.float8_e4m3fn:
            raise ValueError("Per-token-scaled FP4 MLA requires an E4M3 query")
        if kv_dtype != torch.uint8:
            raise ValueError(
                "Per-token-scaled FP4 MLA requires a byte-packed uint8 KV cache"
            )

        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._num_heads = num_heads
        self._page_size = page_size
        self._q_dtype = q_dtype
        self._o_dtype = torch.bfloat16 if out_dtype is None else out_dtype
        self._is_var_seq = is_var_seq
        self._is_persistent = not is_var_seq
        self._is_var_split_kv = False
        self._skip_correction_threshold = 0.0
        self._enable_pdl = (
            device_support_pdl(self._device) if enable_pdl is None else enable_pdl
        )

        self._variant = StandardAttention() if variant is None else variant
        if self._variant.has_logits_transform:
            raise ValueError(
                "MLA decode does not support logits_transform. "
                "Use score_mod, update_statistics, or transform_output instead."
            )
        self._has_params = self._variant.extra_params is not None
        if self._has_params:
            params = self._variant.extra_params.to(torch.float32).to(self._device)
            if not params.is_contiguous():
                raise ValueError(
                    "AttentionVariant.extra_params must be contiguous, "
                    f"got strides {params.stride()} for shape {params.shape}"
                )
            self._params_torch = params
        else:
            self._params_torch = None

        _check_can_implement(
            torch_dtype=self._q_dtype,
            torch_out_dtype=self._o_dtype,
            page_size=self._page_size,
            num_heads=self._num_heads,
            seq_len_q=1,
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            is_persistent=self._is_persistent,
            is_var_seq=self._is_var_seq,
            is_var_split_kv=self._is_var_split_kv,
        )
        self._cache_variant = (
            None if isinstance(self._variant, StandardAttention) else self._variant
        )
        self._params_shape = (
            tuple(self._params_torch.shape) if self._has_params else None
        )
        self._compiled_kernel = _compile_mla_fp4_kernel(
            torch_out_dtype=self._o_dtype,
            page_size=self._page_size,
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            is_persistent=self._is_persistent,
            is_var_seq=self._is_var_seq,
            is_var_split_kv=self._is_var_split_kv,
            skip_correction_threshold=self._skip_correction_threshold,
            is_workspace_size_zero=False,
            enable_pdl=self._enable_pdl,
            variant=self._cache_variant,
            params_shape=self._params_shape,
        )

    def _validate_run_inputs(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_cache_sf: Optional[torch.Tensor],
        out: Optional[torch.Tensor],
    ) -> None:
        if q.ndim != 4:
            raise ValueError("q must have shape [batch, q_len, num_heads, head_dim]")
        if q.shape[1] != 1:
            raise ValueError(f"q_len must be 1, got {q.shape[1]}")
        expected_dim = self._kv_lora_rank + self._qk_rope_head_dim
        if q.shape[-1] != expected_dim or q.shape[2] != self._num_heads:
            raise ValueError(
                f"q must end in [{self._num_heads}, {expected_dim}], "
                f"got shape {tuple(q.shape)}"
            )
        if q.dtype != self._q_dtype:
            raise ValueError(
                f"q.dtype={q.dtype} does not match planned {self._q_dtype}"
            )
        if kv_cache.dtype != torch.uint8:
            raise ValueError("FP4 kv_cache must have dtype torch.uint8")
        if kv_cache_sf is None:
            raise ValueError("FP4 MLA requires kv_cache_sf")
        if kv_cache_sf.dtype != torch.float32:
            raise ValueError("kv_cache_sf must contain FP32 per-token scales")
        if out is not None and out.dtype != self._o_dtype:
            raise ValueError(
                f"out.dtype={out.dtype} does not match planned {self._o_dtype}"
            )

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        softmax_scale: float,
        output_scale: float = 1.0,
        out: Optional[torch.Tensor] = None,
        kv_cache_sf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run per-token-scaled FP4 MLA with the dense wrapper plan/run shape."""

        if self._compiled_kernel is None:
            raise RuntimeError("Call plan() before run().")
        self._validate_run_inputs(q, kv_cache, kv_cache_sf, out)

        if kv_cache.dim() == 4:
            if kv_cache.shape[1] != 1:
                raise ValueError(
                    "4D kv_cache must have shape [pages, 1, page_size, dim]"
                )
            kv_cache = kv_cache.squeeze(1)
        elif kv_cache.dim() != 3:
            raise ValueError("kv_cache must be 3D or 4D")
        if kv_cache.shape[1] != self._page_size:
            raise ValueError(
                f"kv_cache page size {kv_cache.shape[1]} does not match "
                f"planned {self._page_size}"
            )

        if kv_cache_sf.dim() == 4:
            if kv_cache_sf.shape[1] != 1:
                raise ValueError(
                    "4D kv_cache_sf must have shape [pages, 1, page_size, 1]"
                )
            kv_cache_sf = kv_cache_sf.squeeze(1)
        elif kv_cache_sf.dim() != 3:
            raise ValueError("kv_cache_sf must be 3D or 4D")

        expected_storage_dim = self._kv_lora_rank // 2 + self._qk_rope_head_dim
        if kv_cache.shape[-1] != expected_storage_dim:
            raise ValueError(
                f"kv_cache.shape[-1]={kv_cache.shape[-1]}, "
                f"expected {expected_storage_dim}"
            )
        expected_sf_shape = (*kv_cache.shape[:2], 1)
        if tuple(kv_cache_sf.shape) != expected_sf_shape:
            raise ValueError(
                f"kv_cache_sf must have shape {expected_sf_shape}, "
                f"got {tuple(kv_cache_sf.shape)}"
            )
        if q.device != kv_cache.device or q.device != kv_cache_sf.device:
            raise ValueError("q, kv_cache, and kv_cache_sf must share a device")
        if not kv_cache.is_contiguous() or not kv_cache_sf.is_contiguous():
            raise ValueError("kv_cache and kv_cache_sf must be contiguous")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        batch_size, q_len, num_heads, _ = q.shape
        q_latent = q[..., : self._kv_lora_rank]
        q_rope = q[..., self._kv_lora_rank :]
        latent_packed = self._kv_lora_rank // 2
        kv_cache_u8 = kv_cache.view(torch.uint8)
        c_latent = kv_cache_u8[..., :latent_packed]
        c_rope = kv_cache_u8[..., latent_packed:].view(torch.float8_e4m3fn)
        c_latent_sf = kv_cache_sf

        split_kv, workspace_size = _get_split_kv_and_workspace_size(
            batch_size,
            q_len,
            num_heads,
            self._kv_lora_rank,
            get_num_sm(q.device),
        )
        if workspace_size == 0:
            workspace = None
            compiled_kernel = _compile_mla_fp4_kernel(
                torch_out_dtype=self._o_dtype,
                page_size=self._page_size,
                kv_lora_rank=self._kv_lora_rank,
                qk_rope_head_dim=self._qk_rope_head_dim,
                is_persistent=self._is_persistent,
                is_var_seq=self._is_var_seq,
                is_var_split_kv=self._is_var_split_kv,
                skip_correction_threshold=self._skip_correction_threshold,
                is_workspace_size_zero=True,
                enable_pdl=self._enable_pdl,
                variant=self._cache_variant,
                params_shape=self._params_shape,
            )
        else:
            if self._workspace_buffer.numel() < workspace_size:
                raise ValueError(
                    f"workspace_buffer has {self._workspace_buffer.numel()} bytes, "
                    f"but {workspace_size} are required"
                )
            workspace = self._workspace_buffer[:workspace_size]
            compiled_kernel = self._compiled_kernel

        if out is None:
            out = torch.empty(
                (
                    batch_size,
                    q_len,
                    num_heads,
                    self._kv_lora_rank,
                ),
                dtype=self._o_dtype,
                device=q.device,
            )
        lse = torch.empty(
            (batch_size, q_len, num_heads),
            dtype=torch.float32,
            device=q.device,
        )
        cache_seqs = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        compiled_kernel(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            c_latent_sf,
            block_tables,
            out,
            lse,
            workspace,
            Int32(split_kv),
            cache_seqs,
            None,
            Float32(softmax_scale),
            Float32(output_scale),
            self._params_torch if self._has_params else None,
        )
        return out


__all__ = ["BatchMLADecodeCuteDSLFP4Wrapper"]
