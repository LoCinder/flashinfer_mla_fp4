"""Operator-only adapter for the shared-payload / dual-scale MLA kernel."""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from flashinfer.cute_dsl.attention.config import AttentionFusion
from flashinfer.cute_dsl.attention.scheduler.mla_persistent import MAX_SPLITS
from flashinfer.cute_dsl.utils import get_max_active_clusters
from ..mla_config_fp4 import MLAConfigFP4, MLA_DECODE_FP4_SCHEDULE
from ..mla_decode_fp4 import BlackwellMultiLatentAttentionForwardFP4


@functools.cache
def compile_kernel(heads=64, split=True):
    page = 64
    config = MLAConfigFP4(
        latent_dim=512, rope_dim=64, acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32, mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256), max_active_clusters=get_max_active_clusters(2),
        page_size=page, is_persistent=True, is_var_seq=False,
        # Let the reduction prologue overlap with the split-K main-kernel tail.
        is_var_split_kv=False, enable_pdl=split,
    )
    schedule = MLA_DECODE_FP4_SCHEDULE
    kernel = BlackwellMultiLatentAttentionForwardFP4(config, AttentionFusion(), schedule)
    batch, pages, table_pages, worksize = (cute.sym_int() for _ in range(4))
    qlen, nheads = cute.sym_int(), cute.sym_int()

    def tensor(dtype, shape):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16,
        )

    q = tensor(cutlass.Uint8, (batch, qlen, nheads, 256))
    qr = tensor(cutlass.Float8E4M3FN, (batch, qlen, nheads, 64))
    k = tensor(cutlass.Uint8, (pages, page, 256))
    r = tensor(cutlass.Float8E4M3FN, (pages, page, 64))
    a = tensor(cutlass.Float32, (pages, page, 1))
    pt = tensor(cutlass.Int32, (batch, table_pages))
    o = tensor(cutlass.BFloat16, (batch, qlen, nheads, 512))
    lse = tensor(cutlass.Float32, (batch, qlen, nheads))
    workspace = tensor(cutlass.Int8, (worksize,)) if split else None
    lengths = tensor(cutlass.Int32, (batch,))
    qsf = tensor(cutlass.Float8E4M3FN, (batch, 256, 32))
    ksf = tensor(cutlass.Float8E4M3FN, (pages, page, 32))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel, q, qr, k, r, a, pt, o, lse, workspace, Int32(1), lengths,
        None, Float32(1.), Float32(1.), None, stream, qsf, ksf, Float32(1.),
        options="--enable-tvm-ffi --opt-level 2",
    )


class Runner:
    def __init__(self, q, qs, qr, k, ks, rope, a, lengths, splits=16, page=64,
                 q_global=1., kv_global=1., input_sf_packed=False):
        batch, heads = q.shape[:2]
        seq = k.shape[1]
        if heads not in (8, 16, 32, 64, 128) or page != 64:
            raise ValueError("Optimized scope: H=8/16/32/64/128, page size=64")
        if q.shape != (batch, heads, 256) or k.shape != (batch, seq, 256):
            raise ValueError("Expected one query position and packed 512-value latent")
        if splits < 1 or splits > MAX_SPLITS or batch < 1 or seq % page:
            raise ValueError("Invalid batch, page padding, or split count")
        if q_global * kv_global != 1.:
            raise ValueError("Fused latent/RoPE accumulation requires g_Q * g_K == 1")
        self.compiled = compile_kernel(heads, splits > 1)
        self.q = q.view(batch, 1, heads, 256)
        self.qr = qr.view(batch, 1, heads, 64)
        self.k = k.view(-1, page, 256)
        self.r = rope.view(-1, page, 64)
        self.a = a.reshape(-1, page, 1).contiguous()
        self.qs, self.ks = qs.contiguous(), ks.reshape(-1, page, 32).contiguous()
        from ..quantization.pack_scales import pack_q_scales
        self.qs = pack_q_scales(self.qs)
        if not input_sf_packed:
            from ..quantization.pack_scales import pack_k_scales
            self.ks = pack_k_scales(self.ks)
        self.tables = torch.arange(batch * seq // page, device=q.device,
                                   dtype=torch.int32).view(batch, seq // page)
        self.out = torch.empty(batch, 1, heads, 512, device=q.device, dtype=torch.bfloat16)
        self.lse = torch.empty(batch, 1, heads, device=q.device, dtype=torch.float32)
        workspace_size = BlackwellMultiLatentAttentionForwardFP4.get_workspace_size(
            heads, 1, 512, batch, splits, cutlass.Float32,
        )
        self.workspace = torch.empty(workspace_size, device=q.device, dtype=torch.int8) if splits > 1 else None
        self.lengths, self.splits = lengths, splits
        self.qk_global = q_global * kv_global

    def __call__(self):
        self.compiled(
            self.q, self.qr, self.k, self.r, self.a, self.tables, self.out, self.lse,
            self.workspace, self.splits, self.lengths, None, 512**-.5, 1., None,
            self.qs, self.ks, self.qk_global,
        )
        return self.out[:, 0]
