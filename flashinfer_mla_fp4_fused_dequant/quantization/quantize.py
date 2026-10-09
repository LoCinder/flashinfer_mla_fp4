# SPDX-License-Identifier: BSD-3-Clause
"""Prepare block16 E2M1 payloads and E4M3 scales, with global scale fixed to 1."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, PACKED, SF):
    row = tl.program_id(0)
    d = tl.arange(0, 512)
    x = tl.load(X + row * 512 + d).to(tl.float32)
    blocks = tl.reshape(x, (32, 16))
    s = tl.maximum(tl.minimum(tl.div_rn(tl.max(tl.abs(blocks), 1), 6.), 448.), 1. / 512)
    s8 = s.to(tl.float8e4nv)
    tl.store(SF + row * 32 + tl.arange(0, 32), s8)
    normalized = tl.reshape(tl.div_rn(blocks, s8.to(tl.float32)[:, None]), (512,))
    pairs = tl.reshape(normalized, (256, 2))
    lo, hi = tl.split(pairs)
    codes = tl.inline_asm_elementwise(
        "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
        constraints="=r,f,f", args=[lo, hi], dtype=tl.uint32, is_pure=True, pack=1,
    ).to(tl.uint8)
    tl.store(PACKED + row * 256 + tl.arange(0, 256), codes)


def quantize(x, *, output=None):
    if x.shape[-1] != 512 or not x.is_contiguous() or not x.is_cuda:
        raise ValueError("Expected contiguous CUDA [..., 512] input")
    if output is None:
        packed = torch.empty(*x.shape[:-1], 256, device=x.device, dtype=torch.uint8)
        scales = torch.empty(*x.shape[:-1], 32, device=x.device,
                             dtype=torch.float8_e4m3fn)
    else:
        packed, scales = output
    _quantize[(x.numel() // 512,)](x, packed, scales, num_warps=4)
    return packed, scales
