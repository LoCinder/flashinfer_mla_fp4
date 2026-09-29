"""Small streaming-writer controls: BF16/FP32 to E2M1 and linear scales.

One CTA handles one 512-dimensional row. Both formats use the same launch
structure, making write-cost comparisons independent of Python reference code.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, PACKED, SF, NV: tl.constexpr, GLOBAL: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, 512)
    x = tl.load(X + row * 512 + d).to(tl.float32)
    if NV:
        blocks = tl.reshape(x, (32, 16))
        s = tl.maximum(tl.minimum(tl.div_rn(tl.max(tl.abs(blocks), 1), 6 * GLOBAL), 448.), 1. / 512)
        s8 = s.to(tl.float8e4nv)
        tl.store(SF + row * 32 + tl.arange(0, 32), s8)
        normalized = tl.reshape(tl.div_rn(blocks, s8.to(tl.float32)[:, None] * GLOBAL), (512,))
    else:
        s = tl.max(tl.abs(x), 0) / 6
        s = tl.where(s > 0, s, 1.)
        tl.store(SF + row, s)
        normalized = x / s
    pairs = tl.reshape(normalized, (256, 2))
    lo, hi = tl.split(pairs)
    codes = tl.inline_asm_elementwise(
        "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
        constraints="=r,f,f", args=[lo, hi], dtype=tl.uint32, is_pure=True, pack=1,
    ).to(tl.uint8)
    tl.store(PACKED + row * 256 + tl.arange(0, 256), codes)


def quantize(x, *, native=True, global_scale=1., output=None):
    if x.shape[-1] != 512 or not x.is_contiguous():
        raise ValueError("Expected contiguous [..., 512] input")
    if output is None:
        packed = torch.empty(*x.shape[:-1], 256, device=x.device, dtype=torch.uint8)
        scales = torch.empty(*x.shape[:-1], 32 if native else 1, device=x.device,
                             dtype=torch.float8_e4m3fn if native else torch.float32)
    else:
        packed, scales = output
    _quantize[(x.numel() // 512,)](x, packed, scales, native, global_scale, num_warps=4)
    return packed, scales
