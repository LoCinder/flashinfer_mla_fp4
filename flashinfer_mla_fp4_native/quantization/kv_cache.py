"""Shared per-token E2M1 payload, unchanged V scale, independent fitted K scales.

The original source is available only while writing the cache. With z fixed,
least squares gives sum(x*z)/sum(z*z) per block16; nearest E4M3 rounding minimizes
that same objective over representable scales. No payload or PV scale refitting.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(code):
    c = code & 7
    magnitude = tl.where(c < 4, c.to(tl.float32) * .5,
                         tl.where(c == 4, 2., tl.where(c == 5, 3., tl.where(c == 6, 4., 6.))))
    return tl.where((code & 8) != 0, -magnitude, magnitude)


@triton.jit
def _write(X, PAYLOAD, TOKEN_SF, BLOCK_SF, GLOBAL: tl.constexpr,
           FUSED: tl.constexpr, PACKED_SF: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, 512)
    x = tl.load(X + row * 512 + d).to(tl.float32)
    if FUSED:
        # Exactly the original per-token writer, including its rounding mode.
        s = tl.max(tl.abs(x), 0) / 6
        s = tl.where(s > 0, s, 1.)
        tl.store(TOKEN_SF + row, s)
        lo, hi = tl.split(tl.reshape(x / s, (256, 2)))
        codes = tl.inline_asm_elementwise(
            "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
            constraints="=r,f,f", args=[lo, hi], dtype=tl.uint32, is_pure=True, pack=1,
        ).to(tl.uint8)
        tl.store(PAYLOAD + row * 256 + tl.arange(0, 256), codes)
    else:
        codes = tl.load(PAYLOAD + row * 256 + tl.arange(0, 256))
    z = tl.reshape(tl.interleave(_e2m1(codes), _e2m1(codes >> 4)), (32, 16))
    blocks = tl.reshape(x, (32, 16))
    energy = tl.sum(z * z, 1)
    cross = tl.sum(z * blocks, 1)
    ideal = tl.div_rn(cross, tl.maximum(energy, 1e-20))
    scale = tl.div_rn(ideal, GLOBAL)
    # Zero is a valid fitted scale. Unlike a payload quantizer, this path never
    # divides x by the rounded block scale, so underflow need not be clamped up.
    scale = tl.where(energy > 0, tl.minimum(tl.maximum(scale, 0.), 448.), 0.)
    scale = scale.to(tl.float8e4nv)
    block = tl.arange(0, 32)
    if PACKED_SF:
        # Page64 SFB layout, directly consumable by TMA or vector cp.async.
        offset = ((row // 64) * 2048 + (block // 8) * 512 + (row % 32) * 16
                  + ((block % 8) // 4) * 8 + ((row % 64) // 32) * 4 + block % 4)
    else:
        offset = row * 32 + block
    tl.store(BLOCK_SF + offset, scale)


def write_shared(x, *, global_scale=1., packed_sf=False, payload=None,
                 token_scale=None, block_scale=None, fused=True, warps=4):
    """Fused writer or scale-only fitter; preallocated outputs support graphs.

    packed_sf=True requires complete physical pages of 64 tokens. The opaque
    output keeps its linear shape but uses the page64 SFB byte permutation.
    """
    if x.shape[-1] != 512 or not x.is_contiguous() or not x.is_cuda:
        raise ValueError("Expected contiguous CUDA [..., 512] source")
    if not math.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("global_scale must be finite and positive")
    if packed_sf and x.numel() // 512 % 64:
        raise ValueError("Prepacked scales require complete physical pages")
    if not fused and (payload is None or token_scale is None):
        raise ValueError("Scale-only fitting requires existing payload and PV token scale")
    if payload is None:
        payload = torch.empty(*x.shape[:-1], 256, device=x.device, dtype=torch.uint8)
    if token_scale is None:
        token_scale = torch.empty(*x.shape[:-1], 1, device=x.device, dtype=torch.float32)
    if block_scale is None:
        block_scale = torch.empty(*x.shape[:-1], 32, device=x.device, dtype=torch.float8_e4m3fn)
    _write[(x.numel() // 512,)](x, payload, token_scale, block_scale, global_scale,
                               fused, packed_sf, num_warps=warps)
    return payload, token_scale, block_scale
