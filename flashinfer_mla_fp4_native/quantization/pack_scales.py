"""Q/K scale byte permutations; values and block assignments are unchanged."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_q(SRC, DST, HEADS: tl.constexpr):
    batch = tl.program_id(0)
    i = tl.program_id(1) * 1024 + tl.arange(0, 1024)
    cta, local = i // 4096, i % 4096
    atom = local // 512
    row, col = (local % 512) // 16, local % 16
    head = cta * 64 + row + (col // 4) * 32
    sf = tl.load(SRC + (batch * HEADS + head) * 32 + atom * 4 + col % 4,
                 (col < 8) & (head < HEADS), 0)
    tl.store(DST + batch * 8192 + i, sf)


def pack_q_scales(sf):
    """Padded CTA-local SFA layout; query preparation, not decode work."""
    output = torch.empty(sf.shape[0], 256, 32, dtype=sf.dtype, device=sf.device)
    _pack_q[(sf.shape[0], 8)](sf.view(torch.uint8), output.view(torch.uint8), sf.shape[1], num_warps=4)
    return output


@triton.jit
def _pack(SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    page = i // 2048
    stage = (i % 2048) // 512
    row = (i % 512) // 16
    col = i % 16
    token = row + ((col % 8) // 4) * 32
    scale = stage * 8 + (col // 8) * 4 + col % 4
    src = (page * 64 + token) * 32 + scale
    value = tl.load(SRC + src, i < N, 0)
    tl.store(DST + i, value, i < N)


def pack_k_scales(scales, out=None):
    if scales.shape[-2:] != (64, 32):
        raise ValueError("Prepacked SFB storage requires page size 64")
    if out is None:
        out = torch.empty_like(scales)
    _pack[(triton.cdiv(scales.numel(), 512),)](
        scales.view(torch.uint8), out.view(torch.uint8), scales.numel(), 512,
    )
    return out
