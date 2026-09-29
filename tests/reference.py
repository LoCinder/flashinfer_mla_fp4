"""Numerical references for shared-payload native FP4 MLA tests."""

from dataclasses import dataclass

import torch

LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fp4(x):
    """E2M1 round-to-nearest, ties-to-even; saturate to finite range."""
    levels = x.new_tensor(LEVELS)
    ax = x.abs()
    boundaries = x.new_tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    code = torch.bucketize(ax.contiguous(), boundaries)
    # At boundaries where the upper code is even, ties round upward.
    for i in (1, 3, 5):
        code = torch.where(ax == boundaries[i], i + 1, code)
    nibble = code | ((x < 0).long() << 3)
    packed = (nibble[..., ::2] | (nibble[..., 1::2] << 4)).to(torch.uint8)
    values = levels[code] * torch.where(x < 0, -1.0, 1.0)
    return packed.contiguous(), values


def unpack(packed):
    code = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    return packed.new_tensor(LEVELS, dtype=torch.float32)[code & 7] * torch.where(
        code & 8 != 0, -1.0, 1.0
    )


def fit_reference(x, payload, global_scale=1.):
    """FP32 numerical reference, independent of the streaming writer."""
    z = unpack(payload).reshape(*x.shape[:-1], 32, 16)
    blocks = x.float().reshape_as(z)
    energy = z.square().sum(-1)
    scale = (blocks * z).sum(-1) / energy.clamp_min(1e-20) / global_scale
    scale = torch.where(energy > 0, scale.clamp(0, 448), 0.)
    return scale.to(torch.float8_e4m3fn)


def fp8(x):
    return x.clamp(-448, 448).to(torch.float8_e4m3fn).float()


@dataclass
class Quantized:
    packed: torch.Tensor
    z: torch.Tensor
    sf: torch.Tensor
    global_scale: torch.Tensor
    reconstructed: torch.Tensor


def nvfp4(x, global_scale=None):
    """Linear block16 E4M3 scales + a separate dequantization FP32 scale.

    A supplied global scale is mandatory for streaming deployment; whole-input
    amax below is an offline reference convenience, never used to infer that an
    online writer can inspect future tokens. Production experiments use fixed g.
    """
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], -1, 16)
    if global_scale is None:
        global_scale = (blocks.abs().amax() / (6 * 448)).clamp_min(1e-12)
    g = torch.as_tensor(global_scale, device=x.device, dtype=torch.float32)
    sf = (blocks.abs().amax(-1) / (6 * g)).clamp(1 / 512, 448)
    sf = sf.to(torch.float8_e4m3fn)
    effective = sf.float() * g
    packed, z = fp4((blocks / effective[..., None]).reshape(shape))
    dequant = (z.reshape_as(blocks) * effective[..., None]).reshape(shape)
    return Quantized(packed, z, sf, g, dequant)


def online_pv(scores, z, a, tile_size=128):
    """Reproduce online softmax and the per-token-scaled FP8 numerator cast."""
    batch, heads, length = scores.shape
    if length % tile_size:
        raise ValueError("Pad/mask tails before calling this numerical reference")
    m = scores.new_full((batch, heads, 1), -torch.inf)
    denom = torch.zeros_like(m)
    accum = scores.new_zeros((batch, heads, 512))
    a = a.reshape(batch, length)
    for start in range(0, length, tile_size):
        tile = scores[..., start:start + tile_size]
        new_m = torch.maximum(m, tile.amax(-1, keepdim=True))
        old_correction = torch.exp(m - new_m)
        u = torch.exp(tile - new_m)
        denom = denom * old_correction + u.sum(-1, keepdim=True)
        u_scaled = fp8(u * a[:, None, start:start + tile_size])
        accum = accum * old_correction + u_scaled @ z[:, start:start + tile_size].float()
        m = new_m
    return accum / denom


def split_reference(scores, z, a, lengths, splits=1):
    """Mirror tile-aligned split-K and merge; padded KV never affects softmax."""
    output = []
    for batch, length in enumerate(lengths.tolist()):
        tiles = (length + 127) // 128
        tiles_per_split = (tiles + splits - 1) // splits
        pieces, weights = [], []
        for first in range(0, tiles, tiles_per_split):
            last = min(tiles, first + tiles_per_split)
            s = scores[batch:batch + 1, :, first * 128:last * 128].clone()
            if last * 128 > length:
                s[..., length - first * 128:] = -torch.inf
            part = online_pv(s, z[batch:batch + 1, first * 128:last * 128],
                             a[batch:batch + 1, first:last])
            pieces.append(part)
            weights.append(s.logsumexp(-1))
        mixing = torch.stack(weights).softmax(0)[..., None]
        output.append((torch.stack(pieces) * mixing).sum(0))
    return torch.cat(output)
