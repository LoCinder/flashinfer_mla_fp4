"""Independent NVFP4 quantization and fused FP8 PV reference."""

from dataclasses import dataclass

import torch


def unpack(packed):
    code = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    levels = packed.new_tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), dtype=torch.float32)
    return levels[code & 7] * torch.where(code & 8 != 0, -1., 1.)


@dataclass
class Quantized:
    packed: torch.Tensor
    sf: torch.Tensor
    reconstructed: torch.Tensor


def nvfp4(x, global_scale=1.):
    blocks = x.float().reshape(*x.shape[:-1], -1, 16)
    scales = (blocks.abs().amax(-1) / (6 * global_scale)).clamp(1 / 512, 448)
    scales = scales.to(torch.float8_e4m3fn)
    effective = scales.float() * global_scale
    normalized = (blocks / effective[..., None]).reshape(x.shape)
    boundaries = x.new_tensor((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.), dtype=torch.float32)
    absolute = normalized.abs()
    code = torch.bucketize(absolute.contiguous(), boundaries)
    for i in (1, 3, 5):  # Ties-to-even when the upper code is even.
        code = torch.where(absolute == boundaries[i], i + 1, code)
    nibble = code | (torch.signbit(normalized).long() << 3)
    packed = (nibble[..., ::2] | (nibble[..., 1::2] << 4)).to(torch.uint8).contiguous()
    reconstructed = (unpack(packed).reshape_as(blocks) * effective[..., None]).reshape(x.shape)
    return Quantized(packed, scales, reconstructed)


def reference(q, qs, qr, k, ks, rope, lengths, splits):
    qd = unpack(q) * qs.float().repeat_interleave(16, -1)
    kd = unpack(k) * ks.float().repeat_interleave(16, -1)
    vd = kd.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    scores = (qd @ kd.transpose(-1, -2) + qr.float() @ rope.float().transpose(-1, -2)) / 512**.5
    outputs = []
    for b, length in enumerate(lengths.tolist()):
        tiles = (length + 127) // 128
        per = (tiles + splits - 1) // splits
        parts, lses = [], []
        for first in range(0, tiles, per):
            m = scores.new_full((q.shape[1], 1), -torch.inf)
            den = torch.zeros_like(m)
            acc = scores.new_zeros(q.shape[1], 512)
            for tile in range(first, min(first + per, tiles)):
                off = tile * 128
                score = scores[b, :, off:off + 128].clone()
                score[:, max(0, length - off):] = -torch.inf
                new_m = torch.maximum(m, score.amax(-1, keepdim=True))
                cor = (m - new_m).exp()
                p = (score - new_m).exp()
                den = den * cor + p.sum(-1, keepdim=True)
                acc = acc * cor + p.to(torch.float8_e4m3fn).float() @ vd[b, off:off + 128]
                m = new_m
            parts.append(acc / den)
            lses.append(m + den.log())
        outputs.append((torch.stack(parts) * torch.stack(lses).softmax(0)).sum(0))
    return torch.stack(outputs)
