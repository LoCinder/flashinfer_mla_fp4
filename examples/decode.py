"""Small fused-dequantization decode example; prepare and compile first."""

import torch

from flashinfer_mla_fp4_fused_dequant import Runner
from flashinfer_mla_fp4_fused_dequant.quantization.quantize import quantize


def main():
    torch.manual_seed(439)
    batch, heads, seq = 1, 64, 512
    x = torch.randn(batch, seq, 512, device="cuda")
    q0 = torch.randn(batch, heads, 512, device="cuda")
    q, qs = quantize(q0)
    k, ks = quantize(x)
    qr = torch.randn(batch, heads, 64, device="cuda").to(torch.float8_e4m3fn)
    kr = torch.randn(batch, seq, 64, device="cuda").to(torch.float8_e4m3fn)
    lengths = torch.full((batch,), seq, device="cuda", dtype=torch.int32)
    runner = Runner(q, qs, qr, k, ks, kr, lengths, splits=8)
    output = runner()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, finite=True")


if __name__ == "__main__":
    main()
