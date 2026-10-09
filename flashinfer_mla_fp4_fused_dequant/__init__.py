"""NVFP4 QK with fused NVFP4-to-FP8 PV dequantization."""

from importlib.metadata import PackageNotFoundError, version

try:
    _flashinfer_version = version("flashinfer-python")
except PackageNotFoundError:
    raise ImportError("flashinfer_mla_fp4_fused_dequant requires flashinfer-python==0.6.17") from None
if _flashinfer_version != "0.6.17":
    raise ImportError("flashinfer_mla_fp4_fused_dequant requires flashinfer-python==0.6.17")

from .wrappers.batch_mla_fp4 import Runner

__all__ = ["Runner"]
