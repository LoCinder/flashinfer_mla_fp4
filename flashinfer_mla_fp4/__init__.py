# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import version as _package_version

if _package_version("flashinfer-python") != "0.6.17":
    raise ImportError("flashinfer_mla_fp4 requires flashinfer-python==0.6.17")

from .wrappers.batch_mla_fp4 import BatchMLADecodeCuteDSLFP4Wrapper

__all__ = ["BatchMLADecodeCuteDSLFP4Wrapper"]
