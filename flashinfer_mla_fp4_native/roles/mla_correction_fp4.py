# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Correction adapter for mixed E4M3/E2M1 PV."""

import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils

from flashinfer.cute_dsl.attention.config import AttentionFusion
from flashinfer.cute_dsl.attention.roles.mla_correction import MLACorrectionRole

from ..mla_config_fp4 import MLAConfigFP4


class MLACorrectionFP4Role(MLACorrectionRole):
    """Use E4M3 P with E2M1 V while reusing all correction logic."""

    def __init__(
        self,
        config: MLAConfigFP4,
        fusion: AttentionFusion,
        p_dtype=None,
        v_dtype=None,
        o_dtype=None,
    ):
        super().__init__(config, fusion, v_dtype=v_dtype, o_dtype=o_dtype)
        self.p_dtype = p_dtype

    @cute.jit
    def _make_pv_tiled_mma(self):
        return sm100_utils.make_trivial_tiled_mma(
            self.p_dtype,
            self.v_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.TWO,
            self.mma_pv_tiler[:2],
        )


__all__ = ["MLACorrectionFP4Role"]
