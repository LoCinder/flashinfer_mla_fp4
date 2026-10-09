# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Native NVFP4 QK with fused register-to-SMEM FP8 V dequantization."""

from __future__ import annotations

from dataclasses import dataclass, replace

from flashinfer.cute_dsl.attention.mainloop_spec import MLAMainloopSpec
from flashinfer.cute_dsl.attention.mla_config import MLAConfig
from flashinfer.cute_dsl.attention.mla_warp_schedule import MLAWarpScheduleFP8
from flashinfer.cute_dsl.attention.pipeline_topology import (
    PipelineEdge,
    PipelineType,
    make_mla_fp8_topology,
)


@dataclass(frozen=True)
class MLAConfigFP4(MLAConfig):
    """One NVFP4 payload; block16 E4M3 scales serve QK and V dequantization."""

    is_fp8: bool = True
    scale_group_width: int = 512
    load_k_stage: int = 4
    load_v_stage: int = 2
    p_mma_stage: int = 2
    mma_o_stage: int = 2
    token_scale_stages: int = 4

    @property
    def num_scale_groups(self) -> int:
        return self.latent_dim // self.scale_group_width


MLA_DECODE_FP4_SCHEDULE = MLAWarpScheduleFP8(
    empty_warp_ids=(11, 12, 13, 14, 15),
    softmax_reg_num=192,
    correction_reg_num=192,
    other_reg_num=64,
)


def make_mla_fp4_mainloop_spec(
    config: MLAConfigFP4,
    schedule: MLAWarpScheduleFP8 | None = None,
) -> MLAMainloopSpec:
    """Create the mainloop for NVFP4 QK and fused-dequantization FP8 PV."""

    sched = schedule if schedule is not None else MLA_DECODE_FP4_SCHEDULE
    topology = make_mla_fp8_topology(
        sched,
        load_q_stages=config.load_q_stage,
        load_k_stages=config.load_k_stage,
        load_v_stages=config.load_v_stage,
        mma_s_stages=config.mma_s_stage,
        p_mma_stages=config.p_mma_stage,
        p_cor_stages=config.p_cor_stage,
        mma_o_stages=config.mma_o_stage,
        cluster_scale=config.cluster_shape_mnk[0],
    )
    topology.edges = [
        replace(edge, pipeline_type=PipelineType.ASYNC_UMMA,
                producer_warp_ids=(10, 11, 12, 13, 14, 15), tx_count_key=None,
                cluster_scale=config.cluster_shape_mnk[0])
        if edge.name == "load_v" else edge
        for edge in topology.edges
    ]
    return MLAMainloopSpec(
        config=config,
        warp_schedule=sched,
        pipeline_topology=topology,
    )


__all__ = [
    "MLAConfigFP4",
    "MLA_DECODE_FP4_SCHEDULE",
    "make_mla_fp4_mainloop_spec",
]
