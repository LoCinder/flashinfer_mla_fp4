# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile-time configuration for per-token-scaled FP4 MLA decode."""

from __future__ import annotations

from dataclasses import dataclass

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
    """MLA configuration for per-token-scaled FP4 latent and E4M3 RoPE KV.

    The latent dimension is packed E2M1 with one effective FP32 scale per token.  The
    independent RoPE field and query remain E4M3.  This is deliberately a
    sibling configuration rather than a flag on :class:`MLAConfig` so the
    dense FP8 path stays unchanged.
    """

    is_fp8: bool = True
    scale_group_width: int = 512
    load_k_stage: int = 2
    load_v_stage: int = 2
    p_mma_stage: int = 2
    mma_o_stage: int = 2

    @property
    def num_scale_groups(self) -> int:
        return self.latent_dim // self.scale_group_width


MLA_DECODE_FP4_SCHEDULE = MLAWarpScheduleFP8(
    softmax_reg_num=240,
    correction_reg_num=208,
)


def make_mla_fp4_mainloop_spec(
    config: MLAConfigFP4,
    schedule: MLAWarpScheduleFP8 | None = None,
) -> MLAMainloopSpec:
    """Create the mainloop specification for per-token-scaled FP4 MLA."""

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
    topology.edges.append(
        PipelineEdge(
            "load_scale",
            PipelineType.CP_ASYNC,
            stages=config.mma_s_stage,
            producer_warp_ids=(sched.load_tma_k_warp_id,),
            consumer_warp_ids=sched.compute_warp_ids,
        )
    )
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
