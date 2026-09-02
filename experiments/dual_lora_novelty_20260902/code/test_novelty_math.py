from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from novelty_patch import (
    adapter_effective_norm,
    cosine_apply_weight,
    deterministic_window_sample,
    factorized_delta_sq_norm,
    project_adapter_effective_norm,
    resolve_apply_weight,
    select_hard_negatives,
)
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear


def test_factorized_norm_matches_dense_product() -> None:
    generator = torch.Generator().manual_seed(123)
    a = torch.randn(4, 9, generator=generator)
    b = torch.randn(7, 4, generator=generator)
    scaling = 2.0
    expected = torch.linalg.matrix_norm(scaling * (b @ a), ord="fro") ** 2
    actual = factorized_delta_sq_norm(a, b, scaling=scaling)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_cosine_schedule() -> None:
    assert cosine_apply_weight(0, initial=0.5, decay_end=350) == 0.5
    assert math.isclose(
        cosine_apply_weight(175, initial=0.5, decay_end=350), 0.25
    )
    assert cosine_apply_weight(350, initial=0.5, decay_end=350) == 0.0
    assert cosine_apply_weight(500, initial=0.5, decay_end=350) == 0.0


def test_window_sample_is_deterministic_and_occurrence_based() -> None:
    values = ["attack", "attack", "wait", "enter", "inspect"]
    indices1, sample1 = deterministic_window_sample(
        values, sample_size=3, seed=42, step=10
    )
    indices2, sample2 = deterministic_window_sample(
        values, sample_size=3, seed=42, step=10
    )
    assert indices1 == indices2
    assert sample1 == sample2
    assert len(set(indices1)) == 3
    assert all(sample == values[index] for index, sample in zip(indices1, sample1))


def test_hard_negative_selection_uses_scores_and_fifo_ties() -> None:
    values = ["wait", "attack", "wait", "inspect"]
    scores = [-8.0, -3.0, -8.0, -3.0]
    indices, selected = select_hard_negatives(values, scores, sample_size=3)
    assert indices == [0, 1, 3]
    assert selected == ["wait", "attack", "inspect"]


def test_relative_apply_weight_tracks_task_norm_ratio() -> None:
    scale = resolve_apply_weight(
        0.25,
        strength_mode="relative_task_norm",
        task_norm=0.2,
        novelty_norm=0.01,
    )
    assert math.isclose(scale, 5.0)
    assert math.isclose(scale * 0.01 / 0.2, 0.25)
    assert resolve_apply_weight(
        0.25,
        strength_mode="relative_task_norm",
        task_norm=0.2,
        novelty_norm=0.0,
    ) == 0.0


def test_effective_norm_projection() -> None:
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["linear"],
    )
    layer = LoraLinear(
        nn.Linear(9, 7, bias=False),
        "novelty",
        config=config,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    model = nn.Sequential(layer)
    generator = torch.Generator().manual_seed(321)
    with torch.no_grad():
        layer.lora_B["novelty"].weight.copy_(
            torch.randn(layer.lora_B["novelty"].weight.shape, generator=generator)
        )
    original = adapter_effective_norm(model, "novelty")
    before, after = project_adapter_effective_norm(model, "novelty", 0.75)
    assert math.isclose(before, original, rel_tol=1e-5)
    assert math.isclose(after, 0.75, rel_tol=1e-5)


if __name__ == "__main__":
    test_factorized_norm_matches_dense_product()
    test_cosine_schedule()
    test_window_sample_is_deterministic_and_occurrence_based()
    test_hard_negative_selection_uses_scores_and_fifo_ties()
    test_relative_apply_weight_tracks_task_norm_ratio()
    test_effective_norm_projection()
    print("novelty math tests passed")
