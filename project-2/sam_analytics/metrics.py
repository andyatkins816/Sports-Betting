"""Proper scoring rules and calibration diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence


def _validate(probabilities: Sequence[float], outcomes: Sequence[int]) -> None:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("matching non-empty probabilities and outcomes are required")
    if any(not 0.0 <= float(p) <= 1.0 for p in probabilities):
        raise ValueError("probabilities must be in [0, 1]")
    if any(outcome not in (0, 1, False, True) for outcome in outcomes):
        raise ValueError("outcomes must be binary")


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    _validate(probabilities, outcomes)
    return sum((float(p) - int(y)) ** 2 for p, y in zip(probabilities, outcomes)) / len(outcomes)


def log_loss(probabilities: Sequence[float], outcomes: Sequence[int], epsilon: float = 1e-15) -> float:
    _validate(probabilities, outcomes)
    return -sum(
        int(y) * math.log(min(max(float(p), epsilon), 1 - epsilon))
        + (1 - int(y)) * math.log(1 - min(max(float(p), epsilon), 1 - epsilon))
        for p, y in zip(probabilities, outcomes)
    ) / len(outcomes)


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_prediction: float
    observed_rate: float


def calibration_bins(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> List[CalibrationBin]:
    _validate(probabilities, outcomes)
    if bins < 2:
        raise ValueError("at least two bins are required")
    grouped: Dict[int, List[tuple]] = {index: [] for index in range(bins)}
    for probability, outcome in zip(probabilities, outcomes):
        index = min(int(float(probability) * bins), bins - 1)
        grouped[index].append((float(probability), int(outcome)))
    output: List[CalibrationBin] = []
    for index, values in grouped.items():
        if values:
            output.append(
                CalibrationBin(
                    lower=index / bins,
                    upper=(index + 1) / bins,
                    count=len(values),
                    mean_prediction=sum(pair[0] for pair in values) / len(values),
                    observed_rate=sum(pair[1] for pair in values) / len(values),
                )
            )
    return output


def evaluation_report(probabilities: Sequence[float], outcomes: Sequence[int]) -> Dict[str, object]:
    return {
        "sample_size": len(outcomes),
        "brier_score": brier_score(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes),
        "calibration": [bin.__dict__ for bin in calibration_bins(probabilities, outcomes)],
    }
