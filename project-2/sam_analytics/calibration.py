"""Leakage-free probability calibration utilities.

`IsotonicCalibrator` is a small dependency-free implementation of the pooled
adjacent violators algorithm.  Fit it only on an out-of-fold or prior-period
calibration set, never on the same rows used to report final results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class _Block:
    lower: float
    upper: float
    positives: float
    count: int

    @property
    def value(self) -> float:
        return self.positives / self.count


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._blocks: List[_Block] = []

    @property
    def fitted(self) -> bool:
        return bool(self._blocks)

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> "IsotonicCalibrator":
        if len(probabilities) != len(outcomes) or len(probabilities) < 2:
            raise ValueError("calibration needs matching probability/outcome pairs (n >= 2)")
        pairs: List[Tuple[float, int]] = []
        for probability, outcome in zip(probabilities, outcomes):
            probability = float(probability)
            if not 0.0 <= probability <= 1.0 or outcome not in (0, 1, False, True):
                raise ValueError("probabilities must be [0, 1] and outcomes must be binary")
            pairs.append((probability, int(outcome)))
        pairs.sort(key=lambda pair: pair[0])

        # Equal raw scores are one calibration level, not arbitrary ordered rows.
        blocks: List[_Block] = []
        for probability, outcome in pairs:
            if blocks and probability == blocks[-1].upper:
                previous = blocks.pop()
                blocks.append(
                    _Block(
                        lower=previous.lower,
                        upper=probability,
                        positives=previous.positives + outcome,
                        count=previous.count + 1,
                    )
                )
            else:
                blocks.append(_Block(probability, probability, outcome, 1))
            while len(blocks) >= 2 and blocks[-2].value > blocks[-1].value:
                right, left = blocks.pop(), blocks.pop()
                blocks.append(
                    _Block(
                        lower=left.lower,
                        upper=right.upper,
                        positives=left.positives + right.positives,
                        count=left.count + right.count,
                    )
                )
        self._blocks = blocks
        return self

    def predict_one(self, probability: float) -> float:
        if not self._blocks:
            raise RuntimeError("fit the calibrator before predicting")
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        for block in self._blocks:
            if probability <= block.upper:
                return block.value
        return self._blocks[-1].value

    def predict(self, probabilities: Iterable[float]) -> List[float]:
        return [self.predict_one(probability) for probability in probabilities]
