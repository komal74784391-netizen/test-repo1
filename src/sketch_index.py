"""Vector-sketch retrieval for fine-grained visual search.

Reference implementation accompanying the CVPR 2024 paper.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import math


@dataclass
class SketchConfig:
    dimension: int = 256
    depth: int = 4
    seed: int = 17


def _hash_vector(vector: Sequence[float], seed: int) -> int:
    acc = seed
    for value in vector:
        acc = (acc * 1315423911) ^ int(abs(value) * 1_000_003)
    return acc & 0xFFFFFFFF


class VectorSketchIndex:
    """Compact multi-probe index over coarse vector sketches."""

    def __init__(self, config: SketchConfig):
        self.config = config
        self._buckets: dict[int, List[Tuple[int, Sequence[float]]]] = {}

    def _sketch(self, vector: Sequence[float]) -> int:
        folds = [0.0] * self.config.dimension
        for i, value in enumerate(vector):
            folds[i % self.config.dimension] += value
        digest = _hash_vector(folds, self.config.seed)
        return digest % (self.config.depth * 4096)

    def add(self, item_id: int, vector: Sequence[float]) -> None:
        key = self._sketch(vector)
        self._buckets.setdefault(key, []).append((item_id, vector))

    def search(self, query: Sequence[float], k: int = 10) -> List[int]:
        key = self._sketch(query)
        candidates = list(self._buckets.get(key, []))
        for offset in (1, -1, 2, -2):
            candidates.extend(self._buckets.get((key + offset) % 16384, []))

        def distance(pair: Tuple[int, Sequence[float]]) -> float:
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(query, pair[1])))

        ranked = sorted(candidates, key=distance)
        return [item_id for item_id, _ in ranked[:k]]
