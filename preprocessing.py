from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from tensor_core import TensorData


def canonical_subset(subset: Iterable[int], d: int) -> Tuple[int, ...]:
    s = tuple(sorted(subset))
    comp = tuple(i for i in range(d) if i not in s)

    if len(s) < len(comp):
        return s
    if len(comp) < len(s):
        return comp
    return min(s, comp)


def all_unique_bipartitions(d: int) -> List[Tuple[int, ...]]:
    parts = []

    for mask in range(1, 1 << d):
        subset = tuple(i for i in range(d) if (mask >> i) & 1)
        if len(subset) == d:
            continue

        c = canonical_subset(subset, d)
        if subset == c:
            parts.append(c)

    return parts


@dataclass
class PrecomputedSVals:
    tensor_shape: Tuple[int, ...]
    svals: Dict[Tuple[int, ...], np.ndarray]
    fro_norm_sq: float

    def get(self, subset: Iterable[int]) -> np.ndarray:
        key = canonical_subset(subset, len(self.tensor_shape))
        return self.svals[key]


def preprocess_singular_values(X: TensorData) -> PrecomputedSVals:
    svals = {}

    for subset in all_unique_bipartitions(X.order):
        M, _, _ = X.matricize(subset)
        svals[subset] = np.linalg.svd(M, compute_uv=False, full_matrices=False)

    return PrecomputedSVals(
        tensor_shape=X.shape,
        svals=svals,
        fro_norm_sq=X.fro_norm_sq,
    )


if __name__ == "__main__":
    X = TensorData(np.random.randn(4, 5, 6, 7))
    pre = preprocess_singular_values(X)

    print("number of unique bipartitions:", len(pre.svals))
    print("stored keys:", list(pre.svals.keys())[:5])

    key = (0, 2)
    print(f"singular values for {canonical_subset(key, X.order)}:")
    print(pre.get(key))