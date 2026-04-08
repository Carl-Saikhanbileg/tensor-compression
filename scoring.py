from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Dict, List, Tuple

import numpy as np

from preprocessing import PrecomputedSVals, canonical_subset
from sketch import Sketch, immediate_children_subsets


@dataclass
class RankAssignmentResult:
    """
    Result of sketch scoring / rank assignment.

    Attributes
    ----------
    ranks:
        Maps each sketch subset to its assigned rank.
    total_error_sq:
        Surrogate total truncation error:
            sum_s sum_{i > r_s} sigma_{s,i}^2
    estimated_cost:
        Estimated storage cost of the tensor network induced by the sketch.
    """
    ranks: Dict[Tuple[int, ...], int]
    total_error_sq: float
    estimated_cost: int


def truncation_error_sq(svals: np.ndarray, rank: int) -> float:
    """
    Tail energy after rank truncation:
        sum_{i > rank} sigma_i^2

    In zero-based Python slicing, if svals = [sigma_1, ..., sigma_q],
    then keeping rank=r means discarding svals[r:].
    """
    if rank >= len(svals):
        return 0.0
    return float(np.sum(svals[rank:] ** 2))


def cluster_tree_from_subsets(
    shape: Tuple[int, ...],
    subsets: List[Tuple[int, ...]],
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], Tuple[int, ...] | None], Dict[Tuple[int, ...], List[Tuple[int, ...]]]]:
    """
    Build a laminar cluster tree from sketch subsets.

    The tree contains:
    - all singleton mode clusters
    - all sketch subsets
    - the full mode set as root
    """
    d = len(shape)
    root = tuple(range(d))

    clusters = {tuple(sorted(s)) for s in subsets}
    clusters.add(root)
    for i in range(d):
        clusters.add((i,))

    clusters = sorted(clusters, key=lambda s: (len(s), s))

    parent: Dict[Tuple[int, ...], Tuple[int, ...] | None] = {c: None for c in clusters}

    for c in clusters:
        supersets = [p for p in clusters if set(c) < set(p)]
        if supersets:
            parent[c] = min(supersets, key=lambda p: len(p))

    children: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {c: [] for c in clusters}
    for c, p in parent.items():
        if p is not None:
            children[p].append(c)

    for c in children:
        children[c].sort(key=lambda s: (len(s), s))

    return clusters, parent, children


def network_cost_from_subsets_and_ranks(
    shape: Tuple[int, ...],
    subsets: List[Tuple[int, ...]],
    ranks: Dict[Tuple[int, ...], int],
) -> int:
    """
    Estimate tensor-network storage cost induced by a laminar family.

    Cost model:
    - singleton cluster (leaf/original mode i):
        n_i * parent_rank
    - internal cluster:
        product(child_ranks) * parent_rank
    - root:
        product(child_ranks)

    Here cluster ranks correspond to the rank on the edge between a cluster
    and its parent. The root has no parent rank.
    """
    d = len(shape)
    clusters, parent, children = cluster_tree_from_subsets(shape, subsets)

    cost = 0

    for c in clusters:
        child_list = children[c]
        p = parent[c]

        if len(c) == 1:
            # physical leaf
            mode = c[0]
            parent_rank = 1 if p is None else ranks.get(canonical_subset(c, d), 1)
            cost += shape[mode] * parent_rank
        else:
            dims = []

            for ch in child_list:
                dims.append(ranks.get(canonical_subset(ch, d), 1))

            if p is not None:
                dims.append(ranks.get(canonical_subset(c, d), 1))

            if dims:
                cost += int(prod(dims))
            else:
                # degenerate root-only case
                cost += int(prod(shape))

    return int(cost)


def initial_ranks_for_sketch(
    pre: PrecomputedSVals,
    sketch: Sketch,
) -> Dict[Tuple[int, ...], int]:
    """
    Initialize every sketch subset with rank 1.
    """
    d = len(pre.tensor_shape)
    out = {}

    for subset in sketch.subsets():
        c = canonical_subset(subset, d)
        out[c] = 1

    return out


def sketch_unique_subsets(
    pre: PrecomputedSVals,
    sketch: Sketch,
) -> List[Tuple[int, ...]]:
    """
    Canonicalized unique subsets appearing in the sketch.
    """
    d = len(pre.tensor_shape)
    out = []
    seen = set()

    for subset in sketch.subsets():
        c = canonical_subset(subset, d)
        if c not in seen:
            seen.add(c)
            out.append(c)

    out.sort(key=lambda s: (len(s), s))
    return out


def total_sketch_error_sq(
    pre: PrecomputedSVals,
    subsets: List[Tuple[int, ...]],
    ranks: Dict[Tuple[int, ...], int],
) -> float:
    """
    Surrogate global error budget used in the paper's scoring phase:
        sum_s sum_{i > r_s} sigma_{s,i}^2
    """
    total = 0.0
    for subset in subsets:
        svals = pre.get(subset)
        total += truncation_error_sq(svals, ranks[subset])
    return float(total)


def candidate_rank_upgrade_scores(
    pre: PrecomputedSVals,
    shape: Tuple[int, ...],
    subsets: List[Tuple[int, ...]],
    ranks: Dict[Tuple[int, ...], int],
) -> List[Tuple[float, Tuple[int, ...], float, int]]:
    """
    For each possible single-rank upgrade, compute:

        score = error_drop / cost_increase

    Returns list of tuples:
        (score, subset, new_total_error_sq, new_estimated_cost)
    """
    current_cost = network_cost_from_subsets_and_ranks(shape, subsets, ranks)
    current_error = total_sketch_error_sq(pre, subsets, ranks)

    out = []

    for subset in subsets:
        svals = pre.get(subset)
        r = ranks[subset]

        if r >= len(svals):
            continue

        old_err = truncation_error_sq(svals, r)
        new_err = truncation_error_sq(svals, r + 1)
        err_drop = old_err - new_err

        trial_ranks = dict(ranks)
        trial_ranks[subset] = r + 1

        trial_cost = network_cost_from_subsets_and_ranks(shape, subsets, trial_ranks)
        cost_increase = max(1, trial_cost - current_cost)

        score = err_drop / cost_increase
        new_total_error = current_error - err_drop

        out.append((score, subset, float(new_total_error), int(trial_cost)))

    out.sort(key=lambda x: x[0], reverse=True)
    return out


def assign_ranks_greedily(
    pre: PrecomputedSVals,
    sketch: Sketch,
    eps: float,
) -> RankAssignmentResult:
    """
    Greedy rank assignment under the surrogate budget:

        sum_s sum_{i > r_s} sigma_{s,i}^2 <= (eps ||X||_F)^2

    Algorithm:
    1. Start all sketch ranks at 1.
    2. Compute surrogate total error.
    3. While over budget:
         increase the rank of the subset with largest
         error_drop / cost_increase.
    4. Return final ranks and estimated cost.

    This is intentionally modular so it can later be replaced by
    an exact optimizer without touching the rest of the pipeline.
    """
    subsets = sketch_unique_subsets(pre, sketch)
    ranks = initial_ranks_for_sketch(pre, sketch)

    error_budget = (eps ** 2) * pre.fro_norm_sq
    total_error = total_sketch_error_sq(pre, subsets, ranks)

    while total_error > error_budget:
        candidates = candidate_rank_upgrade_scores(
            pre=pre,
            shape=pre.tensor_shape,
            subsets=subsets,
            ranks=ranks,
        )

        if not candidates:
            break

        _, best_subset, new_total_error, _ = candidates[0]
        ranks[best_subset] += 1
        total_error = new_total_error

    estimated_cost = network_cost_from_subsets_and_ranks(
        shape=pre.tensor_shape,
        subsets=subsets,
        ranks=ranks,
    )

    return RankAssignmentResult(
        ranks=ranks,
        total_error_sq=float(total_error),
        estimated_cost=int(estimated_cost),
    )


if __name__ == "__main__":
    import numpy as np

    from tensor_core import TensorData
    from preprocessing import preprocess_singular_values
    from sketch import make_sketch

    X = TensorData(np.random.randn(8, 9, 10, 11))
    pre = preprocess_singular_values(X)

    sketch = make_sketch([
        (0,),
        (1,),
        (0, 1),
    ])

    result = assign_ranks_greedily(pre, sketch, eps=0.10)

    print("sketch subsets:", sketch.subsets())
    print("assigned ranks:", result.ranks)
    print("total error sq:", result.total_error_sq)
    print("estimated cost:", result.estimated_cost)