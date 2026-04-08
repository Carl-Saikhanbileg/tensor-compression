from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from tensor_core import TensorData
from preprocessing import preprocess_singular_values
from sketch import Sketch, enumerate_sketches
from scoring import RankAssignmentResult, assign_ranks_greedily
from execution import ExecutedNetwork, execute_scored_sketch


@dataclass
class ScoredSketch:
    """
    A sketch together with its rank assignment and surrogate score.
    """
    sketch: Sketch
    rank_assignment: RankAssignmentResult
    estimated_cost: int


@dataclass
class SearchResult:
    """
    Final output of the top-level search procedure.
    """
    best_network: ExecutedNetwork
    best_cost: int
    best_scored_sketch: ScoredSketch
    top_scored_sketches: List[ScoredSketch] = field(default_factory=list)
    executed_candidates: List[Tuple[int, ScoredSketch]] = field(default_factory=list)


def top_k_sketches(
    X: TensorData,
    eps: float,
    max_splits: int,
    k: int,
) -> List[ScoredSketch]:
    """
    1. Precompute singular values for all bipartitions.
    2. Enumerate valid sketches up to max_splits.
    3. Score each sketch via surrogate rank assignment.
    4. Return the k best sketches by estimated cost.
    """
    pre = preprocess_singular_values(X)
    sketches = enumerate_sketches(d=X.order, max_splits=max_splits)

    scored = []
    for sketch in sketches:
        rank_result = assign_ranks_greedily(pre, sketch, eps=eps)
        scored.append(
            ScoredSketch(
                sketch=sketch,
                rank_assignment=rank_result,
                estimated_cost=rank_result.estimated_cost,
            )
        )

    scored.sort(key=lambda s: s.estimated_cost)
    return scored[:k]


def structure_search_prototype(
    X: TensorData,
    eps: float,
    max_splits: int = 3,
    top_k: int = 5,
) -> SearchResult:
    """
    End-to-end prototype:

    1. Score all sketches symbolically.
    2. Keep top_k sketches by surrogate cost.
    3. Execute only those top_k sketches.
    4. Return the executed network with minimum actual storage cost.
    """
    top_scored = top_k_sketches(
        X=X,
        eps=eps,
        max_splits=max_splits,
        k=top_k,
    )

    dense_network = ExecutedNetwork.from_tensor(X)
    best_network = dense_network
    best_cost = dense_network.storage_cost()

    if top_scored:
        best_scored = top_scored[0]
    else:
        best_scored = ScoredSketch(
            sketch=Sketch(tuple()),
            rank_assignment=RankAssignmentResult(
                ranks={},
                total_error_sq=0.0,
                estimated_cost=best_cost,
            ),
            estimated_cost=best_cost,
        )

    executed_candidates: List[Tuple[int, ScoredSketch]] = []

    for scored in top_scored:
        try:
            net = execute_scored_sketch(
                X=X,
                sketch=scored.sketch,
                rank_assignment=scored.rank_assignment,
            )
            cost = net.storage_cost()
            executed_candidates.append((cost, scored))

            if cost < best_cost:
                best_cost = cost
                best_network = net
                best_scored = scored
        except Exception:
            # prototype search skips sketches that fail during execution
            continue

    return SearchResult(
        best_network=best_network,
        best_cost=best_cost,
        best_scored_sketch=best_scored,
        top_scored_sketches=top_scored,
        executed_candidates=executed_candidates,
    )


def demo() -> None:
    rng = np.random.default_rng(0)

    X = TensorData(rng.standard_normal((8, 9, 10, 11)))
    eps = 0.10

    result = structure_search_prototype(
        X=X,
        eps=eps,
        max_splits=3,
        top_k=5,
    )

    print("input shape:", X.shape)
    print("dense cost:", X.size)
    print("best executed cost:", result.best_cost)
    print()

    print("best sketch subsets:")
    print(result.best_scored_sketch.sketch.subsets())
    print()

    print("best assigned ranks:")
    print(result.best_scored_sketch.rank_assignment.ranks)
    print()

    print("top scored sketches (estimated costs):")
    for i, scored in enumerate(result.top_scored_sketches, start=1):
        print(
            f"{i}. subsets={scored.sketch.subsets()} "
            f"| estimated_cost={scored.estimated_cost}"
        )
    print()

    print("executed candidate costs:")
    for cost, scored in result.executed_candidates:
        print(
            f"cost={cost} | subsets={scored.sketch.subsets()} "
            f"| estimated_cost={scored.estimated_cost}"
        )
    print()

    print(result.best_network.summary())


if __name__ == "__main__":
    demo()