from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import List, Tuple

from tensor_core import TensorData
from preprocessing import PrecomputedSVals
from sketch import Sketch, enumerate_sketches
from scoring import RankAssignmentResult, assign_ranks_greedily


@dataclass(frozen=True)
class TopKCandidate:
    sketch: Sketch
    rank_assignment: RankAssignmentResult
    estimated_cost: int


def enumerate_o(X: TensorData, max_splits: int) -> List[Sketch]:

    return enumerate_sketches(d=X.order, max_splits=max_splits)


def get_cost(
    Omega: PrecomputedSVals,
    X: TensorData,
    sketch: Sketch,
    epsilon: float,
) -> Tuple[int, RankAssignmentResult]:

    rank_result = assign_ranks_greedily(Omega, sketch, eps=epsilon)
    return rank_result.estimated_cost, rank_result


def update_topk_heap(
    heap: List[Tuple[int, int, TopKCandidate]],
    candidate: TopKCandidate,
    k: int,
    tie_breaker: int,
) -> None:

    item = (-candidate.estimated_cost, tie_breaker, candidate)

    if len(heap) < k:
        heapq.heappush(heap, item)
        return

    current_worst_cost = -heap[0][0]
    if candidate.estimated_cost < current_worst_cost:
        heapq.heapreplace(heap, item)


def top_k(
    Omega: PrecomputedSVals,
    X: TensorData,
    epsilon: float,
    max_splits: int,
    k: int,
) -> List[TopKCandidate]:

    if k <= 0:
        return []

    # line 2
    topk_heap: List[Tuple[int, int, TopKCandidate]] = []
    tie_breaker = 0

    # line 3
    for sketch in enumerate_o(X, max_splits=max_splits):
        # line 4
        cost, rank_assignment = get_cost(
            Omega=Omega,
            X=X,
            sketch=sketch,
            epsilon=epsilon,
        )

        candidate = TopKCandidate(
            sketch=sketch,
            rank_assignment=rank_assignment,
            estimated_cost=cost,
        )

        # line 5
        update_topk_heap(
            heap=topk_heap,
            candidate=candidate,
            k=k,
            tie_breaker=tie_breaker,
        )
        tie_breaker += 1

    # line 6
    result = [item[2] for item in topk_heap]
    result.sort(key=lambda cand: cand.estimated_cost)
    return result


if __name__ == "__main__":
    import numpy as np
    from preprocessing import preprocess_singular_values

    X = TensorData(np.random.randn(8, 9, 10, 11))
    epsilon = 0.10

    Omega = preprocess_singular_values(X)

    candidates = top_k(
        Omega=Omega,
        X=X,
        epsilon=epsilon,
        max_splits=3,
        k=5,
    )

    print("Top-k candidates:")
    for i, cand in enumerate(candidates, start=1):
        print(
            f"{i}. subsets={cand.sketch.subsets()} | "
            f"estimated_cost={cand.estimated_cost} | "
            f"ranks={cand.rank_assignment.ranks}"
        )