from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Dict, List, Tuple

import numpy as np

from tensor_core import TensorData
from preprocessing import canonical_subset
from scoring import RankAssignmentResult
from sketch import Sketch


@dataclass
class TensorNode:
    """
    Concrete tensor node used during execution.

    axis_labels tracks which axes are:
    - original tensor modes: "orig:i"
    - internal rank edges:   "rank:rj"
    """
    name: str
    data: np.ndarray
    axis_labels: Tuple[str, ...]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return int(prod(self.shape))

    @property
    def original_modes(self) -> Tuple[int, ...]:
        return tuple(
            int(label.split(":")[1])
            for label in self.axis_labels
            if label.startswith("orig:")
        )

    @property
    def rank_labels(self) -> Tuple[str, ...]:
        return tuple(
            label.split(":")[1]
            for label in self.axis_labels
            if label.startswith("rank:")
        )

    def axis_for_original_mode(self, mode: int) -> int:
        target = f"orig:{mode}"
        for i, label in enumerate(self.axis_labels):
            if label == target:
                return i
        raise ValueError(f"mode {mode} not found in node {self.name}")


@dataclass
class ExecutedNetwork:
    """
    Concrete executed tensor network produced by repeated OSplit execution.
    """
    nodes: Dict[str, TensorNode] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)
    next_rank_id: int = 0

    @classmethod
    def from_tensor(cls, X: TensorData) -> "ExecutedNetwork":
        root = TensorNode(
            name="X",
            data=X.data.copy(),
            axis_labels=tuple(f"orig:{i}" for i in range(X.order)),
        )
        return cls(nodes={"X": root}, edges=[], next_rank_id=0)

    def fresh_rank_label(self) -> str:
        label = f"r{self.next_rank_id}"
        self.next_rank_id += 1
        return label

    def storage_cost(self) -> int:
        return int(sum(node.size for node in self.nodes.values()))

    def find_node_for_subset(self, subset: Tuple[int, ...]) -> str:
        """
        Find the smallest active node whose original modes strictly contain subset.
        """
        target = set(subset)
        candidates = []

        for name, node in self.nodes.items():
            modes = set(node.original_modes)
            if target.issubset(modes) and target != modes:
                candidates.append((len(modes), name))

        if not candidates:
            raise ValueError(f"no executable node found for subset {subset}")

        candidates.sort()
        return candidates[0][1]

    def execute_osplit(self, subset: Tuple[int, ...], rank: int) -> None:
        """
        Execute one symbolic OSplit(subset, rank) by:
        1. finding the smallest active node containing the subset
        2. matricizing that node with subset on the left
        3. applying truncated SVD
        4. replacing the node by two children connected by a new rank edge
        """
        node_name = self.find_node_for_subset(subset)
        node = self.nodes[node_name]

        left_axes = tuple(node.axis_for_original_mode(mode) for mode in subset)
        all_axes = tuple(range(node.data.ndim))
        right_axes = tuple(ax for ax in all_axes if ax not in left_axes)

        perm = left_axes + right_axes
        permuted = np.transpose(node.data, axes=perm)

        left_dim = int(prod(node.data.shape[ax] for ax in left_axes))
        right_dim = int(prod(node.data.shape[ax] for ax in right_axes))
        M = permuted.reshape(left_dim, right_dim)

        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        r = min(rank, len(s))
        U_r = U[:, :r]
        S_r = s[:r]
        Vt_r = Vt[:r, :]

        right_factor = np.diag(S_r) @ Vt_r

        rank_label = self.fresh_rank_label()

        left_axis_labels = tuple(node.axis_labels[ax] for ax in left_axes) + (f"rank:{rank_label}",)
        right_axis_labels = (f"rank:{rank_label}",) + tuple(node.axis_labels[ax] for ax in right_axes)

        left_shape = tuple(node.data.shape[ax] for ax in left_axes) + (r,)
        right_shape = (r,) + tuple(node.data.shape[ax] for ax in right_axes)

        left_node = TensorNode(
            name=f"{node_name}_L_{rank_label}",
            data=U_r.reshape(left_shape),
            axis_labels=left_axis_labels,
        )

        right_node = TensorNode(
            name=f"{node_name}_R_{rank_label}",
            data=right_factor.reshape(right_shape),
            axis_labels=right_axis_labels,
        )

        del self.nodes[node_name]
        self.nodes[left_node.name] = left_node
        self.nodes[right_node.name] = right_node
        self.edges.append((left_node.name, right_node.name, rank_label))

    def summary(self) -> str:
        lines = ["ExecutedNetwork:"]
        lines.append(f"  storage_cost = {self.storage_cost()}")
        lines.append(f"  num_nodes = {len(self.nodes)}")
        if self.edges:
            lines.append(f"  num_edges = {len(self.edges)}")
        for name, node in sorted(self.nodes.items()):
            lines.append(
                f"  {name}: shape={node.shape}, "
                f"original_modes={node.original_modes}, "
                f"rank_labels={node.rank_labels}"
            )
        return "\n".join(lines)


def execute_scored_sketch(
    X: TensorData,
    sketch: Sketch,
    rank_assignment: RankAssignmentResult,
) -> ExecutedNetwork:
    """
    Execute a scored sketch by applying each OSplit with its assigned rank.

    The sketch is symbolic.
    rank_assignment supplies the concrete rank for each subset.
    """
    net = ExecutedNetwork.from_tensor(X)
    d = X.order

    for op in sketch.ops:
        subset = canonical_subset(op.subset, d)
        rank = rank_assignment.ranks[subset]
        net.execute_osplit(subset=subset, rank=rank)

    return net


if __name__ == "__main__":
    import numpy as np

    from preprocessing import preprocess_singular_values
    from scoring import assign_ranks_greedily
    from sketch import make_sketch

    X = TensorData(np.random.randn(8, 9, 10, 11))
    pre = preprocess_singular_values(X)

    sketch = make_sketch([
        (0,),
        (1,),
        (0, 1),
    ])

    rank_result = assign_ranks_greedily(pre, sketch, eps=0.10)
    net = execute_scored_sketch(X, sketch, rank_result)

    print("assigned ranks:", rank_result.ranks)
    print(net.summary())