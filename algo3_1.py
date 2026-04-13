from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tensor_network import TensorNetwork
from algo2_1 import SplitOp, exec_split


AxisLabel = str


@dataclass(frozen=True)
class OSplitOp:
    """
    Output-directed split:

        OSplit(I, r)

    target_free_labels:
        the desired set of free labels that should lie on one side of the new edge
    target_rank:
        requested rank r
    """
    target_free_labels: Tuple[AxisLabel, ...]
    target_rank: int


def free_labels_of_node(node) -> Tuple[AxisLabel, ...]:
    return tuple(lab for lab in node.axis_labels if lab.startswith("free:"))


def rooted_parent_map(G: TensorNetwork, root: str) -> Dict[str, Optional[str]]:
    parent: Dict[str, Optional[str]] = {root: None}
    stack = [root]

    while stack:
        u = stack.pop()
        for v in G.neighbors(u):
            if v not in parent:
                parent[v] = u
                stack.append(v)

    return parent


def subtree_nodes(G: TensorNetwork, start: str, parent: Optional[str]) -> List[str]:
    out: List[str] = []
    stack = [(start, parent)]

    while stack:
        u, p = stack.pop()
        out.append(u)
        for v in G.neighbors(u):
            if v != p:
                stack.append((v, u))

    return out


def subtree_free_labels(G: TensorNetwork, start: str, parent: Optional[str]) -> Set[AxisLabel]:
    labels: Set[AxisLabel] = set()
    for node_name in subtree_nodes(G, start, parent):
        node = G.get_node(node_name)
        for lab in node.axis_labels:
            if lab.startswith("free:"):
                labels.add(lab)
    return labels


def child_subtree_free_labels(
    G: TensorNetwork,
    node_name: str,
) -> Dict[AxisLabel, Set[AxisLabel]]:
    """
    For each incident edge label of node_name, compute the set of free labels
    in the subtree reached through that edge.

    Returns
    -------
    dict:
        edge_label -> set of free labels in that child-side subtree
    """
    result: Dict[AxisLabel, Set[AxisLabel]] = {}
    node = G.get_node(node_name)

    for nb in G.neighbors(node_name):
        nb_node = G.get_node(nb)
        shared = [lab for lab in node.axis_labels if lab in nb_node.axis_labels]
        if len(shared) != 1:
            raise ValueError(
                f"Expected exactly one shared label between {node_name} and {nb}"
            )

        edge_label = shared[0]
        result[edge_label] = subtree_free_labels(G, nb, node_name)

    return result


def convert_osplit_to_split(
    G: TensorNetwork,
    op: OSplitOp,
) -> Optional[SplitOp]:
    """
    Convert OSplit(I, r) into a concrete Split(X, I_s, r) if possible.

    This follows the paper's logic:

    - If some existing edge already realizes the target free-label partition,
      return None (do nothing).
    - Otherwise find a node X whose child-subtree free-label sets can be grouped
      to form the target set.
    - If impossible, return None to indicate failure.

    Returns
    -------
    SplitOp or None
        SplitOp if a concrete node-based split is found.
        None if either:
          - the partition already exists, or
          - no valid conversion exists.
    """
    target = set(op.target_free_labels)

    for node_name, node in G.nodes.items():
        child_map = child_subtree_free_labels(G, node_name)

        local_split_labels: List[AxisLabel] = []
        conflict = False

        for edge_label, free_set in child_map.items():
            if free_set == target:
                # desired partition already exists in the network
                return None

            if free_set < target:
                # this entire subtree should go to the target side
                local_split_labels.append(edge_label)

            elif free_set & target:
                # overlap but not subset => crossing conflict
                conflict = True
                break

        if conflict:
            continue

        # also include free labels that live directly on this node and belong to target
        for lab in node.axis_labels:
            if lab.startswith("free:") and lab in target:
                local_split_labels.append(lab)

        # check whether this node can realize exactly the target side
        realized: Set[AxisLabel] = set()

        for edge_label in local_split_labels:
            if edge_label.startswith("free:"):
                realized.add(edge_label)
            else:
                realized |= child_map[edge_label]

        if realized == target:
            return SplitOp(
                node_name=node_name,
                left_labels=tuple(local_split_labels),
                target_rank=op.target_rank,
            )

    return None


def exec_osplit(
    G: TensorNetwork,
    epsilon: float,
    op: OSplitOp,
):
    """
    Execute an output-directed split.

    Behavior:
    - if the target partition already exists, return (G, epsilon)
    - if a valid conversion exists, call exec_split(...)
    - if no valid conversion exists, return (None, None)
    """
    split_op = convert_osplit_to_split(G, op)

    if split_op is None:
        # Ambiguous case:
        # either the partition already exists, or conversion failed.
        # We distinguish them by checking whether any existing edge realizes target.
        target = set(op.target_free_labels)

        for node_name in G.nodes:
            child_map = child_subtree_free_labels(G, node_name)
            for free_set in child_map.values():
                if free_set == target:
                    return G, epsilon

        return None, None

    return exec_split(G, epsilon, split_op)