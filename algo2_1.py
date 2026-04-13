from __future__ import annotations

from math import prod
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np

from tensor_network import TensorNetwork, TensorNode


AxisLabel = str


def absorb_matrix_into_axis(
    node: TensorNode,
    axis_label: AxisLabel,
    R: np.ndarray,
) -> None:
    """
    Multiply tensor `node` along the axis named `axis_label` by matrix R.

    If the chosen axis has size k and R has shape (k, r),
    then the axis size changes from k to r.
    """
    ax = node.axis_of(axis_label)

    T = np.moveaxis(node.data, ax, -1)          # (..., k)
    T = np.tensordot(T, R, axes=([-1], [0]))    # (..., r)
    T = np.moveaxis(T, -1, ax)

    node.data = T


def qr_push_to_parent(
    G: TensorNetwork,
    child_name: str,
    parent_name: str,
) -> None:
    """
    Orthogonalize one child toward its parent by QR.

    Suppose child and parent share exactly one rank label e.
    We reshape the child tensor as

        M in R^{(all other axes) x (edge axis)}

    and compute
        M = Q R.

    Then:
    - child tensor becomes reshaped Q
    - R is absorbed into the parent along the shared edge
    """
    child = G.get_node(child_name)
    parent = G.get_node(parent_name)

    shared = list(set(child.axis_labels).intersection(parent.axis_labels))
    if len(shared) != 1:
        raise ValueError(
            f"Expected exactly one shared rank label between {child_name} and {parent_name}"
        )

    edge_label = shared[0]
    edge_axis = child.axis_of(edge_label)

    other_axes = [i for i in range(child.order) if i != edge_axis]
    perm = other_axes + [edge_axis]

    Xp = np.transpose(child.data, axes=perm)

    m = int(prod(child.data.shape[i] for i in other_axes))
    n = child.data.shape[edge_axis]
    M = Xp.reshape(m, n)

    Q, R = np.linalg.qr(M, mode="reduced")
    r_new = Q.shape[1]

    new_shape = tuple(child.data.shape[i] for i in other_axes) + (r_new,)
    Q_tensor = Q.reshape(new_shape)

    inverse_perm = np.argsort(perm)
    child.data = np.transpose(Q_tensor, axes=inverse_perm)

    absorb_matrix_into_axis(parent, edge_label, R)


def rooted_tree_order(
    G: TensorNetwork,
    root: str,
) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """
    Build parent map and DFS order for the tree rooted at `root`.
    """
    parent: Dict[str, Optional[str]] = {root: None}
    order: List[str] = []
    stack = [root]

    while stack:
        u = stack.pop()
        order.append(u)

        for v in G.neighbors(u):
            if v not in parent:
                parent[v] = u
                stack.append(v)

    return parent, order


def orthogonalize_rooted_at(
    G: TensorNetwork,
    root: str,
) -> None:
    """
    Line 2 of Algorithm 2.1:
        Orthogonalize G rooted at X by QR decompositions.

    For a tree tensor network:
    - root the tree at `root`
    - process nodes from leaves upward
    - QR each child toward its parent
    """
    parent, order = rooted_tree_order(G, root)

    for u in reversed(order):
        p = parent[u]
        if p is not None:
            qr_push_to_parent(G, u, p)

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import List, Tuple

import numpy as np

from tensor_network import TensorNode



@dataclass(frozen=True)
class DeltaTruncatedSVDResult:
    """
    Result of line 3 in Algorithm 2.1.

    M = U_keep diag(s_keep) Vt_keep + U_tail diag(s_tail) Vt_tail

    where the kept rank is the smallest one such that the tail Frobenius norm
    is <= delta.
    """
    M: np.ndarray
    row_labels: Tuple[AxisLabel, ...]
    col_labels: Tuple[AxisLabel, ...]

    U_keep: np.ndarray
    s_keep: np.ndarray
    Vt_keep: np.ndarray

    U_tail: np.ndarray
    s_tail: np.ndarray
    Vt_tail: np.ndarray

    matrix_fro_norm: float
    delta: float

    @property
    def kept_rank(self) -> int:
        return len(self.s_keep)

    @property
    def tail_error_sq(self) -> float:
        return float(np.sum(self.s_tail ** 2))

    @property
    def tail_error(self) -> float:
        return float(np.sqrt(self.tail_error_sq))


def unfold_node(
    node: TensorNode,
    row_labels: Tuple[AxisLabel, ...],
) -> Tuple[np.ndarray, Tuple[AxisLabel, ...], Tuple[AxisLabel, ...]]:
    """
    Matricize a tensor node with row_labels on rows and the remaining labels on columns.
    """
    row_labels = tuple(row_labels)
    col_labels = tuple(lab for lab in node.axis_labels if lab not in row_labels)

    perm = [node.axis_of(lab) for lab in row_labels + col_labels]
    Xp = np.transpose(node.data, axes=perm)

    row_dim = int(prod(node.data.shape[node.axis_of(lab)] for lab in row_labels))
    col_dim = int(prod(node.data.shape[node.axis_of(lab)] for lab in col_labels))

    M = Xp.reshape(row_dim, col_dim)
    return M, row_labels, col_labels


def smallest_rank_with_tail_bound(
    singular_values: np.ndarray,
    delta: float,
) -> int:
    """
    Return the smallest rank r such that
        sqrt(sum_{i > r} sigma_i^2) <= delta

    With Python indexing:
    - keep singular_values[:r]
    - discard singular_values[r:]
    """
    q = len(singular_values)
    tail_sq = np.cumsum((singular_values[::-1] ** 2))[::-1]

    for r in range(q + 1):
        err_sq = 0.0 if r == q else float(tail_sq[r])
        if err_sq <= delta * delta:
            return r

    return q


def delta_truncated_svd_of_node(
    node: TensorNode,
    row_labels: Tuple[AxisLabel, ...],
    epsilon: float,
) -> DeltaTruncatedSVDResult:
    """
    Line 3 of Algorithm 2.1:

        Compute δ-truncated SVD:
            X^(I) = U Σ V + E,
            δ = ε ||X^(I)||_F,
            ||E||_F <= δ.

    Returns the kept part and discarded tail separately.
    """
    M, row_labels, col_labels = unfold_node(node, row_labels)

    matrix_fro_norm = float(np.linalg.norm(M, ord="fro"))
    delta = epsilon * matrix_fro_norm

    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    keep_rank = smallest_rank_with_tail_bound(s, delta)

    U_keep = U[:, :keep_rank]
    s_keep = s[:keep_rank]
    Vt_keep = Vt[:keep_rank, :]

    U_tail = U[:, keep_rank:]
    s_tail = s[keep_rank:]
    Vt_tail = Vt[keep_rank:, :]

    return DeltaTruncatedSVDResult(
        M=M,
        row_labels=row_labels,
        col_labels=col_labels,
        U_keep=U_keep,
        s_keep=s_keep,
        Vt_keep=Vt_keep,
        U_tail=U_tail,
        s_tail=s_tail,
        Vt_tail=Vt_tail,
        matrix_fro_norm=matrix_fro_norm,
        delta=delta,
    )


def compute_delta_r(
    target_rank: int,
    kept_rank: int,
) -> int:
    return target_rank - kept_rank

def check_split_feasibility(delta_r: int) -> bool:
    return delta_r >= 0



def line_5_fail_if_rank_too_small(delta_r: int):

    if delta_r < 0:
        return None
    return True


def line_6_augment_svd(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    U_delta: np.ndarray,
    s_delta: np.ndarray,
    Vt_delta: np.ndarray,
    delta_r: int,
):

    # Case Δr = 0 → nothing to add
    if delta_r == 0:
        return U, s, Vt

    # take first Δr components from tail
    k = min(delta_r, len(s_delta))

    U_extra = U_delta[:, :k]
    s_extra = s_delta[:k]
    Vt_extra = Vt_delta[:k, :]

    # concatenate
    if len(s) > 0:
        U_prime = np.concatenate([U, U_extra], axis=1)
        s_prime = np.concatenate([s, s_extra], axis=0)
        Vt_prime = np.concatenate([Vt, Vt_extra], axis=0)
    else:
        # edge case: no kept singular values
        U_prime = U_extra
        s_prime = s_extra
        Vt_prime = Vt_extra

    return U_prime, s_prime, Vt_prime


def line_7_build_X1_from_node(
    node: TensorNode,
    row_labels: Sequence[str],
    U_prime: np.ndarray,
) -> np.ndarray:
    """
    Same as line_7_build_X1, but extracts mode sizes from the node
    using the chosen row_labels.
    """
    left_mode_sizes = tuple(
        node.data.shape[node.axis_of(label)]
        for label in row_labels
    )
    r = U_prime.shape[1]
    return U_prime.reshape(left_mode_sizes + (r,))


def line_8_build_X2_from_node(
    node: TensorNode,
    col_labels: Sequence[str],
    s_prime: np.ndarray,
    Vt_prime: np.ndarray,
) -> np.ndarray:
    """
    Same as line_8_build_X2, but extracts right-mode sizes from the node
    using the chosen column labels.
    """
    right_mode_sizes = tuple(
        node.data.shape[node.axis_of(label)]
        for label in col_labels
    )
    r = len(s_prime)
    SV = np.diag(s_prime) @ Vt_prime
    return SV.reshape((r,) + right_mode_sizes)

def line_9_compute_delta_prime(
    singular_values: np.ndarray,
    final_rank: int,
) -> float:
    return float(np.sum(singular_values[final_rank:] ** 2))

def line_10_replace_node_by_split(
    G: TensorNetwork,
    old_name: str,
    X1_data: np.ndarray,
    X1_labels: List[str],
    X2_data: np.ndarray,
    X2_labels: List[str],
) -> TensorNetwork:
    old_node = G.get_node(old_name)
    old_neighbors = G.neighbors(old_name)

    new_rank_label = X1_labels[-1]   # assumes X1 ends with new rank label
    X1_name = f"{old_name}_1"
    X2_name = f"{old_name}_2"

    left_label_set = set(X1_labels)
    right_label_set = set(X2_labels)

    # Remove old node
    G.remove_node(old_name)

    # Add new nodes
    G.add_node(TensorNode(X1_name, X1_data, list(X1_labels)))
    G.add_node(TensorNode(X2_name, X2_data, list(X2_labels)))

    # Internal edge between X1 and X2
    G.connect(X1_name, X2_name, new_rank_label)

    # Reconnect former neighbors of X
    for nb in old_neighbors:
        nb_node = G.get_node(nb)

        # old neighbor should share exactly one label with old_node
        shared = [lab for lab in nb_node.axis_labels if lab in old_node.axis_labels]
        if len(shared) != 1:
            raise ValueError(
                f"Neighbor {nb} did not share exactly one label with {old_name}"
            )

        edge_label = shared[0]

        if edge_label in left_label_set:
            G.connect(nb, X1_name, edge_label)
        elif edge_label in right_label_set:
            G.connect(nb, X2_name, edge_label)
        else:
            raise ValueError(f"Could not reroute old edge label {edge_label}")

    return G


def line_11_return_result(
    G_prime: TensorNetwork,
    delta_prime: float,
    matrix_fro_norm: float,
) -> Tuple[TensorNetwork, float]:

    remaining_error_ratio = float(np.sqrt(delta_prime) / matrix_fro_norm)
    return G_prime, remaining_error_ratio


def exec_split(G, epsilon, op):
    # line 2
    orthogonalize_rooted_at(G, op.node_name)

    # get the node after orthogonalization
    X_node = G.get_node(op.node_name)

    # line 3
    svd_result = delta_truncated_svd_of_node(
        node=X_node,
        row_labels=op.left_labels,
        epsilon=epsilon,
    )

    # line 4
    delta_r = compute_delta_r(
        target_rank=op.target_rank,
        kept_rank=svd_result.kept_rank,
    )

    # line 5
    if line_5_fail_if_rank_too_small(delta_r) is None:
        return None, None

    # line 6
    U_prime, s_prime, Vt_prime = line_6_augment_svd(
        U=svd_result.U_keep,
        s=svd_result.s_keep,
        Vt=svd_result.Vt_keep,
        U_delta=svd_result.U_tail,
        s_delta=svd_result.s_tail,
        Vt_delta=svd_result.Vt_tail,
        delta_r=delta_r,
    )

    # line 7
    X1 = line_7_build_X1_from_node(
        node=X_node,
        row_labels=svd_result.row_labels,
        U_prime=U_prime,
    )

    # line 8
    X2 = line_8_build_X2_from_node(
        node=X_node,
        col_labels=svd_result.col_labels,
        s_prime=s_prime,
        Vt_prime=Vt_prime,
    )

    # line 9
    full_s = np.concatenate([svd_result.s_keep, svd_result.s_tail])
    delta_prime = line_9_compute_delta_prime(
        singular_values=full_s,
        final_rank=len(s_prime),
    )

    # line 10
    new_rank_label = G.fresh_rank_label()
    X1_labels = list(svd_result.row_labels) + [new_rank_label]
    X2_labels = [new_rank_label] + list(svd_result.col_labels)

    G_prime = line_10_replace_node_by_split(
        G=G,
        old_name=op.node_name,
        X1_data=X1,
        X1_labels=X1_labels,
        X2_data=X2,
        X2_labels=X2_labels,
    )

    # line 11
    return line_11_return_result(
        G_prime=G_prime,
        delta_prime=delta_prime,
        matrix_fro_norm=svd_result.matrix_fro_norm,
    )