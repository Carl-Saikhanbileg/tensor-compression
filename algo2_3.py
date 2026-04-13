from __future__ import annotations

from typing import Callable, Iterable, Tuple
import numpy as np
from tensor_core import TensorData
from preprocessing import PrecomputedSVals, preprocess_singular_values
from tensor_network import TensorNetwork


def line_2_preprocess(
    X: TensorData,
    epsilon: float,
) -> PrecomputedSVals:
    return preprocess_singular_values(X)

def line_3_initialize_network(X: np.ndarray) -> TensorNetwork:
    return TensorNetwork.from_dense_tensor(X, root_name="X")

def line_4_initialize_best_network(G0: TensorNetwork) -> TensorNetwork:
    return G0


def lines_5_to_9_structure_search_loop(
    Omega,
    X,
    G0: TensorNetwork,
    Gmin: TensorNetwork,
    epsilon: float,
    top_k_programs: Callable[[object, object, float], Iterable[object]],
    exec_program: Callable[[object, TensorNetwork, float], Tuple[TensorNetwork | None, float | None]],
    round_network: Callable[[TensorNetwork, float], TensorNetwork],
) -> TensorNetwork:

    for P in top_k_programs(Omega, X, epsilon):              # line 5
        G, epsilon_prime = exec_program(P, G0, epsilon)      # line 6

        if G is None or epsilon_prime is None:
            continue

        G = round_network(G, epsilon_prime)                  # line 7

        if G.storage_cost() < Gmin.storage_cost():           # line 8
            Gmin = G                                         # line 9

    return Gmin

def structure_search(
    X,
    epsilon: float,
    preprocess,
    top_k_programs,
    exec_program,
    round_network,
):
    Omega = preprocess(X, epsilon)                           # line 2
    G0 = TensorNetwork.from_dense_tensor(
        X.data if hasattr(X, "data") else X,
        root_name="X",
    )                                                        # line 3
    Gmin = G0                                                # line 4

    Gmin = lines_5_to_9_structure_search_loop(
        Omega=Omega,
        X=X,
        G0=G0,
        Gmin=Gmin,
        epsilon=epsilon,
        top_k_programs=top_k_programs,
        exec_program=exec_program,
        round_network=round_network,
    )

    return Gmin