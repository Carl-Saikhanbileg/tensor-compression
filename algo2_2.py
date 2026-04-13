from __future__ import annotations

import numpy as np

from typing import Callable, Iterable, Tuple

from tensor_network import TensorNetwork


def line_2_initialize_network(X: np.ndarray) -> TensorNetwork:

    return TensorNetwork.from_dense_tensor(X, root_name="X")


def line_3_initialize_best_network(
    G0: TensorNetwork,
) -> TensorNetwork:
    return G0


def lines_4_to_8_naive_search_loop(
    X,
    G0: TensorNetwork,
    Gmin: TensorNetwork,
    epsilon: float,
    enumerate_programs: Callable[[object], Iterable[object]],
    exec_program: Callable[[object, TensorNetwork, float], Tuple[TensorNetwork | None, float | None]],
    round_network: Callable[[TensorNetwork, float], TensorNetwork],
) -> TensorNetwork:

    for P in enumerate_programs(X):                           # line 4
        G, epsilon_prime = exec_program(P, G0, epsilon)       # line 5

        if G is None or epsilon_prime is None:
            continue

        G = round_network(G, epsilon_prime)                   # line 6

        if G.storage_cost() < Gmin.storage_cost():            # line 7
            Gmin = G                                          # line 8

    return Gmin


def naive_search(
    X,
    epsilon: float,
    enumerate_programs,
    exec_program,
    round_network,
) -> TensorNetwork:
    X_array = X.data if hasattr(X, "data") else X

    G0 = line_2_initialize_network(X_array)                   # line 2
    Gmin = line_3_initialize_best_network(G0)                 # line 3

    Gmin = lines_4_to_8_naive_search_loop(
        X=X,
        G0=G0,
        Gmin=Gmin,
        epsilon=epsilon,
        enumerate_programs=enumerate_programs,
        exec_program=exec_program,
        round_network=round_network,
    )

    return Gmin                                               # line 9