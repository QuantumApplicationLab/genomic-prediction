"""Tests for genomic prediction."""

import logging
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest  # noqa: F401
import seaborn as sns
from numpy import linalg as la
from quantum_inspired_algorithms.quantum_inspired import compute_ls_probs
from quantum_inspired_algorithms.sketching import FKV
from quantum_inspired_algorithms.sketching import Halko
from quantum_inspired_algorithms.visualization import compute_n_matches
from scipy.sparse.linalg import cg
from sklearn.utils.extmath import randomized_svd
from genomic_prediction.plotting import save_fig
from genomic_prediction.utils import construct_A_b
from genomic_prediction.utils import construct_A_b_no_X
from genomic_prediction.utils import find_top_indices
from genomic_prediction.utils import get_low_dimensional_projector
from genomic_prediction.utils import load_data

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

path = Path(Path(__file__).parent.resolve(), "data", "full_rank")


@pytest.mark.parametrize(
    "method",
    [
        ("fkv"),
        ("halko"),
        ("randomized_svd"),
        ("full_svd"),
    ],
)
def test_randomized_low_rank(method: str):
    """Test low-rank pseudoinverse based on randomized algos."""
    # Load data
    _, _, x_sol, _, _, W, Z, X, _, _, top_size_ebv = load_data(path)

    # Simulate `ebv` based on a previous solution
    ebv = np.squeeze(Z @ x_sol[X.shape[1] :])  # use `x_sol` for convenience

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Simulate direct observations
    y = ebv[WZ.shape[0] :]

    # Solve using FKV or regular SVD for increasing rank
    step = 10
    ranks = list(range(step, WZ.shape[0] + step, step))

    if method in ["fkv", "halko"]:
        WZ_ls_prob_rows, WZ_ls_prob_columns_2d, WZ_ls_prob_columns, _, WZ_frobenius = compute_ls_probs(WZ)
        random_states = range(10)

        if method == "fkv":
            r_values = [500]
            c_values = range(1000, 10000, 1000)
        elif method == "halko":
            r_values = ["auto", 200, 300, 400, 500]
            c_values = ["auto"]

        for r in r_values:
            for c in c_values:
                data_to_plot = defaultdict(list)
                for rank in ranks:
                    for random_state in random_states:
                        # Approximate SVD using sketching
                        rng = np.random.RandomState(random_state)
                        if method == "fkv":
                            if isinstance(r, int) and isinstance(c, int):
                                sketcher = FKV(WZ, r, c, WZ_ls_prob_rows, WZ_ls_prob_columns_2d, WZ_frobenius, rng)
                            else:
                                assert False
                        elif method == "halko":
                            if r == "auto":
                                r_auto = rank
                            elif isinstance(r, int):
                                r_auto = r
                            else:
                                assert False
                            if c == "auto":
                                c_auto = rank
                            elif isinstance(c, int):
                                c_auto = c
                            else:
                                assert False
                            sketcher = Halko(WZ, r_auto, c_auto, WZ_ls_prob_rows, WZ_ls_prob_columns, rng)

                        C = sketcher.right_project(sketcher.left_project(WZ))
                        w_left, S, w_right = la.svd(C, full_matrices=False)
                        V = sketcher.left_project(WZ).T @ (w_left[:, :rank] / S[None, :rank])

                        # Estimate lambdas
                        lambdas = []
                        for ell in range(rank):
                            lambdas.append(1 / (S[ell]) ** 2 * np.sum(WZ * np.outer(y, V[:, ell])))

                        # Predict
                        ebv_fkv = sketcher.right_project(Z) @ (w_right.T[:, :rank] @ np.asarray(lambdas)[:, None])
                        ebv_fkv = np.squeeze(ebv_fkv)

                        # Compute number of matches
                        ebv_idx = find_top_indices(np.abs(ebv_fkv), top_size_ebv)
                        n_matches = compute_n_matches(ebv, ebv_idx)

                        # Save data for plotting
                        data_to_plot["rank"].append(rank)
                        data_to_plot["n_matches"].append(n_matches)
                        data_to_plot["random_state"].append(random_state)

                # Plot
                ax = sns.boxplot(data=data_to_plot, x="rank", y="n_matches", fill=False)
                ax.set_ylim(0, top_size_ebv)
                save_fig(f"test_randomized_low_rank_{method}_r{r}_c{c}")
    else:
        data_to_plot = defaultdict(list)
        for rank in ranks:
            # Compute SVD
            if method == "randomized_svd":
                _, S, VT = randomized_svd(WZ, n_components=rank, random_state=10)
                V = VT.T
            elif method == "full_svd":
                U, S, V = np.linalg.svd(WZ, full_matrices=False)
                V2 = WZ.T @ (U[:, :rank] / S[None, :rank])
                V = V.T[:, :rank]
                assert np.allclose(V2, V)

            # Estimate lambdas
            lambdas = []
            for ell in range(rank):
                lambdas.append(1 / (S[ell]) ** 2 * np.sum(WZ * np.outer(y, V[:, ell])))

            # Predict
            ebv_fkv = Z @ (V @ np.asarray(lambdas)[:, None])
            ebv_fkv = np.squeeze(ebv_fkv)

            # Compute number of matches
            ebv_idx = find_top_indices(np.abs(ebv_fkv), top_size_ebv)
            n_matches = compute_n_matches(ebv, ebv_idx)

            # Save data for plotting
            data_to_plot["rank"].append(rank)
            data_to_plot["n_matches"].append(n_matches)

        # Plot
        ax = sns.boxplot(data=data_to_plot, x="rank", y="n_matches", fill=False)
        ax.set_ylim(0, top_size_ebv)
        save_fig(f"test_randomized_low_rank_{method}")


def test_pinv_low_rank():
    """Test low-rank pseudoinverse."""
    # Load data
    _, _, x_sol, _, _, W, Z, X, _, _, top_size_ebv = load_data(path)

    # Simulate `ebv` based on a previous solution
    ebv = np.squeeze(Z @ x_sol[X.shape[1] :])  # use `x_sol` for convenience

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Simulate direct observations
    y = ebv[WZ.shape[0] :]

    # Solve using pseudoinverse for increasing rank
    step = 5
    ranks = list(range(step, WZ.shape[0] + step, step))
    n_matches = []
    U, S, V = np.linalg.svd(WZ, full_matrices=False)
    for rank in ranks:
        # Fit and predict
        WZ_pinv = V[:rank, :].T @ (np.diag(1 / S[:rank])) @ U[:, :rank].T
        ebv_pinv = Z @ WZ_pinv @ y

        # Compute number of matches
        ebv_idx = find_top_indices(np.abs(ebv_pinv), top_size_ebv)
        n_matches.append(compute_n_matches(ebv, ebv_idx))

    # Plot
    data = pd.DataFrame({"rank": ranks, "n_matches": n_matches})
    sns.lineplot(data=data, x="rank", y="n_matches", color="orange")
    save_fig("test_pinv_low_rank")

    # fmt: off
    assert n_matches == [
        6, 9, 13, 11, 13, 17, 19, 22, 23, 24, 25, 27, 27, 32, 28, 31, 33, 31, 33, 34, 36, 34, 36,
        36, 36, 36, 37, 36, 37, 36, 37, 37, 39, 38, 39, 39, 39, 39, 39, 38, 39, 39, 39, 38, 38, 40,
        40, 40, 39, 40, 41, 42, 41, 41, 41, 41, 41, 41, 41, 41, 40, 40, 41, 40, 40, 40, 40, 40, 40,
        40, 41, 41, 42, 42, 41, 41, 41, 42, 42, 42, 43, 43, 43, 43, 43, 43, 45, 44, 45, 44, 44, 45,
        45, 45, 48, 48, 48, 49, 49, 50,
    ]
    # fmt: on


def test_halko_low_rank():
    """Test low-rank pseudoinverse using Halko's algo."""
    # Load data
    _, _, x_sol, _, _, W, Z, X, _, _, top_size_ebv = load_data(path)

    # Simulate `ebv` based on a previous solution
    ebv = np.squeeze(Z @ x_sol[X.shape[1] :])  # use `x_sol` for convenience

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Simulate direct observations
    y = ebv[WZ.shape[0] :]

    # Solve using pseudoinverse for increasing rank
    step = 5
    ranks = list(range(step, WZ.shape[0] + step, step))
    n_matches = []
    for rank in ranks:
        # Reduce dimensionality
        Q = get_low_dimensional_projector(WZ, n_components=rank, random_state=10)
        WZ_reduced = WZ @ Q
        Z_reduced = Z @ Q

        # Fit and predict
        U, S, V = np.linalg.svd(WZ_reduced, full_matrices=False)
        WZ_pinv = V[:rank, :].T @ (np.diag(1 / S[:rank])) @ U[:, :rank].T
        ebv_pinv = Z_reduced @ WZ_pinv @ y

        # Compute number of matches
        ebv_idx = find_top_indices(np.abs(ebv_pinv), top_size_ebv)
        n_matches.append(compute_n_matches(ebv, ebv_idx))

    # Plot
    data = pd.DataFrame({"rank": ranks, "n_matches": n_matches})
    sns.lineplot(data=data, x="rank", y="n_matches", color="orange")
    save_fig("test_halko_low_rank")

    # fmt: off
    assert n_matches == [
        6, 9, 11, 10, 12, 18, 17, 21, 24, 20, 24, 28, 28, 30, 30, 31, 31, 34, 35,
        34, 35, 34, 33, 34, 36, 35, 36, 35, 35, 39, 36, 36, 38, 38, 38, 39, 41, 40,
        40, 36, 41, 39, 39, 41, 39, 38, 41, 40, 40, 39, 39, 41, 41, 42, 40, 40, 39,
        41, 42, 40, 39, 41, 38, 39, 41, 40, 41, 39, 42, 40, 41, 39, 41, 40, 41, 43,
        41, 44, 40, 44, 42, 42, 42, 43, 42, 43, 44, 44, 43, 45, 45, 44, 46, 46, 49,
        48, 48, 49, 49, 50,
    ]
    # fmt: on


def test_cg_halko_low_rank():
    """Test CG with Halko's algo without fixed effects."""
    # Load data
    _, _, x_sol, ebv, y, W, Z, X, _, _, top_size_ebv = load_data(path)

    # Simulate `ebv` based on a previous solution
    ebv = np.squeeze(Z @ x_sol[X.shape[1] :])  # use `x_sol` for convenience

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Simulate direct observations
    y = ebv[WZ.shape[0] :]

    # Reduce dimensionality
    Q = get_low_dimensional_projector(WZ, n_components=210, random_state=10)
    Z_reduced = Z @ Q

    # Construct normal equations
    A, b = construct_A_b_no_X(W, Z_reduced, y)

    # Solve using PCG
    P = np.diag(np.diag(A))
    x_cg, _ = cg(A, b, M=P, atol=1e-5)
    ebv_cg = Z_reduced @ x_cg

    # Compute number of matches
    ebv_idx = find_top_indices(np.abs(ebv_cg), top_size_ebv)
    n_matches = compute_n_matches(ebv, ebv_idx)

    assert n_matches == 40


def test_cg_halko_low_rank_with_fixed_effects():
    """Test CG with Halko's algo and fixed effects."""
    # Load data
    _, _, _, ebv, y, W, Z, X, _, _, top_size_ebv = load_data(path)

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Reduce dimensionality
    Q = get_low_dimensional_projector(WZ, n_components=210, random_state=10)
    Z_reduced = Z @ Q

    # Construct normal equations
    A, b = construct_A_b(W, Z_reduced, X, y)

    # Solve using PCG
    P = np.diag(np.diag(A))
    x_cg, _ = cg(A, b, M=P, atol=1e-5)
    ebv_cg = Z_reduced @ x_cg[X.shape[1] :]

    # Compute number of matches
    ebv_idx = find_top_indices(np.abs(ebv_cg), top_size_ebv)
    n_matches = compute_n_matches(ebv, ebv_idx)

    assert n_matches == 11


if __name__ == "__main__":
    test_randomized_low_rank("halko")
