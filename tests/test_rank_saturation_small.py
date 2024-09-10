"""Tests for genomic prediction."""

import logging
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest  # noqa: F401
import seaborn as sns
from numpy import linalg as la
from quantum_inspired_algorithms.quantum_inspired import compute_C_and_R
from quantum_inspired_algorithms.quantum_inspired import compute_ls_probs
from quantum_inspired_algorithms.visualization import compute_n_matches
from genomic_prediction.plotting import save_fig
from genomic_prediction.utils import find_top_indices
from genomic_prediction.utils import load_data

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

path = Path(Path(__file__).parent.resolve(), "data", "full_rank")


@pytest.mark.parametrize(
    "fkv",
    [
        (True),
        (False),
    ],
)
def test_fkv_low_rank(full_svd: bool):
    """Test low-rank pseudoinverse."""
    # Load data
    _, _, x_sol, _, _, W, Z, X, _, _, top_size_ebv = load_data(path)

    # Simulate `ebv` based on a previous solution
    ebv = np.squeeze(Z @ x_sol[X.shape[1] :])  # use `x_sol` for convenience

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Simulate direct observations
    y = ebv[WZ.shape[0] :]

    # Solve using FKV or regular SVD for increasing rank
    step = 5
    ranks = list(range(step, 220, step))
    data_to_plot = defaultdict(list)

    if not full_svd:
        random_states = range(10)
        WZ_ls_prob_rows, WZ_ls_prob_columns, WZ_row_norms, _, WZ_frobenius = compute_ls_probs(WZ, [])
        r = 500
        for c in range(300, 1000, 100):
            for rank in ranks:
                for random_state in random_states:
                    # Approximate SVD using FKV
                    rng = np.random.RandomState(random_state)
                    C, _, _, WZ_sampled_rows_idx, _ = compute_C_and_R(
                        WZ,
                        r,
                        c,
                        WZ_row_norms,
                        WZ_ls_prob_rows,
                        WZ_ls_prob_columns,
                        WZ_frobenius,
                        rng,
                        [],
                    )
                    w_left, S, _ = la.svd(C, full_matrices=False)
                    R = (
                        WZ[WZ_sampled_rows_idx, :]
                        * WZ_frobenius
                        / (np.sqrt(len(WZ_sampled_rows_idx)) * WZ_row_norms[WZ_sampled_rows_idx, None])
                    )
                    V = R.T @ (w_left[:, :rank] / S[None, :rank])

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
                    data_to_plot["random_state"].append(random_state)
                    data_to_plot["n_rows"].append(C.shape[0])
                    data_to_plot["n_cols"].append(C.shape[1])

            # Plot
            sns.boxplot(data=data_to_plot, x="rank", y="n_matches", fill=False)
            save_fig(f"test_fkv_low_rank_c{c}")

            sns.scatterplot(data=data_to_plot, x="rank", y="n_rows")
            save_fig(f"test_fkv_low_rank_n_rows_c{c}")
            sns.scatterplot(data=data_to_plot, x="rank", y="n_cols")
            save_fig(f"test_fkv_low_rank_n_cols_c{c}")
    else:
        for rank in ranks:
            # Compute SVD
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
        sns.boxplot(data=data_to_plot, x="rank", y="n_matches", fill=False)
        save_fig("test_fkv_low_rank_svd")


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


if __name__ == "__main__":
    test_fkv_low_rank(False)
