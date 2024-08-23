"""Tests for genomic prediction."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytest  # noqa: F401
from quantum_inspired_algorithms.estimator import QILinearEstimator
from quantum_inspired_algorithms.visualization import plot_solution
from scipy.sparse.linalg import cg
from genomic_prediction.utils import construct_A_b_no_X
from genomic_prediction.utils import find_top_indices
from genomic_prediction.utils import load_data
from genomic_prediction.utils import normalize

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


path = Path(Path(__file__).parent.resolve(), "data", "full_rank")


def test_cg_low_rank_ignore_X():
    """Test CG with low-rank genotype data while ignoring fixed effects."""
    _, _, x_sol, ebv, y, W, Z, _, _, top_size_x, top_size_ebv = load_data(path)
    y = y - np.mean(y)  # simple compensation of fixed effects
    A, b = construct_A_b_no_X(W, Z, y, rank_Z=80)
    P = np.diag(np.diag(A))
    x_cg, _ = cg(A, b, M=P, atol=1e-5)

    x_idx = find_top_indices(np.abs(x_cg), top_size_x)
    n_matches = plot_solution(x_sol, x_idx, "test_cg_low_rank_ignore_X_x")
    assert n_matches == 1

    ebv_cg = Z @ x_cg
    ebv_idx = find_top_indices(np.abs(ebv_cg), top_size_ebv)
    n_matches = plot_solution(ebv, ebv_idx, "test_cg_low_rank_ignore_X_ebv", expected_solution=ebv, solution=ebv_cg)
    assert n_matches == 28


def test_cg_ignore_X():
    """Test CG with low-rank genotype data while ignoring fixed effects."""
    _, _, x_sol, ebv, y, W, Z, _, _, top_size_x, top_size_ebv = load_data(path)
    y = y - np.mean(y)  # simple compensation of fixed effects
    A, b = construct_A_b_no_X(W, Z, y, rank_Z=80)
    P = np.diag(np.diag(A))
    x_cg, _ = cg(A, b, M=P, atol=1e-5)

    x_idx = find_top_indices(np.abs(x_cg), top_size_x)
    n_matches = plot_solution(x_sol, x_idx, "test_cg_low_rank_ignore_X_x")
    assert n_matches == 1

    ebv_cg = Z @ x_cg
    ebv_idx = find_top_indices(np.abs(ebv_cg), top_size_ebv)
    n_matches = plot_solution(ebv, ebv_idx, "test_cg_low_rank_ignore_X_ebv", expected_solution=ebv, solution=ebv_cg)
    assert n_matches == 28


def test_cg_saturated_rank_ignore_X():
    """Test CG with low-rank genotype data and varying assumed ranks to analyze when rank saturates."""
    ranks = np.linspace(1, 500, num=100)
    total_n_matches = []
    max_n_found = 0
    max_rank_found = 0
    i_counter = 5

    for rank in ranks:
        _, _, x_sol, ebv, y, W, Z, _, _, top_size_x, top_size_ebv = load_data(path)
        y = y - np.mean(y)  # simple compensation of fixed effects
        A, b = construct_A_b_no_X(W, Z, y, rank_Z=int(np.ceil(rank)))
        P = np.diag(np.diag(A))
        x_cg, _ = cg(A, b, M=P, atol=1e-5)

        x_idx = find_top_indices(np.abs(x_cg), top_size_x)
        n_matches = plot_solution(x_sol, x_idx, "test_cg_low_rank_ignore_X_x")

        ebv_cg = Z @ x_cg
        ebv_idx = find_top_indices(np.abs(ebv_cg), top_size_ebv)
        n_matches = plot_solution(ebv, ebv_idx, "test_cg_low_rank_ignore_X_ebv", expected_solution=ebv, solution=ebv_cg)

        total_n_matches.append(n_matches)

        if n_matches > max_n_found:
            max_n_found = n_matches
        if i_counter == 0:
            increase = max(total_n_matches[-5:]) - min(total_n_matches[-5:])
            if increase < 2 or max_n_found not in total_n_matches[-5:]:
                max_rank_found = rank
                break
            i_counter = 5
        else:
            i_counter -= 1

        print("Rank " + str(int(np.ceil(rank))) + " found " + str(n_matches) + " matches")

    print("Max matches found: " + str(max_n_found) + " saturated at rank " + str(np.ceil(max_rank_found)))
    assert int(np.ceil(max_rank_found)) == 238
    assert max_n_found == 36


def test_qi_no_approx_no_X(method: str):
    """Test quantum-inspired regression and no fixed effects."""
    # Load data
    _, _, x_sol, _, _, W, Z, X, _, _, top_size_ebv = load_data(path)
    # Simulate `ebv` based on a previous solution
    ebv = np.squeeze(Z @ x_sol[X.shape[1] :])  # use `x_sol` for convenience

    Z = Z @ np.identity(Z.shape[1])  # necessary for Dataset conversion

    rank = 233
    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Simulate direct observations
    y = ebv[WZ.shape[0] :]

    # Solve using quantum-inspired algorithm
    r = 1000
    c = 500
    n_samples = 100
    print(f"n_samples: {n_samples} out of {WZ.shape[0] * WZ.shape[1]}")
    n_entries_b = 1000
    func = None
    if method == "ridge":
        var_e = 0.7
        var_g = 0.3 / Z.shape[1] / 0.5

        def _func(arg: float) -> float:
            return (arg**2 + var_e / var_g) / arg

        func = _func
    random_state = 2
    print(f"\nrandom_state: {random_state}")
    rng = np.random.RandomState(random_state)

    # Solve
    qi = QILinearEstimator(r, c, rank, n_samples, rng, func=func)
    qi = qi.fit(WZ, y)

    sampled_indices, sampled_ebv = qi.predict_b(Z, n_entries_b)

    # Find most frequent outcomes
    unique_ebv_idx, counts = np.unique(sampled_indices, return_counts=True)
    sort_idx = np.flip(np.argsort(counts))
    ebv_idx = unique_ebv_idx[sort_idx][:top_size_ebv]

    # Compare results
    df = pd.DataFrame({"ebv_idx_samples": sampled_indices, "ebv_samples": sampled_ebv})
    df_mean = df.groupby("ebv_idx_samples")["ebv_samples"].mean()
    df_counts = df.groupby("ebv_idx_samples").count()
    unique_sampled_indices = df_mean.keys()
    unique_sampled_ebv = np.asarray(df_mean.values)
    n_matches = plot_solution(
        ebv,
        ebv_idx,
        f"{random_state}_test_qi_{method}_no_X_{n_samples}",
        expected_solution=normalize(ebv[unique_sampled_indices]),
        solution=normalize(unique_sampled_ebv),
        expected_counts=n_entries_b * np.abs(normalize(ebv))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    print(n_matches)

    if method == "ridge":
        assert n_matches == 2
    elif method == "ordinary":
        assert n_matches == 2
    else:
        assert False
