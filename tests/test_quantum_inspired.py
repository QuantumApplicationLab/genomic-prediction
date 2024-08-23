"""Tests for genomic prediction using quantum-inspired algorithms."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytest  # noqa: F401
from quantum_inspired_algorithms.estimator import QILinearEstimator
from quantum_inspired_algorithms.visualization import plot_solution
from genomic_prediction.utils import get_low_rank_approx
from genomic_prediction.utils import load_data
from genomic_prediction.utils import normalize

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

path = Path(Path(__file__).parent.resolve(), "data", "full_rank")


@pytest.mark.parametrize(
    "method",
    [
        ("ordinary"),
        ("ridge"),
    ],
)
def test_qi_no_X(method: str):
    """Test quantum-inspired regression and no fixed effects."""
    # Load data
    _, _, x_sol, _, _, W, Z, X, _, _, top_size_ebv = load_data(path)
    rank = 3

    # Simulate specific rank for Z
    Z = get_low_rank_approx(Z, rank)

    # Simulate `ebv` based on a previous solution
    ebv = np.squeeze(Z @ x_sol[X.shape[1] :])  # use `x_sol` for convenience

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Simulate direct observations
    y = ebv[WZ.shape[0] :]

    # Solve using quantum-inspired algorithm
    r = 170
    c = 140
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

    if method == "ridge":
        assert n_matches == 26
    elif method == "ordinary":
        assert n_matches == 29
    else:
        assert False


if __name__ == "__main__":
    test_qi_no_X("ordinary")
