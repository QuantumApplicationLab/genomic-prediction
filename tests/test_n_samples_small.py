"""Tests for genomic prediction using quantum-inspired algorithms."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytest  # noqa: F401
from quantum_inspired_algorithms.estimator import QILinearEstimator
from quantum_inspired_algorithms.visualization import compute_n_matches
from genomic_prediction.utils import load_data

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

path = Path(Path(__file__).parent.resolve(), "data", "full_rank")


@pytest.mark.parametrize(
    "method",
    [
        ("ordinary"),
        ("ridge"),
    ],
)
def test_qi_no_X_n_samples():
    """Test quantum-inspired regression and no fixed effects."""
    # Load data
    _, _, x_sol, _, _, W, Z, X, _, _, top_size_ebv = load_data(path)
    rank = 200

    
    Z = Z @ np.identity(Z.shape[1]) # necessary for Dataset conversion

    # Simulate `ebv` based on a previous solution
    ebv = np.squeeze(Z @ x_sol[X.shape[1] :])  # use `x_sol` for convenience

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Simulate direct observations
    y = ebv[WZ.shape[0] :]

    # Solve using quantum-inspired algorithm
    r = 210
    c = 210
    n_samples = np.linspace(10,500, num=10)
    n_entries_b = 1000
    func = None
    random_state = 2

    n_matches = []

    for n_sample in n_samples:

        n_sample = int(np.ceil(n_sample))
        print(f"n_samples: {n_sample} out of {WZ.shape[0] * WZ.shape[1]}")

        print(f"\nrandom_state: {random_state}")
        rng = np.random.RandomState(random_state)

        # Solve
        qi = QILinearEstimator(r, c, rank, n_sample, rng, func=func)
        qi = qi.fit(WZ, y)

        #Compare results
        sampled_indices, sampled_ebv = qi.predict_b(Z, n_entries_b)
        df = pd.DataFrame({"ebv_idx_samples": sampled_indices, "ebv_samples": sampled_ebv})
        df_mean = df.groupby("ebv_idx_samples")["ebv_samples"].mean()
        unique_sampled_ebv = np.asarray(df_mean.values)
        sort_idx = np.flip(np.argsort(np.abs(unique_sampled_ebv)))
        unique_sampled_indices = np.asarray(df_mean.keys())
        ebv_idx = unique_sampled_indices[sort_idx][:top_size_ebv]

        n_matches.append(compute_n_matches(ebv, ebv_idx))

    assert(n_matches == [3, 0, 3, 3, 3, 2, 6, 3, 4, 3])
        
if __name__ == "__main__":
    test_qi_no_X_n_samples()