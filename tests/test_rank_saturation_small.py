"""Tests for genomic prediction."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytest  # noqa: F401
import seaborn as sns
from quantum_inspired_algorithms.visualization import compute_n_matches
from genomic_prediction.plotting import save_fig
from genomic_prediction.utils import find_top_indices
from genomic_prediction.utils import load_data

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

path = Path(Path(__file__).parent.resolve(), "data", "full_rank")


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
    test_pinv_low_rank()
