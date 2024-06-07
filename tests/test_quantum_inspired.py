"""Tests for the genomic_prediction.quantum_inspired module."""

from pathlib import Path
import numpy as np
import pytest
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.sparse.linalg import cg
from genomic_prediction import quantum_inspired as qi
from genomic_prediction.utils.visualization import plot_solution


def find_top_indices(x: ArrayLike, top_size: int) -> NDArray:
    """Find indices corresponding to the `top_size` largest entires in `x`."""
    return np.flip(np.argsort(x))[:top_size]


def load_data():
    """Load test data."""
    # Define path to data
    path = Path(Path(__file__).parent.resolve(), "data")

    # Load data
    A = np.load(Path(path, "lhssnp.npy"))
    b = np.load(Path(path, "rhssnp.npy"))
    x_sol = np.load(Path(path, "solsnp.npy"))
    P = np.load(Path(path, "m1.npy"))

    # Define parameters for analysis
    top_percent = 0.01
    size = x_sol.size
    top_size = int(top_percent * size)

    return A, b, x_sol, P, top_size


def test_cg():
    """Test CG."""
    A, b, x_sol, P, top_size = load_data()
    x_cg, _ = cg(A, b, M=P, atol=1e-5)
    x_idx = find_top_indices(x_cg, top_size)
    plot_solution(x_sol, x_idx, top_size)

    assert True


def test_fkv():
    """Test FKV."""
    A, b, x_sol, P, top_size = load_data()
    rank = 5
    r = 50
    c = 50
    x_fkv = qi.linear_eqs_fkv(P @ A, P @ b, r, c, rank)
    x_idx = find_top_indices(x_fkv, top_size)
    plot_solution(x_sol, x_idx, top_size)

    assert True


def test_qi():
    """Test quantum-inspired algo."""
    A, b, x_sol, P, top_size = load_data()
    rank = 5
    r = 50
    c = 50
    n_samples = 50
    n_comp_x = top_size
    x_idx, _ = qi.linear_eqs(P @ A, P @ b, r, c, rank, n_samples, n_comp_x)
    plot_solution(x_sol, x_idx, top_size)

    assert True


if __name__ == "__main__":
    test_cg()
