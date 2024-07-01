"""Tests for the genomic_prediction.quantum_inspired module."""

from pathlib import Path
import numpy as np
import plotext as plt
import pytest
from numpy.linalg import eigvalsh
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.sparse.linalg import cg
from genomic_prediction import quantum_inspired as qi
from genomic_prediction.utils.visualization import plot_solution


def _find_top_indices(x: ArrayLike, top_size: int) -> NDArray:
    """Find indices corresponding to the `top_size` largest entires in `x`."""
    return np.flip(np.argsort(x))[:top_size]


def _load_data(rank_genotype=1000):
    """Load test data."""
    # Define path to data
    path_low_rank = Path(Path(__file__).parent.resolve(), "data", f"{rank_genotype}")
    path_full_rank = Path(Path(__file__).parent.resolve(), "data", "full_rank")

    # Load data
    A = np.load(Path(path_low_rank, "lhssvd.npy"))
    b = np.load(Path(path_low_rank, "rhssvd.npy"))
    x_sol = np.load(Path(path_full_rank, "solsnp.npy"))
    P = np.load(Path(path_low_rank, "msvd.npy"))

    # Define parameters for analysis
    top_percent = 0.005
    size = x_sol.size
    top_size = int(top_percent * size)

    return A, b, x_sol, P, top_size


def test_data_consistency():
    """Check if data is consistent."""
    path1 = Path(Path(__file__).parent.resolve(), "data", "full_rank")
    A_to_check1 = np.load(Path(path1, "lhssnp.npy"))
    path2 = Path(Path(__file__).parent.resolve(), "data", "1000")
    A_to_check2 = np.load(Path(path2, "lhssvd.npy"))

    assert np.allclose(A_to_check1, A_to_check2)


def test_coefficient_matrix():
    """Test assumptions on coefficient matrix."""
    # Load matrix
    A, _, _, _, _ = _load_data()

    # Check if A is symmetric
    assert np.allclose(A, A.T)

    # Check if A is positive semi-definite
    eigenvalues = np.flip(eigvalsh(A))
    tol = 1e-8
    assert np.all(eigenvalues >= -tol)

    # Plot eigenvalues
    plt.plot(np.abs(eigenvalues))
    plt.show()
    plt.clear_figure()

    assert True


def test_cg_low_rank():
    """Test CG with low-rank genotype data."""
    A, b, x_sol, P, top_size = _load_data(rank_genotype=100)
    x_cg, _ = cg(A, b, M=P, atol=1e-5)
    x_idx = _find_top_indices(x_cg, top_size)
    plot_solution(x_sol, x_idx, top_size)

    assert True


def test_cg():
    """Test CG."""
    A, b, x_sol, P, top_size = _load_data()
    x_cg, _ = cg(A, b, M=P, atol=1e-5)
    x_idx = _find_top_indices(x_cg, top_size)
    plot_solution(x_sol, x_idx, top_size)

    assert True


def test_fkv():
    """Test FKV."""
    A, b, x_sol, P, top_size = _load_data()
    rank = 5
    r = 50
    c = 50
    x_fkv = qi.linear_eqs_fkv(P @ A, P @ b, r, c, rank)
    x_idx = _find_top_indices(x_fkv, top_size)
    plot_solution(x_sol, x_idx, top_size)

    assert True


def test_qi():
    """Test quantum-inspired algo."""
    A, b, x_sol, P, top_size = _load_data()
    rank = 5
    r = 50
    c = 50
    n_samples = 50
    n_comp_x = top_size
    x_idx, _ = qi.linear_eqs(P @ A, P @ b, r, c, rank, n_samples, n_comp_x)
    plot_solution(x_sol, x_idx, top_size)

    assert True


if __name__ == "__main__":
    test_cg_low_rank()
