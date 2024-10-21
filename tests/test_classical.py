"""Tests for genomic prediction using classical algorithms."""

import logging
from pathlib import Path
import numpy as np
import pytest  # noqa: F401
from numpy.linalg import eigvalsh
from numpy.linalg import pinv
from quantum_inspired_algorithms.visualization import plot_solution
from scipy.sparse.linalg import cg
from genomic_prediction.utils import construct_A_b
from genomic_prediction.utils import construct_A_b_no_X
from genomic_prediction.utils import find_top_indices
from genomic_prediction.utils import get_low_rank_approx
from genomic_prediction.utils import load_data
from genomic_prediction.utils import normalize

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

path = Path(Path(__file__).parent.resolve(), "data", "full_rank")


def test_data():
    """Test assumptions on data."""
    # Load matrix
    A, b, _, _, y, W, Z, X, P, _, _ = load_data(path)

    # Check preconditioner
    P2 = np.diag(np.diag(A))
    assert np.allclose(P, P2)

    # Check reconstruction of A and b
    A2, b2 = construct_A_b(W, Z, X, y)

    assert np.allclose(A, A2)
    assert np.allclose(b, b2)

    # Check if A is symmetric
    assert np.allclose(A, A.T)

    # Check if A is positive semi-definite
    eigenvalues = np.flip(eigvalsh(A))
    tol = 1e-8
    assert np.all(eigenvalues >= -tol)


def test_cg():
    """Test CG."""
    A, b, x_sol, _, _, _, _, _, P, _, _ = load_data(path)
    x_cg, _ = cg(A, b, M=P, atol=1e-5)
    assert np.allclose(np.squeeze(x_sol), x_cg, rtol=1e-01, atol=1e-04)

    _, _, x_sol, _, y, W, Z, X, _, top_size_x, _ = load_data(path)
    A2, b2 = construct_A_b(W, Z, X, y)
    P2 = np.diag(np.diag(A2))
    x_cg2, _ = cg(A2, b2, M=P2, atol=1e-5)
    assert np.allclose(np.squeeze(x_sol), x_cg2, rtol=1e-01, atol=1e-04)

    assert np.allclose(x_cg, x_cg2, rtol=1e-07, atol=1e-07)

    x_idx = find_top_indices(np.abs(x_cg), top_size_x)
    n_matches = plot_solution(
        x_sol,
        x_idx,
        "test_cg",
    )
    assert n_matches == top_size_x


def test_cg_low_rank():
    """Test CG with low-rank genotype data."""
    _, _, x_sol, ebv, y, W, Z, X, _, top_size_x, top_size_ebv = load_data(path)
    A, b = construct_A_b(W, Z, X, y, rank_Z=80)
    P = np.diag(np.diag(A))
    x_cg, _ = cg(A, b, M=P, atol=1e-5)

    x_idx = find_top_indices(np.abs(x_cg), top_size_x)
    n_matches = plot_solution(x_sol, x_idx, "test_cg_low_rank_x")
    assert n_matches == 9

    ebv_cg = Z @ x_cg[X.shape[1] :]
    ebv_idx = find_top_indices(np.abs(ebv_cg), top_size_ebv)
    n_matches = plot_solution(ebv, ebv_idx, "test_cg_low_rank_ebv", expected_solution=ebv, solution=ebv_cg)
    assert n_matches == 31


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


def test_approximate_ridge():
    """Test approximate ridge regression."""
    # Load data
    _, _, x_sol, ebv, y, W, Z, X, _, top_size_x, top_size_ebv = load_data(path)
    rank = 80

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Compute low-rank approximation of `XWZ`
    XWZ = np.hstack([X, WZ])
    XWZ = get_low_rank_approx(XWZ, rank)

    # Solve using approximate ridge regression
    var_e = 0.7
    var_g = 0.3 / Z.shape[1] / 0.5
    alpha = var_e / var_g

    X_n_cols = X.shape[1]
    U, S, Vt = np.linalg.svd(XWZ, full_matrices=False)
    D = np.zeros(S.shape[0])
    D[X_n_cols:] = S[X_n_cols:] / (S[X_n_cols:] ** 2 + alpha)
    x_rr = np.squeeze(Vt.T @ (D[:, None] * (U.T @ y)))

    x_idx = find_top_indices(np.abs(x_rr), top_size_x)
    n_matches = plot_solution(x_sol, x_idx, "test_approximate_ridge_x")
    assert n_matches == 3

    ebv_rr = Z @ x_rr[X_n_cols:]
    ebv_idx = find_top_indices(np.abs(ebv_rr), top_size_ebv)
    n_matches = plot_solution(ebv, ebv_idx, "test_approximate_ridge_ebv", expected_solution=ebv, solution=ebv_rr)
    assert n_matches == 30


def test_ridge_ignore_X():
    """Test approximate ridge regression while ignoring fixed effects."""
    # Load data
    _, _, _, ebv, y, W, Z, _, _, _, top_size_ebv = load_data(path)
    y = np.squeeze(y - np.mean(y))  # simple compensation of fixed effects
    rank = 80

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Compute low-rank approximation
    WZ = get_low_rank_approx(WZ, rank)

    # Solve using ridge regression
    var_e = 0.7
    var_g = 0.3 / Z.shape[1] / 0.5
    alpha = var_e / var_g

    U, S, Vt = np.linalg.svd(WZ, full_matrices=False)
    good_idx = S > 1e-15
    D = np.zeros(S.shape[0])
    D[good_idx] = S[good_idx] / (S[good_idx] ** 2 + alpha)
    x_rr = np.squeeze(Vt.T @ (D[:, None] * (U.T @ y[:, None])))

    ebv_rr = np.squeeze(Z @ x_rr)

    ebv_idx = find_top_indices(np.abs(ebv_rr), top_size_ebv)
    n_matches = plot_solution(
        ebv, ebv_idx, "test_ridge_ignore_X", expected_solution=normalize(ebv), solution=normalize(ebv_rr)
    )
    assert n_matches == 28


def test_least_squares_ignore_X():
    """Test using least-squares while ignoring fixed effects."""
    # Load data
    _, _, _, ebv, y, W, Z, _, _, _, top_size_ebv = load_data(path)
    y = y - np.mean(y)

    for rank, n_matches_expected in [(None, 16), (80, 24)]:
        # Leave out non-phenotyped animals
        WZ = W @ Z
        if rank is not None:
            WZ = get_low_rank_approx(WZ, rank)

        # Solve using pseudoinverse
        x_sol_pinv = np.squeeze(pinv(WZ).dot(y))
        ebv_pinv = Z @ x_sol_pinv

        ebv_idx = find_top_indices(np.abs(ebv_pinv), top_size_ebv)
        plot_name = "test_least_squares_ignore_X"
        if rank is not None:
            plot_name += f"_rank_{rank}"
        n_matches = plot_solution(
            ebv,
            ebv_idx,
            plot_name,
            expected_solution=normalize(ebv),
            solution=normalize(ebv_pinv),
        )
        assert n_matches == n_matches_expected
