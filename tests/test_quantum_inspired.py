"""Tests for genomic prediction."""

import logging
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import pandas as pd
import pytest
from numpy.linalg import eigvalsh
from numpy.linalg import norm
from numpy.linalg import pinv
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from quantum_inspired_algorithms.estimator import QILinearEstimator
from quantum_inspired_algorithms.visualization import plot_solution
from scipy.sparse.linalg import cg

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def _load_data():
    """Load test data."""
    # Define path to data
    path = Path(Path(__file__).parent.resolve(), "data", "full_rank")

    # Load data
    A = np.load(Path(path, "lhssnp.npy"))
    b = np.load(Path(path, "rhssnp.npy"))
    x_sol = np.load(Path(path, "solsnp.npy"))
    ebv = np.load(Path(path, "ebv_snp.npy"))  # estimated breeding values (using PCG)
    y = np.load(Path(path, "phen.npy"))
    W = np.load(Path(path, "W.npy"))
    Z_1 = np.load(Path(path, "Z.npy"))
    f1 = h5py.File(Path(path, "Z.h5"), "r")
    Z_2 = f1["my_dataset"][()]
    assert np.allclose(Z_1, Z_2)
    Z = f1["my_dataset"]
    X = np.load(Path(path, "X.npy"))
    P = np.load(Path(path, "m1.npy"))

    # Define parameters for analysis
    top_percent_ebv = 0.05
    top_size_ebv = int(top_percent_ebv * ebv.size)

    top_percent_x = 0.005
    top_size_x = int(top_percent_x * x_sol.size)

    return A, b, x_sol, ebv, y, W, Z, X, P, top_size_x, top_size_ebv


def _normalize(array: NDArray) -> NDArray:
    return array / norm(array)


def _find_top_indices(x: ArrayLike, top_size: int) -> NDArray:
    """Find indices corresponding to the `top_size` largest entries in x."""
    return np.flip(np.argsort(x))[:top_size]


def _get_low_rank_approx(matrix: NDArray, rank: int) -> NDArray:
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S[rank:] = 0
    return U @ np.diag(S) @ Vt


def _construct_A_b_no_X(W: NDArray, Z: NDArray, y: NDArray, rank_Z: Optional[int] = None) -> tuple[NDArray, NDArray]:
    var_e = 0.7
    var_g = 0.3 / Z.shape[1] / 0.5
    WZ = W @ Z
    if rank_Z is not None:
        WZ = _get_low_rank_approx(WZ, rank_Z)
    A = WZ.T @ WZ * 1 / var_e
    diag_load_idx = np.diag_indices(A.shape[0])
    A[diag_load_idx] += 1 / var_g
    b = WZ.T @ y * 1 / var_e

    return A, b


def _construct_A_b(
    W: NDArray, Z: NDArray, X: NDArray, y: NDArray, rank_Z: Optional[int] = None
) -> tuple[NDArray, NDArray]:
    # Construct A and b
    var_e = 0.7
    var_g = 0.3 / Z.shape[1] / 0.5
    WZ = W @ Z
    if rank_Z is not None:
        WZ = _get_low_rank_approx(WZ, rank_Z)
    XWZ = np.hstack([X, WZ])
    A = XWZ.T @ XWZ * 1 / var_e
    diag_load_row_idx, diag_load_col_idx = np.diag_indices(A.shape[0])
    diag_load_idx = (diag_load_row_idx[X.shape[1] :], diag_load_col_idx[X.shape[1] :])
    A[diag_load_idx] += 1 / var_g
    b = XWZ.T @ y * 1 / var_e

    return A, b


def test_data():
    """Test assumptions on data."""
    # Load matrix
    A, b, _, _, y, W, Z, X, P, _, _ = _load_data()

    # Check preconditioner
    P2 = np.diag(np.diag(A))
    assert np.allclose(P, P2)

    # Check reconstruction of A and b
    A2, b2 = _construct_A_b(W, Z, X, y)

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
    A, b, x_sol, _, _, _, _, _, P, _, _ = _load_data()
    x_cg, _ = cg(A, b, M=P, atol=1e-5)
    assert np.allclose(np.squeeze(x_sol), x_cg, rtol=1e-01, atol=1e-04)

    _, _, x_sol, _, y, W, Z, X, _, top_size_x, _ = _load_data()
    A2, b2 = _construct_A_b(W, Z, X, y)
    P2 = np.diag(np.diag(A2))
    x_cg2, _ = cg(A2, b2, M=P2, atol=1e-5)
    assert np.allclose(np.squeeze(x_sol), x_cg2, rtol=1e-01, atol=1e-04)

    assert np.allclose(x_cg, x_cg2, rtol=1e-07, atol=1e-07)

    x_idx = _find_top_indices(np.abs(x_cg), top_size_x)
    n_matches = plot_solution(
        x_sol,
        x_idx,
        "test_cg",
    )
    assert n_matches == top_size_x

def test_cg_low_rank_ignore_X():
    """Test CG with low-rank genotype data while ignoring fixed effects."""
    _, _, x_sol, ebv, y, W, Z, _, _, top_size_x, top_size_ebv = _load_data()
    y = y - np.mean(y)  # simple compensation of fixed effects
    A, b = _construct_A_b_no_X(W, Z, y, rank_Z=80)
    P = np.diag(np.diag(A))
    x_cg, _ = cg(A, b, M=P, atol=1e-5)

    x_idx = _find_top_indices(np.abs(x_cg), top_size_x)
    n_matches = plot_solution(x_sol, x_idx, "test_cg_low_rank_ignore_X_x")
    assert n_matches == 1

    ebv_cg = Z @ x_cg
    ebv_idx = _find_top_indices(np.abs(ebv_cg), top_size_ebv)
    n_matches = plot_solution(ebv, ebv_idx, "test_cg_low_rank_ignore_X_ebv", expected_solution=ebv, solution=ebv_cg)
    assert n_matches == 28

def test_cg_ignore_X():
    """Test CG with low-rank genotype data while ignoring fixed effects."""
    _, _, x_sol, ebv, y, W, Z, _, _, top_size_x, top_size_ebv = _load_data()
    y = y - np.mean(y)  # simple compensation of fixed effects
    A, b = _construct_A_b_no_X(W, Z, y, rank_Z=80)
    P = np.diag(np.diag(A))
    x_cg, _ = cg(A, b, M=P, atol=1e-5)

    x_idx = _find_top_indices(np.abs(x_cg), top_size_x)
    n_matches = plot_solution(x_sol, x_idx, "test_cg_low_rank_ignore_X_x")
    assert n_matches == 1

    ebv_cg = Z @ x_cg
    ebv_idx = _find_top_indices(np.abs(ebv_cg), top_size_ebv)
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
        _, _, x_sol, ebv, y, W, Z, _, _, top_size_x, top_size_ebv = _load_data()
        y = y - np.mean(y)  # simple compensation of fixed effects
        A, b = _construct_A_b_no_X(W, Z, y, rank_Z=int(np.ceil(rank)))
        P = np.diag(np.diag(A))
        x_cg, _ = cg(A, b, M=P, atol=1e-5)

        x_idx = _find_top_indices(np.abs(x_cg), top_size_x)
        n_matches = plot_solution(x_sol, x_idx, "test_cg_low_rank_ignore_X_x")

        ebv_cg = Z @ x_cg
        ebv_idx = _find_top_indices(np.abs(ebv_cg), top_size_ebv)
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


def test_approximate_ridge():
    """Test approximate ridge regression."""
    # Load data
    _, _, x_sol, ebv, y, W, Z, X, _, top_size_x, top_size_ebv = _load_data()
    rank = 80

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Compute low-rank approximation of `XWZ`
    XWZ = np.hstack([X, WZ])
    XWZ = _get_low_rank_approx(XWZ, rank)

    # Solve using approximate ridge regression
    var_e = 0.7
    var_g = 0.3 / Z.shape[1] / 0.5
    alpha = var_e / var_g

    X_n_cols = X.shape[1]
    U, S, Vt = np.linalg.svd(XWZ, full_matrices=False)
    D = np.zeros(S.shape[0])
    D[X_n_cols:] = S[X_n_cols:] / (S[X_n_cols:] ** 2 + alpha)
    x_rr = np.squeeze(Vt.T @ (D[:, None] * (U.T @ y)))

    x_idx = _find_top_indices(np.abs(x_rr), top_size_x)
    n_matches = plot_solution(x_sol, x_idx, "test_approximate_ridge_x")
    assert n_matches == 3

    ebv_rr = Z @ x_rr[X_n_cols:]
    ebv_idx = _find_top_indices(np.abs(ebv_rr), top_size_ebv)
    n_matches = plot_solution(ebv, ebv_idx, "test_approximate_ridge_ebv", expected_solution=ebv, solution=ebv_rr)
    assert n_matches == 30


def test_ridge_ignore_X():
    """Test approximate ridge regression while ignoring fixed effects."""
    # Load data
    _, _, _, ebv, y, W, Z, _, _, _, top_size_ebv = _load_data()
    y = np.squeeze(y - np.mean(y))  # simple compensation of fixed effects
    rank = 80

    # Leave out non-phenotyped animals
    WZ = W @ Z

    # Compute low-rank approximation
    WZ = _get_low_rank_approx(WZ, rank)

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

    ebv_idx = _find_top_indices(np.abs(ebv_rr), top_size_ebv)
    n_matches = plot_solution(
        ebv, ebv_idx, "test_ridge_ignore_X", expected_solution=_normalize(ebv), solution=_normalize(ebv_rr)
    )
    assert n_matches == 28


def test_least_squares_ignore_X():
    """Test using least-squares while ignoring fixed effects."""
    # Load data
    _, _, _, ebv, y, W, Z, _, _, _, top_size_ebv = _load_data()
    y = y - np.mean(y)

    for rank, n_matches_expected in [(None, 16), (80, 24)]:
        # Leave out non-phenotyped animals
        WZ = W @ Z
        if rank is not None:
            WZ = _get_low_rank_approx(WZ, rank)

        # Solve using pseudoinverse
        x_sol_pinv = np.squeeze(pinv(WZ).dot(y))
        ebv_pinv = Z @ x_sol_pinv

        ebv_idx = _find_top_indices(np.abs(ebv_pinv), top_size_ebv)
        plot_name = "test_least_squares_ignore_X"
        if rank is not None:
            plot_name += f"_rank_{rank}"
        n_matches = plot_solution(
            ebv,
            ebv_idx,
            plot_name,
            expected_solution=_normalize(ebv),
            solution=_normalize(ebv_pinv),
        )
        assert n_matches == n_matches_expected


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
    _, _, x_sol, _, _, W, Z, X, _, _, top_size_ebv = _load_data()
    rank = 3    
    
    # Simulate specific rank for Z
    Z = _get_low_rank_approx(Z, rank)

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
        func = lambda arg: (arg**2 + var_e / var_g) / arg
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
        expected_solution=_normalize(ebv[unique_sampled_indices]),
        solution=_normalize(unique_sampled_ebv),
        expected_counts=n_entries_b * np.abs(ebv / norm(ebv))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    if method == "ridge":
        assert n_matches == 23
    elif method == "ordinary":
        assert n_matches == 29
    else:
        assert False

def test_qi_no_approx_no_X(method: str):
    """Test quantum-inspired regression and no fixed effects."""
    # Load data
    _, _, x_sol, _, _, W, Z, X, _, _, top_size_ebv = _load_data()
    # Simulate `ebv` based on a previous solution
    ebv = np.squeeze(Z @ x_sol[X.shape[1] :])  # use `x_sol` for convenience
    
    Z = Z @ np.identity(Z.shape[1]) # necessary for Dataset conversion

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
        func = lambda arg: (arg**2 + var_e / var_g) / arg
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
        expected_solution=_normalize(ebv[unique_sampled_indices]),
        solution=_normalize(unique_sampled_ebv),
        expected_counts=n_entries_b * np.abs(ebv / norm(ebv))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    print(n_matches)

    if method == "ridge":
        assert n_matches == 2
    elif method == "ordinary":
        assert n_matches == 2
    else:
        assert False


if __name__ == "__main__":
    test_qi_no_X("ordinary")
    test_qi_no_X("ridge")
    test_cg_saturated_rank_ignore_X()
    test_qi_no_approx_no_X("ordinary")
    test_qi_no_approx_no_X("ridge")
