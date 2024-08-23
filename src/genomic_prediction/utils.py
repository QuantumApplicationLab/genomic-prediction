"""Utility functions for data preparation."""

from pathlib import Path
from typing import Optional
import h5py
import numpy as np
from numpy.linalg import norm
from numpy.typing import ArrayLike
from numpy.typing import NDArray


def load_data(path: Path):
    """Load test data."""
    # Load data
    A = np.load(Path(path, "lhssnp.npy"))
    b = np.load(Path(path, "rhssnp.npy"))
    x_sol = np.load(Path(path, "solsnp.npy"))
    ebv = np.load(Path(path, "ebv_snp.npy"))  # estimated breeding values (using PCG)
    y = np.load(Path(path, "phen.npy"))
    W = np.load(Path(path, "W.npy"))
    Z_1 = np.load(Path(path, "Z.npy"))
    f1 = h5py.File(Path(path, "Z.h5"), "r")
    Z_2 = np.asarray(f1["my_dataset"])
    assert np.allclose(Z_1, Z_2)
    Z = Z_2
    X = np.load(Path(path, "X.npy"))
    P = np.load(Path(path, "m1.npy"))

    # Define parameters for analysis
    top_percent_ebv = 0.05
    top_size_ebv = int(top_percent_ebv * ebv.size)

    top_percent_x = 0.005
    top_size_x = int(top_percent_x * x_sol.size)

    return A, b, x_sol, ebv, y, W, Z, X, P, top_size_x, top_size_ebv


def normalize(array: NDArray) -> NDArray:
    """Divide array by L2 norm."""
    return array / norm(array)


def find_top_indices(x: ArrayLike, top_size: int) -> NDArray:
    """Find indices corresponding to the `top_size` largest entries in x."""
    return np.flip(np.argsort(x))[:top_size]


def get_low_rank_approx(matrix: NDArray, rank: int) -> NDArray:
    """Compute best low-rank approximation of matrix."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S[rank:] = 0
    return U @ np.diag(S) @ Vt


def construct_A_b_no_X(W: NDArray, Z: NDArray, y: NDArray, rank_Z: Optional[int] = None) -> tuple[NDArray, NDArray]:
    """Construct normal equations without fixed effects."""
    var_e = 0.7
    var_g = 0.3 / Z.shape[1] / 0.5
    WZ = W @ Z
    if rank_Z is not None:
        WZ = get_low_rank_approx(WZ, rank_Z)
    A = WZ.T @ WZ * 1 / var_e
    diag_load_idx = np.diag_indices(A.shape[0])
    A[diag_load_idx] += 1 / var_g
    b = WZ.T @ y * 1 / var_e

    return A, b


def construct_A_b(
    W: NDArray, Z: NDArray, X: NDArray, y: NDArray, rank_Z: Optional[int] = None
) -> tuple[NDArray, NDArray]:
    """Construct normal equations with fixed effects."""
    # Construct A and b
    var_e = 0.7
    var_g = 0.3 / Z.shape[1] / 0.5
    WZ = W @ Z
    if rank_Z is not None:
        WZ = get_low_rank_approx(WZ, rank_Z)
    XWZ = np.hstack([X, WZ])
    A = XWZ.T @ XWZ * 1 / var_e
    diag_load_row_idx, diag_load_col_idx = np.diag_indices(A.shape[0])
    diag_load_idx = (diag_load_row_idx[X.shape[1] :], diag_load_col_idx[X.shape[1] :])
    A[diag_load_idx] += 1 / var_g
    b = XWZ.T @ y * 1 / var_e

    return A, b
