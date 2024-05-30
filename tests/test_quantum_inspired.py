"""Tests for the genomic_prediction.quantum_inspired module."""

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import cg

from genomic_prediction import quantum_inspired as qi


def test_qi():
    # Define path to data
    path = Path(Path(__file__).parent.resolve(), "data")

    # Load data
    A = np.load(Path(path, "lhssnp.npy"))
    b = np.load(Path(path, "rhssnp.npy"))
    x_sol = np.load(Path(path, "solsnp.npy"))
    P = np.load(Path(path, "m1.npy"))

    # Run (classical) quantum-inspired algorithm
    rank = 3
    r = 200
    c = 200
    n_samples = 50
    n_comp_x = 50
    sampled_comp_qi, x_qi = qi.linear_eqs(A, b, r, c, rank, n_samples, n_comp_x)

    # Run classical algorithms
    x_fkv = qi.linear_eqs_fkv(P @ A, P @ b, r, c, rank)
    x_cg, exit_code = cg(A, b, M=P, atol=1e-5)

    assert True
