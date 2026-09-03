"""Tests de la simulación de censura -- ver día 4 del proyecto."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import build_censored_dataset


def test_censoring_produces_expected_counts():
    """Por motor: 1 copia completa + n_censored_per_engine copias censuradas."""
    rng = np.random.default_rng(0)
    seqs = [rng.normal(size=(T, 3)).astype(np.float32) for T in [50, 120, 200]]
    ruls = [np.arange(T, 0, -1).astype(np.float32) for T in [50, 120, 200]]

    dataset = build_censored_dataset(seqs, ruls, n_censored_per_engine=2, seed=0)

    n_full = sum(1 for e in dataset if e.event == 1.0)
    n_censored = sum(1 for e in dataset if e.event == 0.0)
    assert n_full == len(seqs)
    assert n_censored == len(seqs) * 2


def test_censored_examples_are_strictly_before_failure():
    """Las copias censuradas deben cortarse antes del final real, nunca en él."""
    rng = np.random.default_rng(0)
    seqs = [rng.normal(size=(100, 3)).astype(np.float32)]
    ruls = [np.arange(100, 0, -1).astype(np.float32)]

    dataset = build_censored_dataset(seqs, ruls, n_censored_per_engine=3, seed=0)
    censored = [e for e in dataset if e.event == 0.0]
    assert all(e.length < 100 for e in censored)
