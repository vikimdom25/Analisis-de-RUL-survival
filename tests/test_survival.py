"""Tests de la matemática de supervivencia -- ver día 3 del proyecto."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import censored_nll, compute_log_survival


def test_survival_matches_hand_computed_example():
    """S(t) debe coincidir con el ejemplo numérico derivado a mano en el chat."""
    hazards = torch.tensor([[0.02, 0.03, 0.05, 0.10, 0.30]])
    S = torch.exp(compute_log_survival(hazards))
    expected = torch.tensor([0.98, 0.9506, 0.9031, 0.8128, 0.5689])
    assert torch.allclose(S[0], expected, atol=1e-3)


def test_censored_nll_event_case():
    """Verosimilitud del caso 'evento observado' contra el cálculo a mano."""
    hazards = torch.tensor([[0.02, 0.03, 0.05, 0.10, 0.30]])
    nll = censored_nll(hazards, event=torch.tensor([1.0]), lengths=torch.tensor([5]))
    expected = -np.log(0.2438)  # S(4) * h(5), calculado a mano
    assert abs(nll.item() - expected) < 1e-2


def test_censored_nll_censored_case():
    """Verosimilitud del caso 'censurado' contra el cálculo a mano."""
    hazards = torch.tensor([[0.02, 0.03, 0.05, 0.10, 0.30]])
    nll = censored_nll(hazards, event=torch.tensor([0.0]), lengths=torch.tensor([3]))
    expected = -np.log(0.9031)  # S(3), calculado a mano
    assert abs(nll.item() - expected) < 1e-2
