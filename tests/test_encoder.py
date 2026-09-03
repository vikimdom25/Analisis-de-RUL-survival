"""Tests del encoder causal de dos etapas -- ver día 2 del proyecto."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import CausalTwoStageEncoder


def test_causality_future_does_not_affect_past():
    """Modificar un ciclo futuro no debe cambiar la salida en ciclos anteriores."""
    torch.manual_seed(0)
    B, T, D, dm = 2, 12, 14, 16
    encoder = CausalTwoStageEncoder(n_sensors=D, d_model=dm, n_layers=2, n_heads=4)
    encoder.eval()

    x = torch.randn(B, T, D)
    with torch.no_grad():
        out_before = encoder(x)

    x_modified = x.clone()
    x_modified[:, -1, :] = torch.randn(B, D) * 100
    with torch.no_grad():
        out_after = encoder(x_modified)

    early_diff = (out_before[:, :-1, :] - out_after[:, :-1, :]).abs().max().item()
    last_diff = (out_before[:, -1, :] - out_after[:, -1, :]).abs().max().item()

    assert early_diff < 1e-5, "FALLO DE CAUSALIDAD: el pasado cambió al modificar el futuro"
    assert last_diff > 1e-3, "el ciclo modificado no cambió su propia salida (sospechoso)"


def test_padding_does_not_leak_between_batch_items():
    """Un motor corto debe dar la misma salida solo o dentro de un batch con padding."""
    torch.manual_seed(0)
    lengths = [8, 15, 6, 10]
    D, dm = 5, 16
    seqs = [np.random.randn(L, D).astype(np.float32) for L in lengths]

    x = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(s) for s in seqs], batch_first=True)
    T = x.size(1)
    key_padding_mask = torch.arange(T)[None, :] >= torch.tensor(lengths)[:, None]

    encoder = CausalTwoStageEncoder(n_sensors=D, d_model=dm, n_layers=1, n_heads=4)
    encoder.eval()
    with torch.no_grad():
        out = encoder(x, key_padding_mask=key_padding_mask)

    assert not torch.isnan(out).any()

    short_idx = int(torch.tensor(lengths).argmin())
    x_alone = torch.from_numpy(seqs[short_idx]).unsqueeze(0)
    with torch.no_grad():
        out_alone = encoder(x_alone)

    diff = (out[short_idx, : lengths[short_idx], :] - out_alone[0]).abs().max().item()
    assert diff < 1e-5, "el padding está contaminando la salida de otro motor en el batch"
