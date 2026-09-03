"""
Baseline día 1: LSTM -> regresión directa de RUL.

Este modelo NO usa censura ni supervivencia -- es exactamente el
enfoque estándar que casi todo el mundo hace con C-MAPSS. Sirve
como punto de comparación honesto para el modelo de supervivencia
que se construye a partir del día 2.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from metrics import concordance_index, masked_mse, rmse_mae
from preprocess import RUL_CAP, prepare_fd001


def collate_padded(seqs: list[np.ndarray], targets: list[np.ndarray] | None = None):
    """Empaqueta secuencias de longitud variable con padding + máscara."""
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    x = pad_sequence([torch.from_numpy(s) for s in seqs], batch_first=True)  # (B, T, F)
    y = None
    if targets is not None:
        y = pad_sequence([torch.from_numpy(t) for t in targets], batch_first=True)  # (B, T)
    return x, lengths, y


class LSTMBaseline(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.head(out).squeeze(-1)  # (B, T) RUL predicho en cada ciclo


def train_baseline(
    epochs: int = 100, batch_size: int = 16, lr: float = 1e-3, seed: int = 0,
    data_dir: str = "../data", patience: int = 10, min_delta: float = 1e-3,
):
    """
    Nota (día 6): en el día 1 se dejó a propósito sin pulir -- su único
    propósito era dar un punto de referencia rápido. Para que la
    comparación final contra el modelo de supervivencia sea justa, aquí
    sí se entrena en serio, con early stopping igual que el modelo
    principal (train.py), sobre el mismo split train/val por motor.
    """
    torch.manual_seed(seed)
    d = prepare_fd001(data_dir, seed=seed)
    seqs, ruls = d["train_seqs"], d["train_ruls"]
    val_seqs, val_ruls = d["val_seqs"], d["val_ruls"]
    n_features = len(d["feature_cols"])

    model = LSTMBaseline(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    idx = np.arange(len(seqs))
    history = []
    best_val_mse, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(idx)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            x, lengths, y = collate_padded([seqs[i] for i in batch_idx], [ruls[i] for i in batch_idx])
            pred = model(x, lengths)
            loss = masked_mse(pred, y, lengths)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            x, lengths, y = collate_padded(val_seqs, val_ruls)
            val_loss = masked_mse(model(x, lengths), y, lengths).item()

        history.append({"train_mse": epoch_loss / n_batches, "val_mse": val_loss})
        print(f"epoch {epoch+1:03d}  train_mse={history[-1]['train_mse']:.2f}  val_mse={val_loss:.2f}")

        if val_loss < best_val_mse - min_delta:
            best_val_mse, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping en época {epoch+1}: sin mejora en val MSE durante {patience} épocas.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restaurado el mejor estado (val MSE = {best_val_mse:.2f}).")

    return model, d, history


def evaluate_on_test(model, d):
    """RUL predicho = valor del último ciclo observado de cada motor de test."""
    model.eval()
    seqs = d["test_seqs"]
    true_rul = d["test_rul_at_cutoff"]
    preds = []
    with torch.no_grad():
        for s in seqs:
            x, lengths, _ = collate_padded([s])
            pred = model(x, lengths)
            preds.append(pred[0, lengths[0] - 1].item())
    preds = np.clip(np.array(preds), 0, RUL_CAP)
    metrics = rmse_mae(preds, true_rul)
    # C-index para comparación justa contra el modelo de supervivencia:
    # mayor RUL predicho = menor riesgo, por eso risk = -preds.
    metrics["c_index"] = concordance_index(risk=-preds, time=true_rul, event=np.ones(len(true_rul)))
    metrics["preds"], metrics["true"] = preds, true_rul
    return metrics


if __name__ == "__main__":
    model, d, history = train_baseline()
    metrics = evaluate_on_test(model, d)
    print(f"\nBaseline LSTM en test FD001 -> RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}")
