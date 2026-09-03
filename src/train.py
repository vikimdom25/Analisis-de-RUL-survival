"""
Entrenamiento del modelo de supervivencia. Detecta CPU/GPU
automáticamente -- pensado para correr igual en este entorno que en
Colab (donde si activas GPU, batch_size puede subir bastante respecto
al límite de 8 que encontramos en CPU en el día 4).

Uso:
    python train.py --data_dir data --epochs 40 --batch_size 8
    # en Colab con GPU: --batch_size 32 (o más) suele andar bien
"""

import argparse
import os

import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from metrics import MetricsLogger, evaluate_test, masked_mse, plot_survival_curves, plot_train_val_curves
from model import CausalSurvivalModel, censored_nll
from preprocess import Example, build_censored_dataset, prepare_fd001

W_SURVIVAL = 1.0
W_RUL_AUX = 0.15


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_examples(batch: list[Example]):
    x = pad_sequence([torch.from_numpy(e.features) for e in batch], batch_first=True)
    rul = pad_sequence([torch.from_numpy(e.rul) for e in batch], batch_first=True)
    lengths = torch.tensor([e.length for e in batch], dtype=torch.long)
    event = torch.tensor([e.event for e in batch], dtype=torch.float32)
    T = x.size(1)
    key_padding_mask = torch.arange(T)[None, :] >= lengths[:, None]
    return x, rul, lengths, event, key_padding_mask


def run_epoch(model, examples, batch_size, device, optimizer=None, seed=0):
    is_train = optimizer is not None
    model.train(is_train)

    idx = np.arange(len(examples))
    if is_train:
        np.random.default_rng(seed).shuffle(idx)

    totals = {"loss": 0.0, "nll": 0.0, "aux_mse": 0.0, "n_batches": 0}
    for start in range(0, len(idx), batch_size):
        batch = [examples[i] for i in idx[start:start + batch_size]]
        x, rul, lengths, event, kpm = collate_examples(batch)
        x, rul, lengths, event, kpm = x.to(device), rul.to(device), lengths.to(device), event.to(device), kpm.to(device)

        with torch.set_grad_enabled(is_train):
            hazard, rul_pred = model(x, key_padding_mask=kpm)
            nll = censored_nll(hazard, event, lengths)
            aux_mse = masked_mse(rul_pred, rul, lengths)
            loss = W_SURVIVAL * nll + W_RUL_AUX * aux_mse

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        totals["loss"] += loss.item()
        totals["nll"] += nll.item()
        totals["aux_mse"] += aux_mse.item()
        totals["n_batches"] += 1

    n = totals.pop("n_batches")
    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=None, help="por defecto: 8 en CPU, 32 en GPU")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--patience", type=int, default=8, help="épocas sin mejora en val NLL antes de parar")
    parser.add_argument("--min_delta", type=float, default=1e-3, help="mejora mínima para contar como progreso")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = get_device()
    batch_size = args.batch_size or (32 if device.type == "cuda" else 8)
    print(f"Dispositivo: {device}  |  batch_size: {batch_size}")

    d = prepare_fd001(args.data_dir, seed=args.seed)
    n_features = len(d["feature_cols"])
    train_ex = build_censored_dataset(d["train_seqs"], d["train_ruls"], seed=args.seed)
    val_ex = build_censored_dataset(d["val_seqs"], d["val_ruls"], seed=args.seed + 1)
    print(f"train: {len(train_ex)} ejemplos (con censura simulada)  |  val: {len(val_ex)} ejemplos")

    model = CausalSurvivalModel(n_sensors=n_features, d_model=args.d_model, n_layers=args.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = MetricsLogger()

    best_val_nll = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_m = run_epoch(model, train_ex, batch_size, device, optimizer=opt, seed=epoch)
        val_m = run_epoch(model, val_ex, batch_size, device, optimizer=None)
        logger.log_epoch(train_m, val_m)
        print(
            f"epoch {epoch+1:03d}  "
            f"train: loss={train_m['loss']:.3f} nll={train_m['nll']:.3f} aux_mse={train_m['aux_mse']:.1f}  |  "
            f"val: loss={val_m['loss']:.3f} nll={val_m['nll']:.3f} aux_mse={val_m['aux_mse']:.1f}"
        )

        # Early stopping sobre NLL de validación -- es el objetivo real de
        # supervivencia; aux_mse es solo la cabeza auxiliar (peso 0.15).
        if val_m["nll"] < best_val_nll - args.min_delta:
            best_val_nll = val_m["nll"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping en época {epoch+1}: sin mejora en val NLL durante {args.patience} épocas.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restaurado el mejor estado (val NLL = {best_val_nll:.3f}).")

    logger.save(f"{args.out_dir}/history.json")
    plot_train_val_curves(logger.history, f"{args.out_dir}/train_val_curves.png")
    plot_survival_curves(model, d["val_seqs"], device, f"{args.out_dir}/survival_curves.png")

    test_metrics = evaluate_test(model, d["test_seqs"], d["test_rul_at_cutoff"], device)
    print("\n--- Evaluación en TEST (oficial, censurado) ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.3f}")

    torch.save(model.state_dict(), f"{args.out_dir}/model.pt")
    print(f"\nModelo guardado en {args.out_dir}/model.pt")


if __name__ == "__main__":
    main()
