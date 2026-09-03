"""
Búsqueda de hiperparámetros en dos pasos, pensada para correr en
Colab con GPU (en CPU cada corrida es lenta -- ver nota del chat).

Paso 1: barrido de d_model (ancho), n_layers fijo en 2.
Paso 2: barrido de n_layers (profundidad), con el mejor d_model del paso 1.
Paso 3: barrido de lr, solo sobre la mejor combinación de arquitectura.

Selecciona siempre por NLL de validación (el objetivo real de
supervivencia), no por aux_mse ni por la pérdida combinada.

Uso:
    python search_hparams.py --data_dir data --epochs 40 --patience 8
"""

import argparse

import torch

from metrics import MetricsLogger
from model import CausalSurvivalModel
from preprocess import build_censored_dataset, prepare_fd001
from train import get_device, run_epoch


def train_once(n_features, d_model, n_layers, lr, train_ex, val_ex, device, batch_size, epochs, patience, min_delta, seed=0):
    import copy

    torch.manual_seed(seed)
    model = CausalSurvivalModel(n_sensors=n_features, d_model=d_model, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_nll, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        run_epoch(model, train_ex, batch_size, device, optimizer=opt, seed=epoch)
        val_m = run_epoch(model, val_ex, batch_size, device, optimizer=None)
        if val_m["nll"] < best_val_nll - min_delta:
            best_val_nll, best_state, no_improve = val_m["nll"], copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_val_nll, epoch + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    batch_size = 32 if device.type == "cuda" else 8
    print(f"Dispositivo: {device}  |  batch_size: {batch_size}")

    d = prepare_fd001(args.data_dir, seed=args.seed)
    n_features = len(d["feature_cols"])
    train_ex = build_censored_dataset(d["train_seqs"], d["train_ruls"], seed=args.seed)
    val_ex = build_censored_dataset(d["val_seqs"], d["val_ruls"], seed=args.seed + 1)

    results = []

    print("\n=== Paso 1: d_model (n_layers=2) ===")
    best_d_model, best_nll = None, float("inf")
    for d_model in [16, 32, 64]:
        nll, n_epochs = train_once(n_features, d_model, 2, args.lr, train_ex, val_ex, device, batch_size, args.epochs, args.patience, args.min_delta, args.seed)
        print(f"  d_model={d_model:3d}  val_nll={nll:.3f}  (paró en época {n_epochs})")
        results.append({"stage": "d_model", "d_model": d_model, "n_layers": 2, "lr": args.lr, "val_nll": nll})
        if nll < best_nll:
            best_nll, best_d_model = nll, d_model

    print(f"\nMejor d_model: {best_d_model} (val_nll={best_nll:.3f})")

    print(f"\n=== Paso 2: n_layers (d_model={best_d_model}) ===")
    best_n_layers, best_nll_2 = 2, best_nll
    for n_layers in [1, 2, 3]:
        if n_layers == 2:
            continue  # ya lo tenemos del paso 1
        nll, n_epochs = train_once(n_features, best_d_model, n_layers, args.lr, train_ex, val_ex, device, batch_size, args.epochs, args.patience, args.min_delta, args.seed)
        print(f"  n_layers={n_layers}  val_nll={nll:.3f}  (paró en época {n_epochs})")
        results.append({"stage": "n_layers", "d_model": best_d_model, "n_layers": n_layers, "lr": args.lr, "val_nll": nll})
        if nll < best_nll_2:
            best_nll_2, best_n_layers = nll, n_layers

    print(f"\nMejor n_layers: {best_n_layers} (val_nll={best_nll_2:.3f})")

    print(f"\n=== Paso 3: learning rate (d_model={best_d_model}, n_layers={best_n_layers}) ===")
    best_lr, best_nll_3 = args.lr, best_nll_2
    for lr in [1e-3, 3e-4]:
        if lr == args.lr:
            continue  # ya lo tenemos de los pasos anteriores
        nll, n_epochs = train_once(n_features, best_d_model, best_n_layers, lr, train_ex, val_ex, device, batch_size, args.epochs, args.patience, args.min_delta, args.seed)
        print(f"  lr={lr:.0e}  val_nll={nll:.3f}  (paró en época {n_epochs})")
        results.append({"stage": "lr", "d_model": best_d_model, "n_layers": best_n_layers, "lr": lr, "val_nll": nll})
        if nll < best_nll_3:
            best_nll_3, best_lr = nll, lr

    print("\n=== Resultado final ===")
    print(f"d_model={best_d_model}  n_layers={best_n_layers}  lr={best_lr:.0e}  val_nll={best_nll_3:.3f}")
    print("\nPara entrenar con esta configuración:")
    print(f"  python train.py --data_dir {args.data_dir} --d_model {best_d_model} --n_layers {best_n_layers} --lr {best_lr}")

    import json
    with open("search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nTabla completa guardada en search_results.json")


if __name__ == "__main__":
    main()
