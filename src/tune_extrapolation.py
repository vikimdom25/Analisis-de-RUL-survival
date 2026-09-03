"""
Afina trend_window, r2_threshold y mean_hazard_threshold de
extrapolate_and_median_life contra VALIDACIÓN -- nunca contra test,
por la misma razón que search_hparams.py nunca toca test: es la única
evaluación que se reporta como resultado final.

Uso (con el modelo ya entrenado, model.pt):
    python tune_extrapolation.py --data_dir . --d_model 32 --n_layers 1
"""

import argparse

import numpy as np
import torch

from metrics import extrapolate_and_median_life, rmse_mae
from model import CausalSurvivalModel
from preprocess import build_censored_dataset, prepare_fd001


def collect_val_predictions(model, val_examples, device):
    """Solo los ejemplos CENSURADOS (event=0): son los que imitan el escenario real de test."""
    model.eval()
    hazard_seqs, true_ruls = [], []
    with torch.no_grad():
        for ex in val_examples:
            if ex.event == 1.0:
                continue  # las copias completas no simulan el escenario de corte
            x = torch.from_numpy(ex.features).unsqueeze(0).to(device)
            hazard, _ = model(x)
            hazard_seqs.append(hazard[0].cpu().numpy())
            true_ruls.append(ex.rul[-1])  # RUL verdadero en el punto de corte
    return hazard_seqs, np.array(true_ruls, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = prepare_fd001(args.data_dir, seed=args.seed)
    val_examples = build_censored_dataset(d["val_seqs"], d["val_ruls"], seed=args.seed + 1)

    model = CausalSurvivalModel(n_sensors=len(d["feature_cols"]), d_model=args.d_model, n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    hazard_seqs, true_ruls = collect_val_predictions(model, val_examples, device)
    print(f"Motores censurados de validación: {len(hazard_seqs)}")

    results = []
    for trend_window in [5, 8, 12, 15]:
        for r2_threshold in [0.1, 0.3, 0.5]:
            for mean_hazard_threshold in [1e-4, 1e-3, 5e-3]:
                preds = np.array([
                    extrapolate_and_median_life(
                        h, trend_window=trend_window, r2_threshold=r2_threshold, mean_hazard_threshold=mean_hazard_threshold
                    )
                    for h in hazard_seqs
                ])
                rmse = rmse_mae(preds, true_ruls)["rmse"]
                results.append({
                    "trend_window": trend_window, "r2_threshold": r2_threshold,
                    "mean_hazard_threshold": mean_hazard_threshold, "val_rmse": rmse,
                })

    results.sort(key=lambda r: r["val_rmse"])
    print("\nTop 5 combinaciones (por RMSE de vida mediana en validación):")
    for r in results[:5]:
        print(f"  trend_window={r['trend_window']:2d}  r2_threshold={r['r2_threshold']:.1f}  "
              f"mean_hazard_threshold={r['mean_hazard_threshold']:.0e}  val_rmse={r['val_rmse']:.2f}")

    best = results[0]
    print(f"\nMejor combinación: {best}")
    print("\nPara evaluar en test con esta combinación:")
    print(f"  extrapolation_kwargs = {{'trend_window': {best['trend_window']}, "
          f"'r2_threshold': {best['r2_threshold']}, 'mean_hazard_threshold': {best['mean_hazard_threshold']}}}")

    import json
    with open("extrapolation_search.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
