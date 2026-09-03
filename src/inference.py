"""
Ejemplo mínimo de uso: carga un modelo ya entrenado (model.pt) y corre
una predicción sobre un motor de test, sin reentrenar ni reevaluar todo
el pipeline. Pensado para que alguien que revise el repo entienda cómo
se usa el modelo en 30 segundos, no en 80 épocas de entrenamiento.

Uso:
    python inference.py --data_dir data --model_path model.pt --engine_idx 0
"""

import argparse

import numpy as np
import torch

from model import CausalSurvivalModel
from preprocess import prepare_fd001
from metrics import extrapolate_and_median_life


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--engine_idx", type=int, default=0, help="índice del motor de test a inspeccionar (0-99)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar datos y el motor de test elegido
    d = prepare_fd001(args.data_dir)
    seq = d["test_seqs"][args.engine_idx]              # (T, 14) -- sensores ya normalizados
    true_rul = d["test_rul_at_cutoff"][args.engine_idx]  # RUL real en el punto de corte (ground truth)

    # 2. Cargar el modelo ya entrenado
    model = CausalSurvivalModel(n_sensors=len(d["feature_cols"]), d_model=args.d_model, n_layers=args.n_layers)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    # 3. Una sola pasada hacia adelante -- esto es TODA la inferencia
    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        hazard, rul_pred = model(x)

    hazard = hazard[0].cpu().numpy()
    point_rul = float(np.clip(rul_pred[0, -1].item(), 0, 130))

    # 4. Derivar la función de supervivencia y la vida mediana (ver metrics.py)
    survival = np.exp(np.cumsum(np.log1p(-np.clip(hazard, 1e-6, 1 - 1e-6))))
    median_life = extrapolate_and_median_life(
        hazard, trend_window=12, r2_threshold=0.3, mean_hazard_threshold=1e-4
    )

    print(f"Motor de test #{args.engine_idx}  (secuencia de {len(seq)} ciclos observados)")
    print(f"  RUL real (ground truth):              {true_rul:.0f} ciclos")
    print(f"  RUL predicho (cabeza auxiliar):        {point_rul:.1f} ciclos")
    print(f"  Vida mediana (extrapolación supervivencia): {median_life:.0f} ciclos")
    print(f"  Hazard en el último ciclo observado:   {hazard[-1]:.4f}")
    print(f"  S(t) en el último ciclo observado:     {survival[-1]:.4f}  (probabilidad de seguir operando)")


if __name__ == "__main__":
    main()
