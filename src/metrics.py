"""
Métricas, guardado de historial y gráficas de resultados.

Diseñado para llevar métricas desglosadas de train Y de test por
separado en todo momento (no solo una pérdida agregada), y para
guardar el historial en JSON de forma que sobreviva a una sesión de
Colab que se desconecta a medio entrenamiento.
"""

import json

import numpy as np
import torch


# ---------------------------------------------------------------- métricas base

def masked_mse(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(pred.size(1), device=pred.device)[None, :] < lengths[:, None]
    return ((pred - target) ** 2 * mask).sum() / mask.sum()


def rmse_mae(pred: np.ndarray, true: np.ndarray) -> dict:
    pred, true = np.asarray(pred), np.asarray(true)
    return {
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "mae": float(np.mean(np.abs(pred - true))),
    }


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """
    C-index de Harrell: métrica estándar en supervivencia, mide si el
    modelo ordena correctamente el riesgo relativo entre pares de
    motores comparables (mayor riesgo predicho -> menor tiempo real).
    No estaba en el plan original, pero es la métrica que la
    literatura de supervivencia usa para validar que el ranking de
    riesgo tiene sentido, más allá del error puntual de RUL.
    """
    risk, time, event = np.asarray(risk), np.asarray(time), np.asarray(event)
    n = len(risk)
    concordant, comparable = 0, 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # par comparable: i falló (event=1) antes que j sobrevivió más
            if event[i] == 1 and time[i] < time[j]:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return concordant / comparable if comparable > 0 else float("nan")


def extrapolate_and_median_life(
    hazard_seq: np.ndarray,
    trend_window: int = 8,
    max_horizon: int = 130,
    r2_threshold: float = 0.3,
    mean_hazard_threshold: float = 1e-3,
) -> float:
    """
    Tendencia log-lineal sobre los últimos ciclos + vida mediana.

    max_horizon=130: coincide con el tope que ya usamos para el RUL
    real (RUL_CAP), así que una extrapolación "sana" no queda sesgada
    hacia arriba frente al target.

    Fallback: si el ajuste log-lineal es de mala calidad (R² por
    debajo de r2_threshold) o el hazard promedio es muy pequeño (por
    debajo de mean_hazard_threshold), no hay señal de tendencia
    confiable -- se reporta max_horizon en vez de extrapolar ruido.
    r2_threshold y mean_hazard_threshold quedaron como parámetros
    (antes hardcodeados) para poder buscarlos contra el set de
    validación -- ver tune_extrapolation.py.
    """
    h = np.clip(hazard_seq, 1e-6, 1 - 1e-6)
    window = h[-trend_window:]
    t_window = np.arange(len(window))
    log_h = np.log(window)

    beta, a = np.polyfit(t_window, log_h, deg=1)
    pred = a + beta * t_window
    ss_res = np.sum((log_h - pred) ** 2)
    ss_tot = np.sum((log_h - log_h.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    mean_hazard = window.mean()
    if r2 < r2_threshold or mean_hazard < mean_hazard_threshold:
        return float(max_horizon)

    h_last = h[-1]
    s = 1.0
    for k in range(1, max_horizon + 1):
        h_future = np.clip(h_last * np.exp(beta * k), 1e-6, 1 - 1e-6)
        s *= 1 - h_future
        if s <= 0.5:
            return float(k)
    return float(max_horizon)


# ---------------------------------------------------------------- historial y guardado

class MetricsLogger:
    """Guarda métricas de train Y val por separado, época a época."""

    def __init__(self):
        self.history = {"train": [], "val": []}

    def log_epoch(self, train_metrics: dict, val_metrics: dict):
        self.history["train"].append(train_metrics)
        self.history["val"].append(val_metrics)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetricsLogger":
        logger = cls()
        with open(path) as f:
            logger.history = json.load(f)
        return logger


def evaluate_test(model, test_seqs, test_rul_at_cutoff, device, extrapolation_kwargs: dict | None = None) -> dict:
    """
    Evaluación oficial en test (censurado, ground truth solo en el
    punto de corte -- ver RUL_FD001.txt). Desglosada aparte de train
    a propósito: es el único número que se reporta como resultado
    final, nunca se usa para tomar decisiones de desarrollo.

    extrapolation_kwargs: hiperparámetros de extrapolate_and_median_life
    (trend_window, r2_threshold, mean_hazard_threshold) ya afinados
    contra VALIDACIÓN con tune_extrapolation.py -- nunca contra test.
    """
    extrapolation_kwargs = extrapolation_kwargs or {}
    model.eval()
    rul_preds, hazard_preds = [], []
    with torch.no_grad():
        for seq in test_seqs:
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            hazard, rul_pred = model(x)
            rul_preds.append(rul_pred[0, -1].item())
            hazard_preds.append(hazard[0].cpu().numpy())

    rul_preds = np.clip(np.array(rul_preds), 0, 130)
    metrics = rmse_mae(rul_preds, test_rul_at_cutoff)

    median_lives = np.array([extrapolate_and_median_life(h, **extrapolation_kwargs) for h in hazard_preds])
    metrics["median_life_rmse"] = rmse_mae(median_lives, test_rul_at_cutoff)["rmse"]

    # C-index: usa el ÚLTIMO hazard observado como score de riesgo, no
    # la vida mediana extrapolada. La vida mediana se capa (ver fallback
    # en extrapolate_and_median_life), lo que la vuelve buena para RMSE
    # pero la hace colapsar en empates entre motores sanos -- mal para
    # ranking. El hazard crudo es continuo y no tiene ese problema.
    # Encontrado al ver caer el C-index tras arreglar median_life_rmse.
    last_hazard = np.array([h[-1] for h in hazard_preds])
    metrics["c_index"] = concordance_index(
        risk=last_hazard, time=test_rul_at_cutoff, event=np.ones(len(test_rul_at_cutoff))
    )
    return metrics


# ---------------------------------------------------------------- gráficas

def plot_train_val_curves(history: dict, save_path: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ["loss", "nll", "aux_mse"], ["Pérdida total", "NLL censurada", "MSE auxiliar RUL"]):
        train_vals = [m[key] for m in history["train"]]
        val_vals = [m[key] for m in history["val"]]
        ax.plot(train_vals, label="train")
        ax.plot(val_vals, label="val")
        ax.set_title(title)
        ax.set_xlabel("época")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"Guardado: {save_path}")


def plot_survival_curves(model, seqs: list[np.ndarray], device, save_path: str, n_examples: int = 4):
    import matplotlib.pyplot as plt

    model.eval()
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, seq in enumerate(seqs[:n_examples]):
        x = torch.from_numpy(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            hazard, _ = model(x)
        h = hazard[0].cpu().numpy()
        S = np.exp(np.cumsum(np.log1p(-np.clip(h, 1e-6, 1 - 1e-6))))
        ax.plot(S, label=f"motor {i}")
    ax.set_xlabel("ciclo")
    ax.set_ylabel("S(t) — probabilidad de seguir operando")
    ax.set_title("Funciones de supervivencia predichas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"Guardado: {save_path}")
