"""
Preprocesamiento completo de C-MAPSS FD001 para el proyecto de RUL
como supervivencia censurada.

Uso como script (genera un artefacto ya procesado, para no repetir
el trabajo de pandas cada vez que se entrena, especialmente útil en
Colab):

    python preprocess.py --data_dir data --out processed_fd001.pt

Uso como módulo: `from preprocess import prepare_fd001, build_censored_dataset`
"""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Sensores casi constantes en FD001, documentados en la literatura.
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENSOR_COLS = [c for c in COLS if c.startswith("s") and c not in DROP_SENSORS]
FEATURE_COLS = SENSOR_COLS  # op1-3 descartados: FD001 es de una sola condición

RUL_CAP = 130  # convención estándar en la literatura de C-MAPSS


# ---------------------------------------------------------------- carga y split

def load_fd001(data_dir: str):
    train = pd.read_csv(f"{data_dir}/train_FD001.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(f"{data_dir}/test_FD001.txt", sep=r"\s+", header=None, names=COLS)
    rul_test = pd.read_csv(f"{data_dir}/RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])
    return train, test, rul_test


def add_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    life = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = (life - train["cycle"]).clip(upper=RUL_CAP)
    return train


@dataclass
class Normalizer:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str]) -> "Normalizer":
        return cls(mean=df[cols].mean(), std=df[cols].std().replace(0, 1.0))

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        df[cols] = (df[cols] - self.mean) / self.std
        return df


def split_units(train: pd.DataFrame, val_frac: float = 0.2, seed: int = 0):
    """Parte por MOTOR completo (nunca por fila), para no filtrar información temporal."""
    units = train["unit"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])
    return train[train["unit"].isin(train_units)], train[train["unit"].isin(val_units)]


def build_sequences(df: pd.DataFrame, feature_cols: list[str]):
    sequences, ruls, unit_ids = [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        sequences.append(g[feature_cols].to_numpy(dtype=np.float32))
        if "RUL" in g.columns:
            ruls.append(g["RUL"].to_numpy(dtype=np.float32))
        unit_ids.append(unit)
    return sequences, ruls, unit_ids


def prepare_fd001(data_dir: str, val_frac: float = 0.2, seed: int = 0):
    train_full, test, rul_test = load_fd001(data_dir)
    train_full = add_train_rul(train_full)

    train_df, val_df = split_units(train_full, val_frac=val_frac, seed=seed)

    norm = Normalizer.fit(train_df, FEATURE_COLS)
    train_n = norm.transform(train_df, FEATURE_COLS)
    val_n = norm.transform(val_df, FEATURE_COLS)
    test_n = norm.transform(test, FEATURE_COLS)

    train_seqs, train_ruls, train_units = build_sequences(train_n, FEATURE_COLS)
    val_seqs, val_ruls, val_units = build_sequences(val_n, FEATURE_COLS)
    test_seqs, _, test_units = build_sequences(test_n, FEATURE_COLS)

    test_rul_at_cutoff = np.clip(rul_test["RUL"].to_numpy(dtype=np.float32), 0, RUL_CAP)

    return {
        "feature_cols": FEATURE_COLS,
        "normalizer": norm,
        "train_seqs": train_seqs, "train_ruls": train_ruls, "train_units": train_units,
        "val_seqs": val_seqs, "val_ruls": val_ruls, "val_units": val_units,
        "test_seqs": test_seqs, "test_units": test_units,
        "test_rul_at_cutoff": test_rul_at_cutoff,
    }


# ---------------------------------------------------------------- censura simulada

@dataclass
class Example:
    features: np.ndarray  # (L, F)
    rul: np.ndarray       # (L,) ground truth, siempre conocido en train
    event: float          # 1.0 = vida completa, 0.0 = censurado
    length: int


def build_censored_dataset(seqs, ruls, n_censored_per_engine: int = 2, min_frac: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    examples: list[Example] = []
    for seq, rul in zip(seqs, ruls):
        T = len(seq)
        examples.append(Example(features=seq, rul=rul, event=1.0, length=T))

        min_len = max(2, int(T * min_frac))
        max_len = max(min_len + 1, T - 1)
        if max_len <= min_len:
            continue
        for L in rng.integers(min_len, max_len, size=n_censored_per_engine):
            L = int(L)
            examples.append(Example(features=seq[:L], rul=rul[:L], event=0.0, length=L))
    return examples


# ---------------------------------------------------------------- artefacto serializado

def run_and_save(data_dir: str, out_path: str, val_frac: float = 0.2, seed: int = 0):
    d = prepare_fd001(data_dir, val_frac=val_frac, seed=seed)
    train_examples = build_censored_dataset(d["train_seqs"], d["train_ruls"], seed=seed)
    val_examples = build_censored_dataset(d["val_seqs"], d["val_ruls"], seed=seed + 1)

    payload = {
        "feature_cols": d["feature_cols"],
        "normalizer_mean": d["normalizer"].mean,
        "normalizer_std": d["normalizer"].std,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "test_seqs": d["test_seqs"],
        "test_units": d["test_units"],
        "test_rul_at_cutoff": d["test_rul_at_cutoff"],
    }
    torch.save(payload, out_path)
    print(f"Guardado: {out_path}")
    print(f"  train (con censura simulada): {len(train_examples)} ejemplos")
    print(f"  val   (con censura simulada): {len(val_examples)} ejemplos")
    print(f"  test  (oficial, sin aumentar): {len(d['test_seqs'])} motores")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", default="processed_fd001.pt")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_and_save(args.data_dir, args.out, args.val_frac, args.seed)
