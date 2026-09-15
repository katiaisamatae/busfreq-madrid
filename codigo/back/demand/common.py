"""Utilidades compartidas del Módulo 1: carga de datos, folds y métricas.

Centraliza la lógica reutilizada por train.py (baseline) y compare.py
(comparación de modelos), para que todos los modelos se evalúen exactamente con
los mismos folds walk-forward y las mismas métricas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Carga el dataset de features y aplica el filtro temporal del config."""
    df = pd.read_parquet(ROOT.parents[1] / cfg["data"]["processed_path"])
    fmin = cfg["data"].get("fecha_min")
    if fmin:
        df = df[df["fecha"] >= pd.Timestamp(fmin)]
    return df.sort_values(["fecha", "linea"]).reset_index(drop=True)


def feature_cols(cfg: dict, df: pd.DataFrame) -> list[str]:
    """Lista de columnas predictoras presentes en el dataset."""
    f = cfg["features"]
    cols = (f["calendar"] + f["weather"] + f["line_attr"]
            + f["lags"] + f["rolling"] + f["categorical"])
    return [c for c in cols if c in df.columns]


def folds_walk_forward(fechas: pd.Series, n_folds: int, test_days: int):
    """Cortes temporales expansivos (entrena con el pasado, prueba el bloque
    siguiente de `test_days` días). Devuelve [(test_ini, test_fin), ...]."""
    fin = fechas.max()
    cortes = []
    for i in range(n_folds, 0, -1):
        t_ini = fin - pd.Timedelta(days=test_days * i) + pd.Timedelta(days=1)
        t_fin = t_ini + pd.Timedelta(days=test_days) - pd.Timedelta(days=1)
        cortes.append((t_ini, t_fin))
    return cortes


def make_lgbm(m: dict):
    """Crea un LGBMRegressor a partir del bloque `model` del config.

    Pasa todos los parámetros presentes (permitiendo que los valores afinados
    fluyan por train.py/compare.py), salvo los que no acepta el constructor.
    """
    import lightgbm as lgb
    excluir = {"early_stopping_rounds"}
    params = {k: v for k, v in m.items() if k not in excluir}
    params.setdefault("verbose", -1)
    params.setdefault("n_jobs", -1)
    return lgb.LGBMRegressor(**params)


def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = y_true > 0
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100) if mask.any() else np.nan
    total = float(np.sum(y_true))
    wape = float(np.sum(np.abs(err)) / total * 100) if total > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape}


def seleccionar_folds(cortes, indices):
    """Selecciona un subconjunto de folds (1-indexado) de la lista `cortes`.

    Permite separar los folds de ajuste de hiperparámetros del fold final
    intocado con el que se reporta la métrica out-of-sample honesta.
    """
    return [cortes[i - 1] for i in indices if 1 <= i <= len(cortes)]


def clasificar_tipo_dia(fechas):
    """Clasifica cada fecha en 'laborable', 'sabado' o 'festivo'.

    Festivo = domingo o festivo nacional/Comunidad de Madrid. Coincide con la
    segmentación de los factores de demanda medidos (sábado / festivo-domingo).
    Devuelve un ndarray posicional (mismo orden que `fechas`).
    """
    fechas = pd.to_datetime(pd.Series(fechas).reset_index(drop=True))
    try:
        import holidays
        anios = range(int(fechas.dt.year.min()), int(fechas.dt.year.max()) + 1)
        cal = holidays.country_holidays("ES", subdiv="MD", years=anios)
        es_fest = fechas.dt.date.map(lambda d: d in cal).to_numpy()
    except Exception:  # noqa: BLE001
        es_fest = np.zeros(len(fechas), dtype=bool)
    dow = fechas.dt.dayofweek.to_numpy()
    return np.where(es_fest | (dow == 6), "festivo",
                    np.where(dow == 5, "sabado", "laborable"))
