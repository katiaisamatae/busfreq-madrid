"""Modelo de demanda para FECHAS FUTURAS (sin lags) — Módulo 1.

Para una fecha futura no hay retardos recientes de demanda (los datos abiertos
llegan con 1-2 meses de retraso). Este módulo entrena un LightGBM SIN lags, solo
con variables de calendario y la línea, utilizable en cualquier fecha, y lo usa
como demanda base del plan futuro. Sustituye a la antigua "media de los últimos 8
días tipo": el LightGBM capta estacionalidad mensual/anual y festivos, opera en
la ruta real de predicción del producto, y su error se evalúa walk-forward igual
que el modelo principal (ver mape_walkforward()).

La meteorología NO entra en el modelo base: se aplica después como factor de
contexto (lluvia) en la capa de API, para no duplicar su efecto.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demand.common import folds_walk_forward, make_lgbm, metricas
from ingest.config import DATA_PROCESSED

# Calendario + línea (SIN lags, rolling ni meteo).
FEATURES = ["mes", "dia_semana", "semana", "dia_anio", "finde", "is_festivo"]
CAT = ["linea"]
_PARAMS = {"objective": "regression_l1", "metric": "mae", "n_estimators": 500,
           "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 30,
           "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
           "random_state": 42}


def _dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / "dataset_features.parquet")
    df = df[df["fecha"] >= "2022-01-01"].copy()
    df["linea"] = df["linea"].astype("category")
    return df


@lru_cache(maxsize=1)
def _modelo():
    df = _dataset()
    m = make_lgbm(_PARAMS)
    m.fit(df[FEATURES + CAT], df["viajeros"], categorical_feature=CAT)
    return m


def _calendario(fecha: pd.Timestamp, lineas: list[str]) -> pd.DataFrame:
    import holidays
    cal = holidays.country_holidays("ES", subdiv="MD", years=range(2019, 2031))
    base = {"mes": fecha.month, "dia_semana": fecha.dayofweek,
            "semana": int(fecha.isocalendar().week), "dia_anio": fecha.dayofyear,
            "finde": int(fecha.dayofweek >= 5), "is_festivo": int(fecha.date() in cal)}
    df = pd.DataFrame([{**base, "linea": l} for l in lineas])
    df["linea"] = df["linea"].astype("category")
    return df


def predecir_dia(fecha: pd.Timestamp, lineas: list[str]) -> pd.DataFrame:
    """Demanda diaria prevista por línea para una fecha (LightGBM sin lags)."""
    X = _calendario(fecha, lineas)
    pred = np.clip(_modelo().predict(X[FEATURES + CAT]), 0, None)
    return pd.DataFrame({"linea": lineas, "viajeros": pred})


@lru_cache(maxsize=1)
def mape_walkforward() -> float:
    """MAPE walk-forward del modelo sin lags (mismo protocolo que el principal)."""
    df = _dataset()
    errores = []
    for t_ini, t_fin in folds_walk_forward(df["fecha"], 4, 90):
        tr = df[df["fecha"] < t_ini]
        te = df[(df["fecha"] >= t_ini) & (df["fecha"] <= t_fin)]
        if te.empty:
            continue
        m = make_lgbm(_PARAMS)
        m.fit(tr[FEATURES + CAT], tr["viajeros"], categorical_feature=CAT)
        errores.append(metricas(te["viajeros"].to_numpy(), m.predict(te[FEATURES + CAT]))["MAPE"])
    return round(float(np.mean(errores)), 1)


def demanda_futura_por_franja(fecha: pd.Timestamp,
                              piloto: list[str]) -> tuple[pd.DataFrame, str, str]:
    """Demanda por línea y franja para una fecha futura.

    Base = LightGBM sin lags (calendario+línea), repartida por franja con el perfil
    intradía. Los factores de contexto (lluvia, eventos) se aplican después en la
    API. Devuelve (tabla[linea, franja, pax], tipo_dia, descripción del método).
    """
    from optimizer.optimize import tipo_dia_calendario
    td = tipo_dia_calendario(fecha)
    base = predecir_dia(fecha, piloto)
    perfil = pd.read_parquet(DATA_PROCESSED / "perfil_intradia.parquet")
    d = base.merge(perfil[perfil["tipo_dia"] == td], on="linea", how="left")
    d["pax"] = d["viajeros"] * d["share"]
    d = d[d["pax"].notna()]
    metodo = (f"LightGBM sin lags (calendario+línea), MAPE walk-forward "
              f"{mape_walkforward()} % + factores de contexto detectados")
    return d[["linea", "franja", "pax"]], td, metodo
