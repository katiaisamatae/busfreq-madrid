"""Lags y medias móviles: sin fuga (>=7 días) y sin cruzar líneas (Fase 1)."""
import numpy as np
import pandas as pd

from demand.features import añadir_lags_rolling


def _synthetic():
    """2 líneas, 60 días; viajeros = índice del día (para comprobar desfases)."""
    fechas = pd.date_range("2024-01-01", periods=60, freq="D")
    filas = [{"linea": linea, "fecha": f, "viajeros": base + i}
             for linea, base in (("A", 0), ("B", 1000))
             for i, f in enumerate(fechas)]
    return pd.DataFrame(filas)


def test_lag7_es_valor_de_hace_7_dias():
    df = añadir_lags_rolling(_synthetic(), lags=[7, 14], rolling=[7])
    a = df[df["linea"] == "A"].sort_values("fecha").reset_index(drop=True)
    assert a.loc[10, "lag_7"] == a.loc[10, "viajeros"] - 7
    assert pd.isna(a.loc[0, "lag_7"])            # sin histórico -> NaN


def test_lags_no_cruzan_lineas():
    df = añadir_lags_rolling(_synthetic(), lags=[7], rolling=[7])
    b = df[df["linea"] == "B"].sort_values("fecha").reset_index(drop=True)
    assert b.loc[:6, "lag_7"].isna().all()       # B no toma valores del final de A
    assert b.loc[10, "lag_7"] == b.loc[10, "viajeros"] - 7


def test_rolling_desplazado_no_usa_dias_recientes():
    df = añadir_lags_rolling(_synthetic(), lags=[7], rolling=[7])
    a = df[df["linea"] == "A"].sort_values("fecha").reset_index(drop=True)
    i = 20
    # shift(7).rolling(7).mean() en i = media de viajeros[i-13 .. i-7]
    esperado = np.mean([a.loc[j, "viajeros"] for j in range(i - 13, i - 6)])
    assert abs(a.loc[i, "roll_mean_7"] - esperado) < 1e-6
