"""Construcción del dataset unificado de features — Módulo 1 (día × línea).

Une todas las fuentes de la Fase 1 en un único dataset diario por línea, listo
para el modelado de demanda:

  demanda (objetivo)  +  calendario  +  festivos  +  meteorología (AEMET)
  +  saturación media de línea  +  indicador de régimen COVID  +  lags/medias.

Granularidad: DÍA × LÍNEA (la demanda pública no baja de ahí; ver notas de la
Fase 1). Se restringe a las 12 líneas piloto.

Salida: datos/processed/dataset_features.parquet

Uso:
    python -m demand.features
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import DATA_PROCESSED

ROOT = Path(__file__).resolve().parent.parent
DEMANDA = DATA_PROCESSED / "demanda_diaria_linea.parquet"
AEMET = DATA_PROCESSED / "aemet_historico.parquet"
OCUP = DATA_PROCESSED / "ocupacion_linea_mes.parquet"
PILOTO = DATA_PROCESSED / "lineas_piloto.csv"
SALIDA = DATA_PROCESSED / "dataset_features.parquet"

# Ventana COVID (estado de alarma + recuperación) para marcar régimen atípico.
COVID_INI = pd.Timestamp("2020-03-14")
COVID_FIN = pd.Timestamp("2021-06-30")

LAGS = [7, 14, 28]          # días (>=7 -> seguros para horizonte 24-48 h)
ROLLING = [7, 28]           # medias móviles (desplazadas 7 días, sin fuga)


def _festivos(fechas: pd.Series) -> pd.Series:
    """Serie booleana de festivos (nacional + Comunidad de Madrid)."""
    try:
        import holidays
    except ImportError:
        print("[features] 'holidays' no instalado; is_festivo=0 (instálalo con pip)")
        return pd.Series(False, index=fechas.index)
    anios = range(int(fechas.dt.year.min()), int(fechas.dt.year.max()) + 1)
    cal = holidays.Spain(subdiv="MD", years=anios)
    return fechas.dt.date.map(lambda d: d in cal).astype(bool)


def _clima_diario() -> pd.DataFrame:
    """Serie meteo diaria de ciudad: Barajas (3129) principal, hueco->Retiro."""
    aem = pd.read_parquet(AEMET)
    cols = ["tmed", "tmin", "tmax", "prec", "velmedia", "racha"]
    barajas = aem[aem["estacion"] == "3129"].set_index("fecha")[cols]
    retiro = aem[aem["estacion"] == "3195"].set_index("fecha")[cols]
    clima = barajas.combine_first(retiro).sort_index()
    # Relleno SOLO hacia delante (nunca el día siguiente): interpolate usaba el
    # futuro. Nota: la evaluación emplea meteo OBSERVADA, no previsión a 24-48 h,
    # por lo que el rendimiento operativo real será algo peor (limitación).
    clima = clima.ffill(limit=3).reset_index()
    return clima


def añadir_lags_rolling(df: pd.DataFrame, lags=LAGS, rolling=ROLLING,
                        shift_seguro: int = 7) -> pd.DataFrame:
    """Añade lags y medias/desv móviles de 'viajeros' por línea SIN fuga temporal.

    Los lags usan desfases >= 7 días; las ventanas móviles se calculan con
    ``transform`` DENTRO de cada línea (no cruzan líneas) y desplazadas
    ``shift_seguro`` días, para no filtrar el futuro al horizonte de 24-48 h.
    """
    df = df.sort_values(["linea", "fecha"]).reset_index(drop=True)
    g = df.groupby("linea")["viajeros"]
    for k in lags:
        df[f"lag_{k}"] = g.shift(k)
    for w in rolling:
        df[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(shift_seguro).rolling(w).mean())
        df[f"roll_std_{w}"] = g.transform(lambda s: s.shift(shift_seguro).rolling(w).std())
    return df


def main() -> None:
    assert DEMANDA.exists(), f"Falta {DEMANDA} (ingest/emt_opendata.py --demanda)"
    assert PILOTO.exists(), f"Falta {PILOTO} (codigo/notebooks/seleccion_piloto.py)"

    piloto = pd.read_csv(PILOTO, dtype={"linea": str})["linea"].tolist()
    df = pd.read_parquet(DEMANDA)
    df = df[df["linea"].isin(piloto)].copy()
    print(f"[features] demanda piloto: {len(df):,} filas, {df['linea'].nunique()} líneas")

    # --- Calendario ---
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.dayofweek        # 0=lunes
    df["semana"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["dia_anio"] = df["fecha"].dt.dayofyear
    df["finde"] = (df["dia_semana"] >= 5).astype(int)
    df["is_festivo"] = _festivos(df["fecha"]).astype(int)
    df["covid"] = df["fecha"].between(COVID_INI, COVID_FIN).astype(int)

    # --- Meteorología (AEMET) ---
    if AEMET.exists():
        df = df.merge(_clima_diario(), on="fecha", how="left")
    else:
        print(f"[features] aviso: falta {AEMET}; sin features meteo")

    # --- Saturación media por línea (atributo estático, disponible siempre) ---
    if OCUP.exists():
        ocup = pd.read_parquet(OCUP)
        ocup["saturacion"] = 100 - ocup["pct_calidad_ocupacion"]
        sat = (ocup[ocup["anio"] >= 2023].groupby("linea")["saturacion"].mean()
               .rename("sat_linea_media").reset_index())
        df = df.merge(sat, on="linea", how="left")

    # --- Lags y medias móviles por línea (sin fuga: desplazados >=7 días) ---
    df = añadir_lags_rolling(df)

    antes = len(df)
    df = df.dropna(subset=[f"lag_{max(LAGS)}"]).reset_index(drop=True)
    print(f"[features] descartadas {antes - len(df):,} filas iniciales sin histórico de lags")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)
    print(f"[features] {len(df):,} filas × {df.shape[1]} columnas")
    print(f"[features] rango {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"[features] columnas: {list(df.columns)}")
    print(f"[features] guardado en {SALIDA}")


if __name__ == "__main__":
    main()
