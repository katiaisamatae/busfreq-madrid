"""Tiempo de ciclo por línea desde el GTFS — entrada del optimizador (Fase 2).

La espera de los usuarios es headway/2 = ciclo_linea/(2·n), NO 60/(2·n): asumir
una duración fija de 60 min para todas las líneas distorsiona el reparto óptimo,
porque los ciclos difieren mucho entre líneas. Este script estima el tiempo de
ciclo (ida + vuelta, en minutos) de cada línea piloto a partir de `stop_times`
del GTFS y lo guarda para que el optimizador pondere la espera con el ciclo real.

Salida: datos/processed/ciclos_linea.parquet [linea, ciclo_min, dur_media_min, n_trips]

Uso:
    python -m optimizer.ciclos
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import DATA_PROCESSED, DATA_RAW

GTFS_ZIP = DATA_RAW / "gtfs_emt.zip"
PILOTO = DATA_PROCESSED / "lineas_piloto.csv"
SALIDA = DATA_PROCESSED / "ciclos_linea.parquet"


def _a_minutos(s: pd.Series) -> pd.Series:
    """Convierte 'H:MM:SS' del GTFS (admite horas >= 24) a minutos."""
    p = s.str.split(":", expand=True).apply(pd.to_numeric, errors="coerce")
    return p[0] * 60 + p[1] + p[2] / 60.0


def calcular_ciclos(piloto: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(GTFS_ZIP) as z:
        with z.open("routes.txt") as f:
            routes = pd.read_csv(f, dtype=str)[["route_id", "route_short_name"]]
        with z.open("trips.txt") as f:
            trips = pd.read_csv(f, dtype=str)[["route_id", "trip_id"]]
        with z.open("stop_times.txt") as f:
            st = pd.read_csv(f, usecols=["trip_id", "arrival_time", "departure_time",
                                         "stop_sequence"],
                            dtype={"trip_id": str, "arrival_time": str,
                                   "departure_time": str})
    # Solo viajes de líneas piloto (reduce stop_times antes de agregar).
    routes["linea"] = routes["route_short_name"]
    trips = trips.merge(routes[["route_id", "linea"]], on="route_id", how="left")
    trips = trips[trips["linea"].isin(piloto)]
    st = st[st["trip_id"].isin(set(trips["trip_id"]))].copy()
    st = st.dropna(subset=["arrival_time", "departure_time"])
    st = st.sort_values(["trip_id", "stop_sequence"])

    # Duración de cada viaje (un sentido): última salida - primera llegada.
    ini = _a_minutos(st.groupby("trip_id")["arrival_time"].first())
    fin = _a_minutos(st.groupby("trip_id")["departure_time"].last())
    dur = (fin - ini).rename("dur_min")
    dur = dur[dur > 0]

    trips = trips.merge(dur, left_on="trip_id", right_index=True)
    agg = (trips.groupby("linea")
           .agg(dur_media_min=("dur_min", "mean"), n_trips=("trip_id", "nunique"))
           .reset_index())
    # Ciclo (ida + vuelta) ~= duración media de un sentido x 2.
    agg["ciclo_min"] = (agg["dur_media_min"] * 2).round(1)
    return agg[["linea", "ciclo_min", "dur_media_min", "n_trips"]].sort_values("linea")


def main() -> None:
    assert PILOTO.exists(), f"Falta {PILOTO}"
    piloto = pd.read_csv(PILOTO, dtype={"linea": str})["linea"].tolist()
    out = calcular_ciclos(piloto)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SALIDA, index=False)
    print(out.round(1).to_string(index=False))
    print(f"[ciclos] ciclo medio {out['ciclo_min'].mean():.1f} min "
          f"(min {out['ciclo_min'].min():.0f}, max {out['ciclo_min'].max():.0f}) "
          f"vs T=60 fijo del modelo anterior")
    print(f"[ciclos] guardado en {SALIDA}")


if __name__ == "__main__":
    main()
