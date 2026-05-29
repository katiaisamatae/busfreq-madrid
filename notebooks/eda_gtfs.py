"""EDA de la red GTFS de la EMT Madrid — versión script (Fase 1).

Equivalente al notebook 01_eda_gtfs.ipynb pero ejecutable sin jupyter.
Imprime un resumen por consola y guarda las figuras en notebooks/figs/.

Uso (desde la raíz del proyecto, con el .venv):
    .venv/bin/python notebooks/eda_gtfs.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla (guarda a archivo)
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GTFS_ZIP = ROOT / "data" / "raw" / "gtfs_emt.zip"
FIGS = ROOT / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def load_gtfs(name: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(GTFS_ZIP) as z:
        with z.open(name) as f:
            return pd.read_csv(f, dtype=str, **kwargs)


def main() -> None:
    assert GTFS_ZIP.exists(), f"No encuentro {GTFS_ZIP}"

    routes = load_gtfs("routes.txt")
    stops = load_gtfs("stops.txt")
    trips = load_gtfs("trips.txt")
    calendar = load_gtfs("calendar.txt")
    frequencies = load_gtfs("frequencies.txt")

    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    frequencies["headway_secs"] = frequencies["headway_secs"].astype(int)

    print("=" * 60)
    print("RESUMEN DE LA RED EMT MADRID (GTFS)")
    print("=" * 60)
    print(f"Líneas:      {routes['route_short_name'].nunique():>7}")
    print(f"Paradas:     {len(stops):>7,}")
    print(f"Viajes:      {len(trips):>7,}")
    print(f"Tramos freq: {len(frequencies):>7,}")
    print()
    print("Tipos de servicio (calendar.txt):")
    print(calendar[["service_id", "monday", "saturday", "sunday"]].to_string(index=False))

    # --- Frecuencias enriquecidas con línea ---
    trip_line = trips[["trip_id", "route_id", "service_id"]].merge(
        routes[["route_id", "route_short_name"]], on="route_id", how="left"
    )
    freq = frequencies.merge(trip_line, on="trip_id", how="left")
    freq["headway_min"] = freq["headway_secs"] / 60
    freq["start_hour"] = freq["start_time"].str.split(":").str[0].astype(int)

    print()
    print("Cadencia (min entre buses) — estadísticos globales:")
    print(freq["headway_min"].describe().round(1).to_string())

    # --- Gráfica 1: cadencia media por hora (laborable) ---
    la = freq[freq["service_id"] == "LA"]
    by_hour = la.groupby("start_hour")["headway_min"].mean()
    ax = by_hour.plot(kind="bar", figsize=(11, 4), color="#0178BC")
    ax.set_title("Cadencia media por hora — laborable (menor = más frecuente)")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Headway medio (min)")
    plt.tight_layout()
    plt.savefig(FIGS / "cadencia_por_hora.png", dpi=120)
    plt.close()

    # --- Top líneas por variabilidad (candidatas al piloto) ---
    var_by_line = (
        la.groupby("route_short_name")["headway_min"]
        .agg(["min", "max", "mean", "std"])
        .dropna()
        .sort_values("std", ascending=False)
    )
    print()
    print("Top 15 líneas por variabilidad de cadencia (candidatas al piloto):")
    print(var_by_line.head(15).round(1).to_string())

    # --- Gráfica 2: mapa de paradas ---
    paradas = stops[stops["location_type"].isna() | (stops["location_type"] == "0")]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(paradas["stop_lon"], paradas["stop_lat"], s=2, alpha=0.4, color="#0178BC")
    ax.set_title(f"Paradas de la EMT Madrid (n={len(paradas):,})")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(FIGS / "mapa_paradas.png", dpi=120)
    plt.close()

    print()
    print(f"Figuras guardadas en: {FIGS}")
    print("  - cadencia_por_hora.png")
    print("  - mapa_paradas.png")


if __name__ == "__main__":
    main()
