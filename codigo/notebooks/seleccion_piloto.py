"""Selección definitiva de líneas piloto — cruce demanda × GTFS (Fase 1).

Combina los tres criterios del proyecto (CLAUDE.md):
  1. Volumen de demanda (que la línea sea relevante).
  2. Variabilidad de demanda (donde optimizar la frecuencia tiene más impacto).
  3. Diversidad geográfica (líneas repartidas por sectores de Madrid).

Entradas:
  - datos/processed/demanda_diaria_linea.parquet  (ingest/emt_opendata.py)
  - datos/raw/gtfs_emt.zip                         (topología de la red)

Salidas:
  - datos/processed/lineas_piloto.csv              (lista definitiva)
  - notebooks/figs/mapa_lineas_piloto.png

Uso:
    python notebooks/seleccion_piloto.py
"""
from __future__ import annotations

import math
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # raíz del repo (TFM/)
PARQUET = ROOT / "datos" / "processed" / "demanda_diaria_linea.parquet"
GTFS_ZIP = ROOT / "datos" / "raw" / "gtfs_emt.zip"
FIGS = Path(__file__).resolve().parent / "figs"
OUT_CSV = ROOT / "datos" / "processed" / "lineas_piloto.csv"
FIGS.mkdir(parents=True, exist_ok=True)

OBJETIVO = 12          # nº de líneas piloto a seleccionar
VENTANA_DIAS = 730     # ventana reciente para stats (evita distorsión COVID)
SECTORES = ["E", "NE", "N", "NO", "O", "SO", "S", "SE"]  # 8 sectores de 45º


def load_gtfs(name: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(GTFS_ZIP) as z, z.open(name) as f:
        return pd.read_csv(f, dtype=str, **kwargs)


def stats_demanda() -> pd.DataFrame:
    """Estadísticos de demanda por línea en la ventana reciente."""
    df = pd.read_parquet(PARQUET)
    corte = df["fecha"].max() - pd.Timedelta(days=VENTANA_DIAS)
    reciente = df[df["fecha"] >= corte]
    stats = (
        reciente.groupby("linea")["viajeros"]
        .agg(dias="count", media="mean", std="std", p95=lambda s: s.quantile(0.95))
        .assign(cv=lambda x: x["std"] / x["media"])
        .query("dias >= 300")
    )
    return stats


def geografia_lineas(candidatas: set[str]) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Centroide y nº de paradas por línea (route_short_name) desde el GTFS."""
    routes = load_gtfs("routes.txt")[["route_id", "route_short_name", "route_long_name"]]
    stops = load_gtfs("stops.txt")[["stop_id", "stop_lat", "stop_lon"]]
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)

    centro_red = (stops["stop_lat"].mean(), stops["stop_lon"].mean())

    cand_routes = routes[routes["route_short_name"].isin(candidatas)]
    trips = load_gtfs("trips.txt")[["route_id", "trip_id"]]
    trips = trips[trips["route_id"].isin(cand_routes["route_id"])]

    st = load_gtfs("stop_times.txt", usecols=["trip_id", "stop_id"])
    st = st[st["trip_id"].isin(trips["trip_id"])].merge(trips, on="trip_id")

    st = st[["route_id", "stop_id"]].drop_duplicates().merge(stops, on="stop_id")
    geo = (
        st.groupby("route_id")
        .agg(n_paradas=("stop_id", "nunique"),
             lat=("stop_lat", "mean"), lon=("stop_lon", "mean"))
        .reset_index()
        .merge(cand_routes, on="route_id")
        .set_index("route_short_name")
    )
    return geo, centro_red


def sector(lat: float, lon: float, centro: tuple[float, float]) -> str:
    """Sector compás (8) del centroide de la línea respecto al centro de la red."""
    ang = math.degrees(math.atan2(lat - centro[0], lon - centro[1])) % 360
    return SECTORES[int((ang + 22.5) % 360 // 45)]


def seleccionar(tabla: pd.DataFrame) -> pd.DataFrame:
    """Greedy: prioriza variabilidad garantizando reparto por sectores."""
    pool = tabla.sort_values("cv", ascending=False)
    elegidas: list[str] = []
    por_sector: dict[str, int] = defaultdict(int)
    for cap in range(1, 6):  # relaja el cupo por sector si hace falta
        for linea, row in pool.iterrows():
            if len(elegidas) >= OBJETIVO:
                break
            if linea in elegidas:
                continue
            if por_sector[row["sector"]] < cap:
                elegidas.append(linea)
                por_sector[row["sector"]] += 1
        if len(elegidas) >= OBJETIVO:
            break
    return tabla.loc[elegidas]


def main() -> None:
    stats = stats_demanda()

    # Pool relevante: volumen alto (>= P60) para no elegir líneas testimoniales.
    umbral = stats["media"].quantile(0.60)
    pool = stats[stats["media"] >= umbral].copy()

    geo, centro = geografia_lineas(set(pool.index))
    tabla = pool.join(geo, how="inner")  # solo líneas con match en GTFS
    sin_match = sorted(set(pool.index) - set(tabla.index), key=lambda x: (len(x), x))
    if sin_match:
        print(f"Líneas de demanda sin match en GTFS (descartadas): {sin_match}")

    tabla["sector"] = [sector(r.lat, r.lon, centro) for r in tabla.itertuples()]
    piloto = seleccionar(tabla).sort_values("cv", ascending=False)

    print("=" * 78)
    print(f"LÍNEAS PILOTO SELECCIONADAS ({len(piloto)}) — demanda × GTFS")
    print("=" * 78)
    cols = ["route_long_name", "media", "cv", "p95", "n_paradas", "sector"]
    vista = piloto[cols].rename(columns={"route_long_name": "recorrido"})
    print(vista.round({"media": 0, "cv": 3, "p95": 0}).to_string())
    print(f"\nReparto por sector: {piloto['sector'].value_counts().to_dict()}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    piloto.reset_index().rename(columns={"index": "linea"}).to_csv(OUT_CSV, index=False)
    print(f"Lista guardada en {OUT_CSV}")

    # --- Mapa: red completa + líneas piloto resaltadas ---
    all_stops = load_gtfs("stops.txt")[["stop_lat", "stop_lon"]].astype(float)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(all_stops["stop_lon"], all_stops["stop_lat"], s=1, alpha=0.15,
               color="lightgrey")
    cmap = plt.get_cmap("tab20")
    for i, (linea, row) in enumerate(piloto.iterrows()):
        ax.scatter(row["lon"], row["lat"], s=90, color=cmap(i % 20),
                   edgecolor="black", zorder=3)
        ax.annotate(linea, (row["lon"], row["lat"]), fontsize=9, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points")
    ax.scatter(*centro[::-1], marker="+", s=200, color="red", label="centro de la red")
    ax.set_title(f"Líneas piloto seleccionadas (n={len(piloto)}) sobre la red EMT")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "mapa_lineas_piloto.png", dpi=120)
    plt.close()
    print(f"Mapa guardado en {FIGS / 'mapa_lineas_piloto.png'}")


if __name__ == "__main__":
    main()
