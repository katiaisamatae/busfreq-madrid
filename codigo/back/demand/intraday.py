"""Reparto intradía de la demanda diaria a franjas horarias — Módulo 1/2.

La demanda pública es día × línea; el optimizador necesita la dimensión de
franja horaria. Este módulo estima esa distribución intradía a partir del
perfil de la OFERTA real (coches en servicio por línea, tipo de día y franja),
que sí está disponible por franja.

Idea: para cada línea y tipo de día (LA laborable, SA sábado, FE festivo), se
calcula qué fracción del servicio diario se concentra en cada franja. Ese perfil
se aplica al total diario de demanda (previsto o real) para repartirlo por franja.

    demanda_franja(l, fecha, f) = demanda_dia(l, fecha) * share(l, tipo_dia(fecha), f)

Salidas:
  - datos/processed/perfil_intradia.parquet   [linea, tipo_dia, franja, share]
  - notebooks/figs/perfil_intradia.png

Uso:
    python -m demand.intraday
"""
from __future__ import annotations

import glob
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import DATA_PROCESSED, DATA_RAW

ROOT = Path(__file__).resolve().parent.parent
OFERTA_DIR = DATA_RAW / "oferta"
GTFS_ZIP = DATA_RAW / "gtfs_emt.zip"
PILOTO = DATA_PROCESSED / "lineas_piloto.csv"
FIGS = ROOT.parent / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def _mapa_codigo_a_linea() -> dict[str, str]:
    """route_id (código, p.ej. '070') -> route_short_name (p.ej. '70')."""
    with zipfile.ZipFile(GTFS_ZIP) as z, z.open("routes.txt") as f:
        routes = pd.read_csv(f, dtype=str)
    return dict(zip(routes["route_id"], routes["route_short_name"]))


def cargar_oferta(desde: str = "2023-01-01") -> pd.DataFrame:
    """Oferta desde `desde` (por defecto 2023): pondera por recencia y evita el
    régimen COVID 2020-2021, que no representa la forma actual del día."""
    files = sorted(glob.glob(str(OFERTA_DIR / "*.csv")))
    assert files, f"No hay CSVs de oferta en {OFERTA_DIR} (emt_opendata.py --oferta)"
    dfs = []
    for f in files:
        d = pd.read_csv(f, sep=";", encoding="utf-8-sig",
                        dtype={"Linea": str, "TIPODIAMO": str,
                               "CodFranja": int, "NumCCReal": float})
        dfs.append(d[["Linea", "TIPODIAMO", "CodFranja", "NumCCReal", "FECHASER"]])
    of = pd.concat(dfs, ignore_index=True)
    of["fecha"] = pd.to_datetime(of["FECHASER"], errors="coerce").dt.normalize()
    return of[of["fecha"] >= pd.Timestamp(desde)].reset_index(drop=True)


def construir_perfiles(oferta: pd.DataFrame, mapa: dict[str, str]) -> pd.DataFrame:
    """Perfil normalizado de servicio por línea, tipo de día y franja.

    Circularidad (honestidad): repartir la demanda diaria en proporción a la
    OFERTA asume que la EMT dimensiona bien la FORMA del día dentro de cada línea;
    el optimizador solo cuestiona el NIVEL relativo entre líneas. Por eso el ahorro
    estimado es un LÍMITE INFERIOR (no ve el desajuste intradía dentro de una línea).
    """
    oferta = oferta.copy()
    oferta["linea"] = oferta["Linea"].map(mapa)
    oferta = oferta.dropna(subset=["linea"])
    # Suma de coches por (linea, tipo_dia, franja) a lo largo del histórico.
    agg = (oferta.groupby(["linea", "TIPODIAMO", "CodFranja"])["NumCCReal"]
           .sum().reset_index()
           .rename(columns={"TIPODIAMO": "tipo_dia", "CodFranja": "franja"}))
    # Normaliza a fracción del día (suma 1 por línea y tipo de día).
    total = agg.groupby(["linea", "tipo_dia"])["NumCCReal"].transform("sum")
    agg["share"] = (agg["NumCCReal"] / total).fillna(0.0)
    return agg[["linea", "tipo_dia", "franja", "share"]]


def tipo_dia(fecha: pd.Timestamp, es_festivo: bool) -> str:
    """Clasifica una fecha en el esquema de la oferta: LA / SA / FE."""
    if es_festivo or fecha.dayofweek == 6:   # domingo o festivo
        return "FE"
    if fecha.dayofweek == 5:                 # sábado
        return "SA"
    return "LA"


def repartir(demanda_diaria: pd.DataFrame, perfiles: pd.DataFrame) -> pd.DataFrame:
    """Reparte la demanda diaria [linea, fecha, viajeros, tipo_dia] por franja.

    Devuelve [linea, fecha, franja, viajeros_franja].
    """
    d = demanda_diaria.merge(perfiles, on=["linea", "tipo_dia"], how="left")
    d["viajeros_franja"] = d["viajeros"] * d["share"]
    return d[["linea", "fecha", "franja", "viajeros_franja"]]


def main() -> None:
    mapa = _mapa_codigo_a_linea()
    oferta = cargar_oferta()
    perfiles = construir_perfiles(oferta, mapa)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    salida = DATA_PROCESSED / "perfil_intradia.parquet"
    perfiles.to_parquet(salida, index=False)
    print(f"[intraday] perfiles: {len(perfiles):,} filas | "
          f"{perfiles['linea'].nunique()} líneas | tipos {sorted(perfiles['tipo_dia'].unique())}")
    print(f"[intraday] guardado en {salida}")

    # Comprobación: las shares suman ~1 por (linea, tipo_dia).
    chk = perfiles.groupby(["linea", "tipo_dia"])["share"].sum()
    print(f"[intraday] suma de shares (debe ser ~1): min={chk.min():.3f} max={chk.max():.3f}")

    # --- Figura: perfil intradía de las líneas piloto en día laborable ---
    piloto = (pd.read_csv(PILOTO, dtype={"linea": str})["linea"].tolist()
              if PILOTO.exists() else perfiles["linea"].unique()[:6].tolist())
    la = perfiles[(perfiles["tipo_dia"] == "LA") & (perfiles["linea"].isin(piloto))]
    fig, ax = plt.subplots(figsize=(11, 5))
    for linea, g in la.groupby("linea"):
        g = g.sort_values("franja")
        ax.plot(g["franja"], g["share"] * 100, marker=".", label=linea, lw=1)
    ax.set_title("Perfil intradía del servicio (oferta) — día laborable, líneas piloto")
    ax.set_xlabel("Franja horaria (código, orden cronológico)")
    ax.set_ylabel("% del servicio diario")
    ax.legend(ncol=3, fontsize=8, title="Línea")
    plt.tight_layout()
    plt.savefig(FIGS / "perfil_intradia.png", dpi=120)
    plt.close()
    print(f"[intraday] figura en {FIGS / 'perfil_intradia.png'}")

    # --- Demo: reparto de un día para una línea piloto ---
    if PILOTO.exists():
        dem = pd.read_parquet(DATA_PROCESSED / "demanda_diaria_linea.parquet")
        dem = dem[dem["linea"].isin(piloto)].copy()
        # tipo de día aproximado (sin festivos aquí: fin de semana por dow)
        dem["tipo_dia"] = dem["fecha"].apply(lambda f: tipo_dia(f, False))
        muestra = dem[dem["fecha"] == dem["fecha"].max()]
        rep = repartir(muestra, perfiles)
        linea_ej = muestra.loc[muestra["viajeros"].idxmax(), "linea"]
        tot = rep[rep["linea"] == linea_ej]["viajeros_franja"].sum()
        print(f"[intraday] demo reparto {muestra['fecha'].max().date()} línea {linea_ej}: "
              f"{tot:,.0f} viajeros repartidos en {rep[rep['linea']==linea_ej]['franja'].nunique()} franjas")


if __name__ == "__main__":
    main()
