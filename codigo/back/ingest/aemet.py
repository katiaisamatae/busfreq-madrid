"""Ingesta de datos meteorológicos de AEMET OpenData (Fase 1).

Descarga:
  * Histórico climatológico diario por estación (Madrid-Retiro 3195,
    Barajas 3129), paginado por tramos (la API limita el rango por llamada).
  * Previsión diaria del municipio de Madrid (28079).

AEMET responde en 2 pasos (1ª llamada -> URL 'datos'; 2ª -> contenido, en
ISO-8859-15). Todas las peticiones llevan reintentos por los cortes TLS de la red.

Uso:
    python -m ingest.aemet --historico --ini 2019-01-01 --fin 2026-07-09
    python -m ingest.aemet --prevision
    python -m ingest.aemet --all
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import AEMET_API_KEY, DATA_PROCESSED, DATA_RAW, require

BASE = "https://opendata.aemet.es/opendata/api"
ESTACIONES_DEF = ["3195", "3129"]  # Madrid-Retiro, Madrid-Barajas
MUNICIPIO_MADRID = "28079"
TRAMO_DIAS = 150  # < 6 meses por llamada (límite del climatológico diario)


def _get(url: str, session: requests.Session, intentos: int = 6, **kwargs):
    for i in range(intentos):
        try:
            return session.get(url, timeout=60, **kwargs)
        except (requests.exceptions.SSLError, requests.exceptions.Timeout):
            print(f"      (corte de red, reintento {i + 1}/{intentos})")
            time.sleep(2)
    return None


def _aemet_datos(path: str, api_key: str, session: requests.Session,
                 reintentos_429: int = 5):
    """Flujo de 2 pasos: devuelve el JSON de datos (o None).

    Ante estado 429 (límite de peticiones por minuto) espera y reintenta, en
    lugar de descartar el tramo (evita lagunas en el histórico).
    """
    for _ in range(reintentos_429 + 1):
        r1 = _get(f"{BASE}{path}", session, headers={"api_key": api_key})
        if r1 is None:
            return None
        try:
            meta = r1.json()
        except ValueError:
            return None
        if meta.get("estado") == 429:
            print("      429 (límite/min): esperando 65 s y reintentando...")
            time.sleep(65)
            continue
        if meta.get("estado") != 200 or "datos" not in meta:
            print(f"      estado={meta.get('estado')} "
                  f"desc='{meta.get('descripcion')}'")
            return None
        r2 = _get(meta["datos"], session)
        if r2 is None:
            return None
        r2.encoding = "ISO-8859-15"  # los ficheros de AEMET NO son UTF-8
        try:
            return r2.json()
        except ValueError:
            return None
    print("      429 persistente, tramo omitido")
    return None


def _num(x) -> float | None:
    """Convierte los numéricos de AEMET (coma decimal, 'Ip', vacío)."""
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "Ip", "Acum", "varias"):
        return 0.0 if s == "Ip" else None  # Ip = precipitación inapreciable
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _tramos(ini: date, fin: date, dias: int = TRAMO_DIAS):
    a = ini
    while a <= fin:
        b = min(a + timedelta(days=dias - 1), fin)
        yield a, b
        a = b + timedelta(days=1)


# --------------------------------------------------------------------------- #
def descargar_historico(estaciones: list[str], ini: date, fin: date,
                        api_key: str, session: requests.Session) -> Path:
    frames = []
    for est in estaciones:
        print(f"[aemet] histórico estación {est} ({ini} → {fin})")
        for a, b in _tramos(ini, fin):
            path = (f"/valores/climatologicos/diarios/datos/"
                    f"fechaini/{a}T00:00:00UTC/fechafin/{b}T23:59:59UTC/"
                    f"estacion/{est}")
            print(f"   tramo {a} → {b}")
            datos = _aemet_datos(path, api_key, session)
            if datos:
                frames.append(pd.DataFrame(datos))
            time.sleep(1)  # cortesía con la API

    if not frames:
        raise RuntimeError("AEMET no devolvió datos históricos")
    crudo = pd.concat(frames, ignore_index=True)
    (DATA_RAW / "aemet").mkdir(parents=True, exist_ok=True)
    crudo.to_csv(DATA_RAW / "aemet" / "historico_crudo.csv", index=False)

    df = pd.DataFrame({
        "fecha": pd.to_datetime(crudo["fecha"], errors="coerce"),
        "estacion": crudo["indicativo"],
        "tmed": crudo["tmed"].map(_num),
        "tmin": crudo["tmin"].map(_num),
        "tmax": crudo["tmax"].map(_num),
        "prec": crudo["prec"].map(_num),
        "velmedia": crudo.get("velmedia", pd.Series()).map(_num),
        "racha": crudo.get("racha", pd.Series()).map(_num),
        "sol": crudo.get("sol", pd.Series()).map(_num),
    }).dropna(subset=["fecha"]).sort_values(["estacion", "fecha"])

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    destino = DATA_PROCESSED / "aemet_historico.parquet"
    df.to_parquet(destino, index=False)
    print(f"[aemet] {len(df):,} filas | estaciones {sorted(df['estacion'].unique())} "
          f"| {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"[aemet] guardado en {destino}")
    return destino


def descargar_prevision(municipio: str, api_key: str,
                        session: requests.Session) -> Path:
    print(f"[aemet] previsión municipio {municipio}")
    datos = _aemet_datos(
        f"/prediccion/especifica/municipio/diaria/{municipio}", api_key, session)
    if not datos:
        raise RuntimeError("AEMET no devolvió previsión")
    dias = datos[0].get("prediccion", {}).get("dia", [])
    registros = []
    for d in dias:
        prob = [p.get("value") for p in d.get("probPrecipitacion", [])
                if p.get("value") is not None]
        registros.append({
            "fecha": pd.to_datetime(d.get("fecha")),
            "municipio": datos[0].get("nombre"),
            "tmax": d.get("temperatura", {}).get("maxima"),
            "tmin": d.get("temperatura", {}).get("minima"),
            "prob_prec_max": max(prob) if prob else None,
        })
    df = pd.DataFrame(registros)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    destino = DATA_PROCESSED / "aemet_prevision.parquet"
    df.to_parquet(destino, index=False)
    print(f"[aemet] previsión {len(df)} días: {df['fecha'].min().date()} → "
          f"{df['fecha'].max().date()}")
    print(f"[aemet] guardado en {destino}")
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta AEMET OpenData")
    parser.add_argument("--historico", action="store_true")
    parser.add_argument("--prevision", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--estaciones", default=",".join(ESTACIONES_DEF),
                        help="IDs de estación separados por coma")
    parser.add_argument("--ini", default="2019-01-01", help="YYYY-MM-DD")
    parser.add_argument("--fin", default=date.today().isoformat(), help="YYYY-MM-DD")
    args = parser.parse_args()

    if not (args.historico or args.prevision or args.all):
        parser.error("Indica --historico, --prevision o --all")

    api_key = require("AEMET_API_KEY", AEMET_API_KEY)
    session = requests.Session()

    if args.historico or args.all:
        descargar_historico(
            args.estaciones.split(","),
            date.fromisoformat(args.ini), date.fromisoformat(args.fin),
            api_key, session)
    if args.prevision or args.all:
        descargar_prevision(MUNICIPIO_MADRID, api_key, session)


if __name__ == "__main__":
    main()
