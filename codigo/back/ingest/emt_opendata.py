"""Descarga de datasets del portal de datos abiertos de EMT (datos.emtmadrid.es).

Fase 1 — Datos. La demanda histórica NO está en la API en tiempo real, sino
aquí (ver sonda scripts/probe_emt_validations.py). Este módulo descarga y
normaliza:

  * Demanda diaria de viajeros  -> día x línea (histórico 2019+).
  * Oferta de autobuses         -> dotación real por línea x franja x mes.

La red de trabajo corta la conexión TLS de forma intermitente con
datos.emtmadrid.es, así que todas las descargas van con reintentos.

Uso:
    python ingest/emt_opendata.py --demanda     # descarga + normaliza demanda
    python ingest/emt_opendata.py --oferta      # descarga CSVs de oferta
    python ingest/emt_opendata.py --all         # ambos
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Permite ejecutar como script (python ingest/emt_opendata.py) además de
# como módulo (python -m ingest.emt_opendata).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import DATA_PROCESSED, DATA_RAW

CKAN_BASE = "https://datos.emtmadrid.es/api/3/action"
DATASET_DEMANDA = "demanda-diaria-viajeros-autobus"
DATASET_OFERTA = "oferta-de-autobuses-diaria"
DATASET_CALENDARIO = "calendario-de-operacion-de-lineas-de-autobuses-de-emtmadrid"

# Grado de ocupación: sólo está en el portal del Ayuntamiento (datos.madrid.es),
# un XLS por año. El sufijo del recurso cambia por año (no sigue orden natural).
OCUPACION_URL = (
    "https://datos.madrid.es/dataset/300342-0-emt-grado-ocupacion/resource/"
    "300342-{suf}-emt-grado-ocupacion/download/300342-{suf}-emt-grado-ocupacion.xls"
)
# (El recurso "2018"/sufijo 6 es en realidad otro dataset —velocidad comercial—,
# por eso no se incluye. La ocupación cubre 2019-2024 con formato homogéneo.)
OCUPACION_SUFIJOS = {2024: "0", 2023: "3", 2022: "2", 2021: "1",
                     2020: "4", 2019: "5"}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _session() -> requests.Session:
    return requests.Session()


def _get(url: str, session: requests.Session, intentos: int = 6, **kwargs):
    """GET con reintentos ante cortes TLS / timeouts."""
    ultimo = None
    for i in range(intentos):
        try:
            r = session.get(url, timeout=90, **kwargs)
            r.raise_for_status()
            return r
        except (requests.exceptions.SSLError, requests.exceptions.Timeout) as e:
            ultimo = e
            print(f"    (corte de red, reintento {i + 1}/{intentos})")
            time.sleep(2)
    raise RuntimeError(f"No se pudo descargar {url}: {ultimo}")


def ckan_recursos(dataset: str, session: requests.Session) -> list[dict]:
    """Devuelve la lista de recursos {name, format, url} de un dataset CKAN."""
    r = _get(f"{CKAN_BASE}/package_show", session, params={"id": dataset})
    return r.json()["result"]["resources"]


def descargar_archivo(url: str, destino: Path, session: requests.Session) -> Path:
    """Descarga una URL a `destino` (crea carpetas). Devuelve la ruta."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    r = _get(url, session)
    destino.write_bytes(r.content)
    print(f"    guardado {destino.name} ({len(r.content):,} bytes)")
    return destino


# --------------------------------------------------------------------------- #
# Demanda
# --------------------------------------------------------------------------- #
def descargar_demanda(session: requests.Session) -> Path:
    """Descarga el CSV de demanda diaria por línea a datos/raw/."""
    print("[demanda] localizando recurso CSV en CKAN...")
    recursos = ckan_recursos(DATASET_DEMANDA, session)
    csv = next((x for x in recursos if (x.get("format") or "").upper() == "CSV"), None)
    if csv is None:
        raise RuntimeError("No encontré recurso CSV en el dataset de demanda")
    destino = DATA_RAW / "demandadialinea.csv"
    print(f"[demanda] descargando {csv['url']}")
    return descargar_archivo(csv["url"], destino, session)


def normalizar_demanda(raw_csv: Path) -> Path:
    """Normaliza el CSV crudo a esquema tabular y lo guarda en processed.

    Salida: columnas [fecha (datetime), linea (str), viajeros (int)].
    """
    print("[demanda] normalizando...")
    df = pd.read_csv(raw_csv, sep=";", encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={"Fecha": "fecha", "Linea": "linea", "TotalViajeros": "viajeros"}
    )
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df["viajeros"] = pd.to_numeric(df["viajeros"], errors="coerce")
    df["linea"] = df["linea"].str.strip()
    antes = len(df)
    df = df.dropna(subset=["fecha", "linea", "viajeros"])
    df["viajeros"] = df["viajeros"].astype(int)
    df = df.sort_values(["fecha", "linea"]).reset_index(drop=True)
    if antes != len(df):
        print(f"    descartadas {antes - len(df)} filas con datos inválidos")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    destino = DATA_PROCESSED / "demanda_diaria_linea.parquet"
    df.to_parquet(destino, index=False)
    print(f"[demanda] {len(df):,} filas | {df['linea'].nunique()} líneas | "
          f"{df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"[demanda] guardado en {destino}")
    return destino


# --------------------------------------------------------------------------- #
# Oferta
# --------------------------------------------------------------------------- #
def descargar_oferta(session: requests.Session) -> list[Path]:
    """Descarga todos los CSV mensuales de oferta a datos/raw/oferta/."""
    print("[oferta] localizando recursos CSV en CKAN...")
    recursos = ckan_recursos(DATASET_OFERTA, session)
    csvs = [x for x in recursos if (x.get("format") or "").upper() == "CSV" and x.get("url")]
    print(f"[oferta] {len(csvs)} CSV encontrados")
    out_dir = DATA_RAW / "oferta"
    rutas = []
    for x in csvs:
        nombre = x["url"].rsplit("/", 1)[-1]
        try:
            rutas.append(descargar_archivo(x["url"], out_dir / nombre, session))
        except RuntimeError as e:
            print(f"    ⚠️  saltado {nombre}: {e}")
    return rutas


# --------------------------------------------------------------------------- #
# Calendario de operación (tipo de día, temporada, clima) — features de calendario
# --------------------------------------------------------------------------- #
def descargar_calendario(session: requests.Session) -> Path:
    """Descarga y normaliza el calendario de operación de líneas de la EMT.

    Salida: datos/processed/calendario.parquet con la fecha como índice y el tipo
    de día oficial (TipoDiaEs: LA laborable, SA sábado, FE festivo...), temporada,
    día de la semana, mes/trimestre y clima/temperatura cuando están disponibles.
    """
    print("[calendario] localizando recurso CSV en CKAN...")
    recursos = ckan_recursos(DATASET_CALENDARIO, session)
    csv = next((x for x in recursos if (x.get("format") or "").upper() == "CSV"), None)
    if csv is None:
        raise RuntimeError("No encontré recurso CSV en el dataset de calendario")
    raw = descargar_archivo(csv["url"], DATA_RAW / "calendario.csv", session)

    print("[calendario] normalizando...")
    df = pd.read_csv(raw, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Fecha": "fecha", "TipoDiaEs": "tipo_dia",
                            "TemporadaTG": "temporada", "DiaSemana": "dia_semana",
                            "Semana": "semana", "Mes": "mes", "Trimestre": "trimestre",
                            "Clima": "clima", "TemperaturaMax": "temp_max",
                            "TemperaturaMin": "temp_min"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for c in ("clima",):
        if c in df:
            df[c] = df[c].astype(str).str.strip().replace("", pd.NA)
    df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    destino = DATA_PROCESSED / "calendario.parquet"
    df.to_parquet(destino, index=False)
    print(f"[calendario] {len(df):,} días | {df['fecha'].min().date()} → "
          f"{df['fecha'].max().date()} | tipos de día: "
          f"{sorted(df['tipo_dia'].dropna().unique())}")
    print(f"[calendario] guardado en {destino}")
    return destino


# --------------------------------------------------------------------------- #
# Grado de ocupación (proxy de saturación) — línea x mes, 2018-2024
# --------------------------------------------------------------------------- #
def _leer_ocupacion_xls(content: bytes, anio: int) -> pd.DataFrame:
    """Parsea el XLS (formato irregular) de ocupación de un año a formato tidy.

    Localiza dinámicamente la fila con los nombres de mes y en qué columnas están.
    Si el fichero no tiene ese layout (p.ej. un dataset distinto), devuelve vacío.
    """
    engine = "xlrd" if content[:4].hex() == "d0cf11e0" else "openpyxl"
    crudo = pd.read_excel(io.BytesIO(content), engine=engine, header=None)
    texto = crudo.apply(lambda s: s.astype(str).str.strip().str.lower())

    # Fila que contiene los nombres de mes (enero..diciembre).
    fila_meses = None
    for idx in range(len(texto)):
        if "enero" in set(texto.iloc[idx].tolist()):
            fila_meses = idx
            break
    if fila_meses is None:
        print(f"    ⚠️  {anio}: no parece un fichero de ocupación, se ignora")
        return pd.DataFrame()

    col_de_mes = {MESES.index(v) + 1: c for c, v in texto.iloc[fila_meses].items()
                  if v in MESES}
    # Código/Etiqueta/Denominación están en las 3 primeras columnas.
    datos = crudo.iloc[fila_meses + 1:, :]
    datos = datos[datos[0].notna()]
    registros = []
    for _, r in datos.iterrows():
        codigo = str(r[0]).strip()
        if not codigo or codigo.lower() == "nan":
            continue
        for mes, col in col_de_mes.items():
            registros.append({"anio": anio, "mes": mes,
                              "linea": str(r[1]).strip(),
                              "linea_codigo": codigo,
                              "denominacion": str(r[2]).strip(),
                              "pct_calidad_ocupacion": pd.to_numeric(r[col],
                                                                     errors="coerce")})
    return pd.DataFrame(registros)


def descargar_ocupacion(session: requests.Session) -> Path:
    """Descarga los XLS anuales de ocupación (2018-2024) y los normaliza.

    Salida: datos/processed/ocupacion_linea_mes.parquet
    [anio, mes, linea, denominacion, pct_calidad_ocupacion]. El valor es el % de
    viajes que cumplen el criterio de calidad de ocupación (≈100 = holgado;
    bajo = saturación). Se trata el 0 como dato ausente.
    """
    out_dir = DATA_RAW / "ocupacion"
    partes = []
    for anio, suf in OCUPACION_SUFIJOS.items():
        url = OCUPACION_URL.format(suf=suf)
        print(f"[ocupacion] {anio}: descargando...")
        try:
            r = _get(url, session)
        except RuntimeError as e:
            print(f"    ⚠️  saltado {anio}: {e}")
            continue
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / f"ocupacion_{anio}.xls").write_bytes(r.content)
        partes.append(_leer_ocupacion_xls(r.content, anio))

    if not partes:
        raise RuntimeError("No se pudo descargar ningún año de ocupación")
    df = pd.concat(partes, ignore_index=True)
    # 0 = mes sin dato reportado -> ausente
    df.loc[df["pct_calidad_ocupacion"] == 0, "pct_calidad_ocupacion"] = pd.NA
    df = df.dropna(subset=["pct_calidad_ocupacion"])

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    destino = DATA_PROCESSED / "ocupacion_linea_mes.parquet"
    df.to_parquet(destino, index=False)
    print(f"[ocupacion] {len(df):,} filas | {df['linea'].nunique()} líneas | "
          f"años {sorted(df['anio'].unique())}")
    print(f"[ocupacion] guardado en {destino}")
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga de datos abiertos EMT")
    parser.add_argument("--demanda", action="store_true", help="Demanda diaria")
    parser.add_argument("--oferta", action="store_true", help="Oferta de autobuses")
    parser.add_argument("--ocupacion", action="store_true",
                        help="Grado de ocupación (saturación), datos.madrid.es")
    parser.add_argument("--calendario", action="store_true",
                        help="Calendario de operación (tipo de día, clima)")
    parser.add_argument("--all", action="store_true", help="Todos los datasets")
    args = parser.parse_args()

    if not (args.demanda or args.oferta or args.ocupacion
            or args.calendario or args.all):
        parser.error("Indica --demanda, --oferta, --ocupacion, --calendario o --all")

    session = _session()
    if args.demanda or args.all:
        normalizar_demanda(descargar_demanda(session))
    if args.oferta or args.all:
        descargar_oferta(session)
    if args.ocupacion or args.all:
        descargar_ocupacion(session)
    if args.calendario or args.all:
        descargar_calendario(session)


if __name__ == "__main__":
    main()
