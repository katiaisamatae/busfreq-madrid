# -*- coding: utf-8 -*-
"""Contexto automático para la predicción — clima y eventos sin entrada manual.

Consulta las fuentes externas, georreferencia los eventos contra las paradas
GTFS de las líneas piloto y devuelve factores de demanda listos para inyectar
al modelo/optimizador:

  * AEMET OpenData   — previsión diaria municipio 28079 (AEMET_API_KEY)
  * Open-Meteo       — previsión diaria de contraste (sin clave)
  * datos.madrid.es  — agenda de actividades y eventos (sin clave)
  * openfootball     — calendario de LaLiga: partidos en el Bernabéu, el
                       Metropolitano y Vallecas (sin clave)
  * Ticketmaster     — grandes conciertos/deporte (TICKETMASTER_API_KEY, opcional)

Uso directo (prueba):
    python -m ingest.contexto            # imprime el contexto de 7 días
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import unicodedata
import zipfile
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

if __package__ in (None, ""):  # permite `python ingest/contexto.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import (AEMET_API_KEY, DATA_PROCESSED, DATA_RAW,
                           TICKETMASTER_API_KEY)

log = logging.getLogger("busfreq.contexto")

LAT_MADRID, LON_MADRID = 40.4168, -3.7038
MUNICIPIO = "28079"
RADIO_M = 600            # paradas a menos de esta distancia del recinto
TTL_S = 6 * 3600         # caché en memoria del contexto (6 h)
TIMEOUT = 20

# Umbral y efecto de lluvia. El cociente directo entre la media de días lluviosos
# y la de días secos daba 1,06, pero era un ARTEFACTO DE ESTACIONALIDAD: en Madrid
# llueve en los meses de demanda alta y no llueve en julio y agosto, que son el
# valle del año, así que el grupo «seco» arrastraba el verano. Controlando por
# mes x año (OLS sobre log de viajeros, errores agrupados por fecha) el efecto se
# desvanece: -2,4 % con IC 95 % [-4,9, +0,1], no significativo, y el placebo sale
# limpio. No se aplica factor por lluvia (ver memoria, calibración del contexto).
PROB_LLUVIA_MIN = 60
FACTOR_LLUVIA = 1.0

# Factor de evento CALIBRADO con el histórico (notebooks/calibracion_evento.py):
# diferencias en diferencias sobre 2022-2026 con los partidos de LaLiga en el
# Bernabéu, el Metropolitano y Vallecas. Efecto medido sobre la demanda DIARIA
# de las líneas cercanas: +6,1 % (Bernabéu, 70.000 espectadores), +10,8 %
# (Metropolitano, 57.000) y no distinguible de cero en Vallecas (13.500). Al
# repartirlo entre las franjas del evento y descontar el placebo salen los
# factores de abajo. Antes esto era `1 + asistencia/75000` (un partido daba
# +80 % de demanda), una suposición sin medir.
#
# NO se usa una ley en la asistencia. El efecto medido DECRECE con el aforo: el
# Metropolitano (57.000, cuatro líneas periféricas) mueve más demanda de bus que
# el Bernabéu (70.000, seis corredores de la Castellana), porque lo que manda es
# la red local y no el tamaño del recinto. Ajustar una recta creciente sobre los
# dos únicos puntos calibrados invertía su orden y erraba ambos (el Bernabéu un
# 76 % por exceso, el Metropolitano un 42 % por defecto).
ASISTENCIA_UMBRAL = 13_500     # por debajo, el efecto no es detectable → 1,0
FACTOR_RECINTO = {         # factor por franja MEDIDO en cada recinto calibrado
    "bernabeu": 1.193,
    "metropolitano": 1.453,
    "vallecas": 1.0,      # 13.500 espectadores: efecto no significativo
}
FACTOR_EVENTO_EXTRAP = 1.19    # recinto sin medición propia sobre el umbral:
                               # el menor efecto significativo medido. Es una
                               # extrapolación declarada, no una medición.

# Estadios de LaLiga dentro de la red URBANA de la EMT, con los que se calibró
# el factor. Quedan fuera Getafe, Leganés y Alcalá: los sirven autobuses
# interurbanos del Consorcio, no la EMT. La asistencia es la media estimada del
# recinto (aforo x ocupación habitual); no hay fuente abierta con el dato por
# partido, así que todos los partidos de un mismo estadio pesan igual.
ESTADIOS_LALIGA = {
    "bernabeu": {"equipo": "Real Madrid", "nombre": "Santiago Bernabéu",
                 "lat": 40.4531, "lon": -3.6883, "asistencia": 70_000},
    "metropolitano": {"equipo": "Atlético de Madrid", "nombre": "Metropolitano",
                      "lat": 40.4362, "lon": -3.5995, "asistencia": 57_000},
    "vallecas": {"equipo": "Rayo Vallecano", "nombre": "Estadio de Vallecas",
                 "lat": 40.3919, "lon": -3.6588, "asistencia": 13_500},
}

URL_LALIGA_JSON = ("https://raw.githubusercontent.com/openfootball/football.json/"
                   "master/{temporada}/es.1.json")
URL_LALIGA_TXT = ("https://raw.githubusercontent.com/openfootball/espana/"
                  "master/{temporada}/1-liga.txt")
TTL_FIXTURES_D = 7            # refresca el calendario cada semana (cambian horarios)
HORA_PARTIDO_DEFECTO = 21     # si la fuente aún no publica el horario

# Aforo estimado por recinto (la Discovery API no siempre lo publica). Los tres
# estadios usan la asistencia CALIBRADA de ESTADIOS_LALIGA, no el aforo máximo,
# para que un partido dé el mismo factor lo detecte LaLiga o Ticketmaster. Los
# pabellones son aforo máximo: no entran en la calibración (el mayor recinto
# donde el efecto era medible es Vallecas, y ninguno lo supera).
AFOROS = {
    "metropolitano": ESTADIOS_LALIGA["metropolitano"]["asistencia"],
    "bernabeu": ESTADIOS_LALIGA["bernabeu"]["asistencia"],
    "bernabéu": ESTADIOS_LALIGA["bernabeu"]["asistencia"],
    "vallecas": ESTADIOS_LALIGA["vallecas"]["asistencia"],
    "wizink": 15500, "movistar arena": 15500, "vistalegre": 14000,
    "caja magica": 12500, "caja mágica": 12500, "vallehermoso": 10000,
    "ifema": 25000, "la riviera": 2500,
}
AFORO_DEFECTO_TM = 5000       # evento Ticketmaster sin aforo conocido
AFORO_DEFECTO_AGENDA = 3000   # actividad de la agenda municipal


# --------------------------------------------------------------------------- #
# Paradas GTFS de las líneas piloto (se construye una vez y se cachea a disco)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _paradas_piloto() -> pd.DataFrame:
    cache = DATA_PROCESSED / "paradas_linea.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    piloto = set(pd.read_csv(DATA_PROCESSED / "lineas_piloto.csv",
                             dtype={"linea": str})["linea"])
    with zipfile.ZipFile(DATA_RAW / "gtfs_emt.zip") as z:
        routes = pd.read_csv(z.open("routes.txt"), dtype=str)
        routes = routes[routes["route_short_name"].isin(piloto)]
        trips = pd.read_csv(z.open("trips.txt"), dtype=str)
        trips = trips[trips["route_id"].isin(set(routes["route_id"]))]
        sub = ["route_id", "direction_id"] if "direction_id" in trips.columns else ["route_id"]
        trips = trips.drop_duplicates(sub)[["route_id", "trip_id"]]
        st = pd.read_csv(z.open("stop_times.txt"), dtype=str,
                         usecols=["trip_id", "stop_id"])
        st = st[st["trip_id"].isin(set(trips["trip_id"]))]
        stops = pd.read_csv(z.open("stops.txt"), dtype={"stop_id": str})
    df = (st.merge(trips, on="trip_id")
            .merge(routes[["route_id", "route_short_name"]], on="route_id")
            .merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id")
            .rename(columns={"route_short_name": "linea"}))
    df = (df[["linea", "stop_lat", "stop_lon"]]
          .astype({"stop_lat": float, "stop_lon": float}).drop_duplicates())
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def _haversine_m(lat1, lon1, lat2, lon2):
    """Distancia haversine en metros (acepta escalares o arrays numpy)."""
    import numpy as np
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * np.arcsin(np.sqrt(a))


def _lineas_cercanas(lat: float, lon: float, radio_m: float = RADIO_M) -> list[str]:
    """Líneas piloto con alguna parada a < radio_m del punto (haversine)."""
    p = _paradas_piloto()
    d = _haversine_m(lat, lon, p["stop_lat"].to_numpy(), p["stop_lon"].to_numpy())
    return sorted(p.loc[d < radio_m, "linea"].unique().tolist())


# --------------------------------------------------------------------------- #
# Clima
# --------------------------------------------------------------------------- #
def _clima_aemet() -> dict[str, dict]:
    """Previsión diaria AEMET (2 pasos). {fecha_iso: {tmax, prob_prec}}"""
    base = "https://opendata.aemet.es/opendata/api"
    r = requests.get(f"{base}/prediccion/especifica/municipio/diaria/{MUNICIPIO}",
                     params={"api_key": AEMET_API_KEY}, timeout=TIMEOUT).json()
    datos = requests.get(r["datos"], timeout=TIMEOUT).json()
    out = {}
    for d in datos[0].get("prediccion", {}).get("dia", []):
        prob = [p.get("value") for p in d.get("probPrecipitacion", [])
                if p.get("value") is not None]
        out[str(pd.to_datetime(d["fecha"]).date())] = {
            "tmax": d.get("temperatura", {}).get("maxima"),
            "prob_prec": max(prob) if prob else None,
        }
    return out


def _clima_openmeteo(n_dias: int) -> dict[str, dict]:
    """Previsión diaria Open-Meteo (sin clave). {fecha_iso: {tmax, prob_prec, prec_mm}}"""
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": LAT_MADRID, "longitude": LON_MADRID,
                "daily": "temperature_2m_max,precipitation_sum,precipitation_probability_max",
                "forecast_days": min(n_dias, 16), "timezone": "Europe/Madrid"},
        timeout=TIMEOUT).json()["daily"]
    return {f: {"tmax": t, "prec_mm": p, "prob_prec": pr}
            for f, t, p, pr in zip(r["time"], r["temperature_2m_max"],
                                   r["precipitation_sum"],
                                   r["precipitation_probability_max"])}


# --------------------------------------------------------------------------- #
# Eventos
# --------------------------------------------------------------------------- #
def _aforo(nombre_lugar: str, defecto: int) -> int:
    n = (nombre_lugar or "").lower()
    for clave, aforo in AFOROS.items():
        if clave in n:
            return aforo
    return defecto


def _factor_evento(asistencia: int, lugar: str = "") -> float:
    """Factor de demanda en las franjas del evento para las líneas cercanas.

    Se aplica la medición del recinto cuando existe, y una extrapolación
    conservadora cuando no. No hay ley en la asistencia: con tres recintos
    calibrados y un efecto que decrece con el aforo, una recta creciente
    invierte los únicos puntos que hay (véase el comentario de FACTOR_RECINTO).
    """
    clave = _estadio_de(lugar) if lugar else None
    if clave in FACTOR_RECINTO:
        return FACTOR_RECINTO[clave]
    if asistencia <= ASISTENCIA_UMBRAL:
        return 1.0
    return FACTOR_EVENTO_EXTRAP


def _franjas_evento(hora: int | None) -> list[int]:
    """Franjas afectadas: llegada (2 h antes) y, si es nocturno, la salida."""
    if hora is None:
        return list(range(11, 21))          # actividad de día completo
    llegada = [h % 24 for h in range(max(hora - 2, 0), hora + 1)]
    salida = [h % 24 for h in range(hora + 2, hora + 5)] if hora >= 18 else []
    return sorted(set(llegada + salida))


def _ventana_evento(franjas: list[int]) -> str:
    """Etiqueta legible de las franjas afectadas: '19–21 y 23–1 h'.

    Los eventos nocturnos cruzan la medianoche, así que la lista ordenada
    empieza en 0 y acaba en 23: imprimir primera-última daba "0–23 h" (parecía
    el día entero). Aquí se agrupan los tramos contiguos en módulo 24.
    """
    s = set(franjas)
    if not s:
        return ""
    tramos = []
    # el día de servicio empieza a las 5:00, así que las franjas de madrugada
    # van al final (un evento a las 22 h afecta a "20–22 y 0–2 h", en ese orden)
    for ini in sorted((f for f in s if (f - 1) % 24 not in s),
                      key=lambda f: (f - 5) % 24):
        fin = ini
        while (fin + 1) % 24 in s:
            fin = (fin + 1) % 24
        tramos.append(f"{ini}–{fin}" if fin != ini else f"{ini}")
    return " y ".join(tramos) + " h"


def _norm(s: str) -> str:
    """Minúsculas sin acentos, para comparar nombres de equipos y recintos."""
    return "".join(c for c in unicodedata.normalize("NFKD", (s or "").lower())
                   if not unicodedata.combining(c))


def _estadio_de(nombre_lugar: str) -> str | None:
    """Identifica el estadio a partir del nombre del recinto ('bernabeu', ...)."""
    n = _norm(nombre_lugar)
    for clave in ESTADIOS_LALIGA:
        if clave in n:
            return clave
    return None


# --------------------------------------------------------------------------- #
# Calendario de LaLiga (openfootball, dominio público, sin clave)
# --------------------------------------------------------------------------- #
_MESES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
          "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_RE_DIA = re.compile(r"^\s*(?:mon|tue|wed|thu|fri|sat|sun)\s+([a-z]{3})\s+(\d{1,2})"
                     r"(?:\s+(\d{4}))?\s*$", re.I)
_RE_PARTIDO = re.compile(r"^\s*(?:(\d{1,2}):(\d{2}))?\s+(.+?)\s+(?:v|vs\.?)\s+(.+?)\s*$",
                         re.I)
_RE_RESULTADO = re.compile(r"\s+\d+-\d+(?:\s+\(\d+-\d+\))?\s*$")


def _temporadas(desde: date, hasta: date) -> list[str]:
    """Temporadas de LaLiga que cubren el rango ('2026-27'). Julio parte aguas."""
    out = []
    for a in {desde.year, hasta.year}:
        for ini in (a - 1, a):
            t = f"{ini}-{str(ini + 1)[2:]}"
            if t not in out:
                out.append(t)
    return sorted(out)


def _descargar_fixtures(temporada: str) -> str | None:
    """Calendario de una temporada, cacheado en disco (TTL_FIXTURES_D días).

    Intenta el JSON de football.json y, si esa temporada aún no está publicada
    ahí, el texto de openfootball/espana (que suele adelantarse). Devuelve el
    contenido crudo o None si ninguna fuente responde.
    """
    cache = DATA_RAW / "partidos" / f"laliga_{temporada}"
    if cache.exists():
        edad = time.time() - cache.stat().st_mtime
        if edad < TTL_FIXTURES_D * 86400:
            return cache.read_text(encoding="utf-8")
    for url in (URL_LALIGA_JSON, URL_LALIGA_TXT):
        try:
            r = requests.get(url.format(temporada=temporada), timeout=TIMEOUT)
        except requests.RequestException:
            continue
        if r.status_code == 200 and r.text.strip():
            r.encoding = "utf-8"
            try:                       # el disco puede ser de solo lectura (Azure)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(r.text, encoding="utf-8")
            except OSError:
                log.info("no se pudo cachear el calendario de %s", temporada)
            return r.text
    return cache.read_text(encoding="utf-8") if cache.exists() else None


def _parsear_fixtures(crudo: str) -> list[tuple[date, int, str, str]]:
    """(fecha, hora, local, visitante) desde el JSON o el texto de openfootball."""
    crudo = crudo.lstrip()
    if crudo.startswith("{"):
        partidos = []
        for m in json.loads(crudo).get("matches", []):
            try:
                f = datetime.strptime(m["date"], "%Y-%m-%d").date()
            except (KeyError, ValueError):
                continue
            hora = int((m.get("time") or "")[:2] or HORA_PARTIDO_DEFECTO)
            partidos.append((f, hora, m.get("team1", ""), m.get("team2", "")))
        return partidos

    # Formato de texto: la fecha encabeza el bloque y la hora se hereda de la
    # última indicada (las líneas siguientes de la misma hora la omiten).
    partidos, f_actual, hora_actual, anio, mes_prev = [], None, None, None, None
    for linea in crudo.splitlines():
        if not linea.strip() or linea.lstrip().startswith(("=", "#", "▪")):
            continue
        m = _RE_DIA.match(linea)
        if m:
            mes = _MESES.get(m.group(1).lower())
            if mes is None:
                continue
            if m.group(3):
                anio = int(m.group(3))
            elif anio and mes_prev and mes < mes_prev:
                anio += 1                      # cambio de año dentro de la temporada
            if anio is None:
                continue
            mes_prev = mes
            try:
                f_actual = date(anio, mes, int(m.group(2)))
            except ValueError:
                f_actual = None
            hora_actual = None
            continue
        m = _RE_PARTIDO.match(_RE_RESULTADO.sub("", linea))
        if m and f_actual:
            if m.group(1):
                hora_actual = int(m.group(1))
            partidos.append((f_actual, hora_actual if hora_actual is not None
                             else HORA_PARTIDO_DEFECTO,
                             m.group(3).strip(), m.group(4).strip()))
    return partidos


def partidos_laliga(desde: date, hasta: date) -> list[tuple[date, int, str, str]]:
    """Partidos de LaLiga entre dos fechas (cualquier estadio)."""
    out = []
    for t in _temporadas(desde, hasta):
        crudo = _descargar_fixtures(t)
        if not crudo:
            continue
        out += [p for p in _parsear_fixtures(crudo) if desde <= p[0] <= hasta]
    return sorted(set(out))


def _equipo_corto(nombre: str) -> str:
    """'Club Atlético de Madrid' → 'Atlético de Madrid'; 'Real Madrid CF' → 'Real Madrid'."""
    n = nombre.strip()
    for p in ("Club ", "Real Club ", "RC ", "RCD ", "CA ", "CD ", "UD ", "SD ", "CF ", "FC "):
        if n.startswith(p):
            n = n[len(p):]
    for s in (" CF", " FC", " de Fútbol", " Balompié"):
        if n.endswith(s):
            n = n[: -len(s)]
    return n


def _eventos_laliga(hoy: date, fin: date) -> list[dict]:
    """Partidos en los estadios de Madrid servidos por la EMT.

    Se conservan aunque NO afecten a ninguna línea piloto (`lineas` vacía): el
    Metropolitano, por ejemplo, no lo sirve ninguna de las 12, y verlo es
    información honesta sobre el alcance del piloto.
    """
    equipos = {clave: _norm(cfg["equipo"]) for clave, cfg in ESTADIOS_LALIGA.items()}
    todos = partidos_laliga(hoy, fin)

    # LaLiga confirma los horarios con dos o tres semanas de antelación; hasta
    # entonces la fuente publica una hora única para toda la jornada. Si un día
    # tiene muchos partidos y todos a la misma hora, el horario es provisional y
    # la ventana de franjas puede estar desplazada: hay que decirlo en la UI.
    horas_dia: dict[date, list[int]] = {}
    for f, hora, *_ in todos:
        horas_dia.setdefault(f, []).append(hora)
    provisional = {f: len(hs) >= 4 and len(set(hs)) == 1 for f, hs in horas_dia.items()}

    out = []
    for f, hora, local, visitante in todos:
        local_n = _norm(local)
        clave = next((k for k, eq in equipos.items() if eq in local_n), None)
        if clave is None:
            continue
        cfg = ESTADIOS_LALIGA[clave]
        franjas = _franjas_evento(hora)
        factor = _factor_evento(cfg["asistencia"], cfg["nombre"])
        out.append({
            "fecha": str(f), "hora": hora,
            "nombre": f"LaLiga: {_equipo_corto(local)} – {_equipo_corto(visitante)}",
            "lugar": cfg["nombre"],
            "asistencia": cfg["asistencia"],
            "fuente": "LaLiga (openfootball)",
            "estadio": clave,
            "lineas": _lineas_cercanas(cfg["lat"], cfg["lon"]),
            "factor": factor,
            "efecto_pct": round((factor - 1) * 100),
            "franjas": franjas,
            "ventana": _ventana_evento(franjas),
            "hora_provisional": provisional.get(f, False),
        })
    return out


def _sin_duplicados(eventos: list[dict]) -> list[dict]:
    """Un mismo partido puede llegar por LaLiga y por Ticketmaster: deja uno.

    Se prefiere el de LaLiga, que trae la asistencia con la que se calibró el
    factor en lugar del aforo que publique el proveedor de entradas.
    """
    estadios_laliga = {(e["fecha"], e.get("estadio")) for e in eventos
                       if e["fuente"].startswith("LaLiga")}
    out = []
    for e in eventos:
        if not e["fuente"].startswith("LaLiga"):
            clave = (e["fecha"], _estadio_de(e["lugar"]))
            if clave[1] and clave in estadios_laliga:
                continue
        out.append(e)
    return out


def _eventos_agenda(hoy: date, fin: date) -> list[dict]:
    """Agenda de actividades y eventos — datos.madrid.es (JSON, sin clave)."""
    url = "https://datos.madrid.es/egob/catalogo/206974-0-agenda-eventos-culturales-100.json"
    j = requests.get(url, timeout=TIMEOUT).json()
    out = []
    for e in j.get("@graph", []):
        try:
            ini = pd.to_datetime(e.get("dtstart"))
            fin_e = pd.to_datetime(e.get("dtend")) if e.get("dtend") else ini
        except (ValueError, TypeError):
            continue
        if pd.isna(ini):
            continue
        # descarta exposiciones/ciclos largos: solo citas puntuales (<= 3 días)
        if (fin_e - ini).days > 3:
            continue
        if not (hoy <= ini.date() <= fin):
            continue
        loc = e.get("location") or {}
        lat, lon = loc.get("latitude"), loc.get("longitude")
        if not lat or not lon:
            continue
        lineas = _lineas_cercanas(float(lat), float(lon))
        if not lineas:
            continue
        lugar = e.get("event-location") or e.get("title", "")
        asistencia = _aforo(lugar, AFORO_DEFECTO_AGENDA)
        hora = None if ini.hour == 0 else ini.hour
        franjas = _franjas_evento(hora)
        factor = _factor_evento(asistencia, lugar)
        out.append({
            "fecha": str(ini.date()),
            "hora": hora,
            "nombre": (e.get("title") or "Actividad municipal").strip()[:90],
            "lugar": str(lugar).strip()[:70],
            "asistencia": asistencia,
            "fuente": "datos.madrid.es",
            "lineas": lineas,
            "factor": factor,
            "efecto_pct": round((factor - 1) * 100),
            "franjas": franjas,
            "ventana": _ventana_evento(franjas),
        })
    return out


def _eventos_ticketmaster(hoy: date, fin: date) -> list[dict]:
    """Discovery API — grandes conciertos y deporte en Madrid (requiere clave)."""
    r = requests.get(
        "https://app.ticketmaster.com/discovery/v2/events.json",
        params={"apikey": TICKETMASTER_API_KEY, "city": "Madrid", "countryCode": "ES",
                "startDateTime": f"{hoy}T00:00:00Z", "endDateTime": f"{fin}T23:59:59Z",
                "size": 100, "sort": "date,asc"},
        timeout=TIMEOUT).json()
    out = []
    for e in (r.get("_embedded", {}) or {}).get("events", []):
        venues = (e.get("_embedded", {}) or {}).get("venues", [])
        if not venues:
            continue
        v = venues[0]
        loc = v.get("location") or {}
        if not loc.get("latitude"):
            continue
        lineas = _lineas_cercanas(float(loc["latitude"]), float(loc["longitude"]))
        if not lineas:
            continue
        start = e.get("dates", {}).get("start", {})
        f, hhmm = start.get("localDate"), start.get("localTime", "")
        if not f:
            continue
        hora = int(hhmm[:2]) if hhmm else 20
        asistencia = _aforo(v.get("name", ""), AFORO_DEFECTO_TM)
        franjas = _franjas_evento(hora)
        factor = _factor_evento(asistencia, v.get("name", ""))
        out.append({
            "fecha": f, "hora": hora,
            "nombre": (e.get("name") or "Evento").strip()[:90],
            "lugar": (v.get("name") or "").strip()[:70],
            "asistencia": asistencia,
            "fuente": "Ticketmaster",
            "lineas": lineas,
            "factor": factor,
            "efecto_pct": round((factor - 1) * 100),
            "franjas": franjas,
            "ventana": _ventana_evento(franjas),
        })
    return out


# --------------------------------------------------------------------------- #
# Contexto agregado por día
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _festivos():
    import holidays
    return holidays.country_holidays("ES", subdiv="MD", years=range(2024, 2031))


def tipo_dia(f: date) -> str:
    if f in _festivos() or f.weekday() == 6:
        return "FE"
    return "SA" if f.weekday() == 5 else "LA"


_cache: dict = {"ts": 0.0, "dias": 0, "data": None}


def contexto_dias(n_dias: int = 7, forzar: bool = False) -> dict:
    """Contexto de los próximos n_dias: clima, eventos y factores por línea."""
    if (not forzar and _cache["data"] is not None
            and _cache["dias"] == n_dias and time.time() - _cache["ts"] < TTL_S):
        return _cache["data"]

    hoy = date.today()
    fin = hoy + timedelta(days=n_dias - 1)
    fuentes: dict[str, str] = {}

    clima_om: dict[str, dict] = {}
    try:
        clima_om = _clima_openmeteo(n_dias)
        fuentes["open_meteo"] = "ok"
    except Exception as e:  # noqa: BLE001 — la fuente puede caerse sin tirar la API
        fuentes["open_meteo"] = f"error: {type(e).__name__}"

    clima_ae: dict[str, dict] = {}
    if AEMET_API_KEY:
        try:
            clima_ae = _clima_aemet()
            fuentes["aemet"] = "ok"
        except Exception as e:  # noqa: BLE001
            fuentes["aemet"] = f"error: {type(e).__name__}"
    else:
        fuentes["aemet"] = "sin clave"

    eventos: list[dict] = []
    try:
        eventos += _eventos_agenda(hoy, fin)
        fuentes["agenda_madrid"] = "ok"
    except Exception as e:  # noqa: BLE001
        fuentes["agenda_madrid"] = f"error: {type(e).__name__}"
    try:
        partidos = _eventos_laliga(hoy, fin)
        eventos += partidos
        # sin partidos no es un fallo: puede no haber jornada en el rango (parón
        # de selecciones, verano) o no estar publicada aún la temporada
        fuentes["laliga"] = "ok" if partidos else "sin partidos en el rango"
    except Exception as e:  # noqa: BLE001
        fuentes["laliga"] = f"error: {type(e).__name__}"
    if TICKETMASTER_API_KEY:
        try:
            eventos += _eventos_ticketmaster(hoy, fin)
            fuentes["ticketmaster"] = "ok"
        except Exception as e:  # noqa: BLE001
            fuentes["ticketmaster"] = f"error: {type(e).__name__}"
    else:
        fuentes["ticketmaster"] = "sin clave"
    eventos = _sin_duplicados(eventos)

    dias = []
    for i in range(n_dias):
        f = hoy + timedelta(days=i)
        iso = str(f)
        om, ae = clima_om.get(iso, {}), clima_ae.get(iso, {})
        prob = ae.get("prob_prec") if ae.get("prob_prec") is not None else om.get("prob_prec")
        prec = om.get("prec_mm")
        lluvia = bool((prob or 0) >= PROB_LLUVIA_MIN and (prec is None or prec >= 0.5))
        evs = sorted([e for e in eventos if e["fecha"] == iso],
                     key=lambda e: -e["asistencia"])
        dias.append({
            "fecha": iso,
            "dia_semana": f.weekday(),
            "tipo_dia": tipo_dia(f),
            "clima": {
                "tmax": ae.get("tmax") if ae.get("tmax") is not None else om.get("tmax"),
                "prob_prec": prob, "prec_mm": prec, "lluvia": lluvia,
                "factor_red": FACTOR_LLUVIA if lluvia else 1.0,
                "fuentes": [s for s, c in (("AEMET", clima_ae), ("Open-Meteo", clima_om))
                            if iso in c],
            },
            "eventos": evs,
        })

    data = {"generado": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fuentes": fuentes, "dias": dias}
    _cache.update(ts=time.time(), dias=n_dias, data=data)
    return data


if __name__ == "__main__":
    ctx = contexto_dias()
    print(json.dumps(ctx, indent=2, ensure_ascii=False))
