"""Funciones puras del contexto: factor de evento, franjas, calendario y haversine.

Nada de red: el calendario de LaLiga se parsea desde muestras incrustadas.
"""
from datetime import date

import pytest

from ingest.contexto import (ASISTENCIA_UMBRAL, FACTOR_EVENTO_EXTRAP, FACTOR_RECINTO,
                            _equipo_corto, _estadio_de, _factor_evento, _franjas_evento,
                            _haversine_m, _parsear_fixtures, _sin_duplicados,
                            _temporadas, _ventana_evento)


def test_factor_evento_por_recinto():
    """Factores MEDIDOS por recinto (notebooks/calibracion_evento.py)."""
    assert _factor_evento(70000, "Santiago Bernabéu") == FACTOR_RECINTO["bernabeu"]
    assert _factor_evento(57000, "Metropolitano") == FACTOR_RECINTO["metropolitano"]
    assert _factor_evento(13500, "Estadio de Vallecas") == 1.0


def test_factor_evento_no_es_monotono_en_el_aforo():
    """El efecto medido DECRECE con el aforo: el Metropolitano (57.000) mueve más
    demanda de bus que el Bernabéu (70.000) porque a él solo llegan cuatro líneas
    periféricas. Una ley creciente en la asistencia invertía estos dos puntos, que
    son los únicos calibrados; este test evita que alguien la reintroduzca.
    """
    bernabeu = _factor_evento(70000, "Santiago Bernabéu")
    metropolitano = _factor_evento(57000, "Metropolitano")
    assert metropolitano > bernabeu


def test_factor_evento_umbral_y_extrapolacion():
    """Bajo el umbral no se aplica nada; sobre él, sin medición, la extrapolación."""
    assert _factor_evento(0) == 1.0
    assert _factor_evento(ASISTENCIA_UMBRAL) == 1.0            # umbral: sin efecto
    assert _factor_evento(3000, "Centro cultural") == 1.0      # actividad municipal
    assert _factor_evento(15500, "Movistar Arena") == FACTOR_EVENTO_EXTRAP
    assert _factor_evento(65000, "Iberdrola Music") == FACTOR_EVENTO_EXTRAP
    # la extrapolación no supera al mayor efecto medido
    assert FACTOR_EVENTO_EXTRAP <= max(FACTOR_RECINTO.values())


def test_franjas_evento_rango_valido():
    for hora in (None, 10, 18, 20, 22):
        fr = _franjas_evento(hora)
        assert fr                                             # nunca vacío
        assert all(0 <= f <= 23 for f in fr)                 # franjas horarias válidas
    assert _franjas_evento(10) == [8, 9, 10]                 # diurno: solo llegada
    assert {18, 19, 20}.issubset(_franjas_evento(20))        # incluye hora y 2 previas


def test_ventana_evento_cruza_medianoche():
    """Un evento nocturno no debe rotularse como si durase el día entero."""
    assert _ventana_evento(_franjas_evento(10)) == "8–10 h"
    # 21:00 → llegada 19-21 y salida 23-01; ordenado empieza en 0, y antes se
    # imprimía "0–23 h" (el bug detectado en la revisión)
    assert _ventana_evento(_franjas_evento(21)) == "19–21 y 23–1 h"
    assert _ventana_evento(_franjas_evento(22)) == "20–22 y 0–2 h"   # madrugada al final
    assert _ventana_evento([]) == ""


FIXTURES_TXT = """= Spain Primera Division 2026/27

# Date       Sun Aug 16 2026 - Sun May 30 2027 (287d)

▪ Matchday 1
  Sun Aug 16 2026
    17:00  Club Atlético de Madrid v Málaga CF
           Real Madrid CF          v Real Sociedad de Fútbol

▪ Matchday 18
  Sat Dec 19
    21:00  Rayo Vallecano de Madrid v Getafe CF
  Sun Jan 3
    18:30  Real Madrid CF          v FC Barcelona           2-1 (1-0)
"""

FIXTURES_JSON = """{"name": "LaLiga", "matches": [
  {"date": "2025-09-14", "time": "16:15", "team1": "Real Madrid CF", "team2": "CA Osasuna"},
  {"date": "2025-09-21", "team1": "Club Atlético de Madrid", "team2": "Sevilla FC"}]}"""


def test_parsear_fixtures_texto():
    """Formato de texto: hora heredada, año implícito y resultado al final."""
    p = _parsear_fixtures(FIXTURES_TXT)
    assert (date(2026, 8, 16), 17, "Club Atlético de Madrid", "Málaga CF") in p
    # la segunda línea de la jornada no repite la hora: la hereda
    assert (date(2026, 8, 16), 17, "Real Madrid CF", "Real Sociedad de Fútbol") in p
    # diciembre sin año explícito sigue en 2026; enero ya es 2027
    assert (date(2026, 12, 19), 21, "Rayo Vallecano de Madrid", "Getafe CF") in p
    assert (date(2027, 1, 3), 18, "Real Madrid CF", "FC Barcelona") in p
    assert len(p) == 4


def test_parsear_fixtures_json():
    p = _parsear_fixtures(FIXTURES_JSON)
    assert (date(2025, 9, 14), 16, "Real Madrid CF", "CA Osasuna") in p
    assert p[1][1] == 21                        # sin hora publicada → la de defecto


def test_temporadas_cubren_el_rango():
    assert "2026-27" in _temporadas(date(2026, 8, 13), date(2026, 8, 20))
    assert "2025-26" in _temporadas(date(2026, 5, 1), date(2026, 5, 8))


def test_estadio_y_equipo():
    assert _estadio_de("Estadio Santiago Bernabéu") == "bernabeu"
    assert _estadio_de("Riyadh Air Metropolitano") == "metropolitano"
    assert _estadio_de("Movistar Arena") is None
    assert _equipo_corto("Club Atlético de Madrid") == "Atlético de Madrid"
    assert _equipo_corto("Real Madrid CF") == "Real Madrid"


def test_sin_duplicados_prefiere_laliga():
    """El mismo partido por dos fuentes cuenta una vez, y gana el de LaLiga."""
    evs = [
        {"fecha": "2026-08-16", "fuente": "LaLiga (openfootball)",
         "estadio": "bernabeu", "lugar": "Santiago Bernabéu"},
        {"fecha": "2026-08-16", "fuente": "Ticketmaster",
         "lugar": "Estadio Santiago Bernabéu"},
        {"fecha": "2026-08-16", "fuente": "Ticketmaster", "lugar": "Movistar Arena"},
        {"fecha": "2026-08-17", "fuente": "Ticketmaster",
         "lugar": "Estadio Santiago Bernabéu"},
    ]
    out = _sin_duplicados(evs)
    assert len(out) == 3
    assert out[0]["fuente"].startswith("LaLiga")
    assert [e["lugar"] for e in out[1:]] == ["Movistar Arena", "Estadio Santiago Bernabéu"]


def test_haversine_casos_conocidos():
    assert float(_haversine_m(40.4, -3.7, 40.4, -3.7)) == pytest.approx(0.0, abs=1e-6)
    d = float(_haversine_m(40.0, -3.7, 41.0, -3.7))          # 1º de latitud ~ 111 km
    assert 110_000 < d < 112_000
