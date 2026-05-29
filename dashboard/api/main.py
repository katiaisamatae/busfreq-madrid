"""API FastAPI del dashboard — Módulo 3.

Fase 4 del plan de trabajo. Sirve predicciones de demanda, resultados del
optimizador y comparativas de escenarios al frontend React.

Uso (desarrollo):
    uvicorn dashboard.api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="BusFreq Madrid API",
    description="Predicción de demanda y optimización de frecuencias de la EMT Madrid",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """Comprobación básica de que la API está viva."""
    return {"status": "ok"}


# TODO Fase 4 — endpoints previstos:
#   GET  /lines                      -> líneas piloto disponibles
#   GET  /forecast/{line}            -> demanda prevista por franja
#   POST /optimize                   -> frecuencias recomendadas para un escenario
#   GET  /compare/{line}             -> frecuencia actual vs. recomendada + espera
