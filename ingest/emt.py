"""Ingesta de validaciones históricas desde la API de EMT Madrid.

Fase 1 — Datos y análisis exploratorio.

PASO CRÍTICO DEL TFM: verificar que la API expone validaciones históricas por
línea/parada/franja con al menos 12 meses de profundidad. De esto depende todo
el alcance del proyecto (ver CLAUDE.md → "data collection is a bottleneck").

Flujo previsto:
1. Login en la API EMT para obtener el token de sesión.
2. Descarga paginada de validaciones por rango de fechas y línea.
3. Cacheo del crudo en data/raw/ (respetar rate limits de la API).
4. Normalización a un esquema tabular: [line, stop, datetime, validations].

Uso:
    python ingest/emt.py --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse

from ingest.config import DATA_RAW


def login() -> str:
    """Autentica contra la API EMT y devuelve el token de sesión."""
    raise NotImplementedError("Pendiente: implementar login API EMT")


def fetch_validations(start: str, end: str) -> None:
    """Descarga validaciones en [start, end] y las cachea en data/raw/."""
    raise NotImplementedError("Pendiente: implementar descarga de validaciones")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de validaciones EMT Madrid")
    parser.add_argument("--start", required=True, help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Fecha fin YYYY-MM-DD")
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"[ingest.emt] Descargando validaciones {args.start} → {args.end}")
    fetch_validations(args.start, args.end)


if __name__ == "__main__":
    main()
