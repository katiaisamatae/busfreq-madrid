"""Carga de configuración y secretos desde variables de entorno (.env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # raíz del repo (TFM/)
DATA_RAW = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "datos/raw")
DATA_PROCESSED = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "datos/processed")

# Credenciales EMT
EMT_EMAIL = os.getenv("EMT_EMAIL")
EMT_PASSWORD = os.getenv("EMT_PASSWORD")
EMT_CLIENT_ID = os.getenv("EMT_CLIENT_ID")
EMT_PASSKEY = os.getenv("EMT_PASSKEY")

# Credenciales AEMET
AEMET_API_KEY = os.getenv("AEMET_API_KEY")

# Ticketmaster Discovery (opcional — eventos grandes; gratis en developer.ticketmaster.com)
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY")


def require(name: str, value: str | None) -> str:
    """Devuelve el valor o lanza un error claro si falta la variable."""
    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno '{name}'. "
            f"Copia .env.example a .env y rellena tus claves."
        )
    return value
