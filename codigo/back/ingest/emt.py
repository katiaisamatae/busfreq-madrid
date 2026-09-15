"""Ingesta de datos de demanda desde la API / portal de datos de EMT Madrid.

Fase 1 — Datos y análisis exploratorio.

PASO CRÍTICO DEL TFM: verificar dónde y con qué granularidad existen los datos
históricos de demanda. Resultado de la sonda `scripts/probe_emt_validations.py`:

  * API tiempo real (openapi.emtmadrid.es): líneas, paradas, llegadas y
    OCUPACIÓN EN TIEMPO REAL. NO expone validaciones históricas.
  * Portal datos abiertos (datos.emtmadrid.es): dataset "Demanda diaria
    Viajeros Autobús" -> granularidad DÍA x LÍNEA, histórico (>12 meses),
    CSV con actualización mensual. NO hay demanda por parada ni por franja.

Por tanto, la demanda histórica se descarga del portal de datos abiertos
(ver `ingest/emt_opendata.py`, pendiente), no de esta API en tiempo real.
Este módulo implementa la autenticación y utilidades de la API en tiempo real.

Uso:
    python ingest/emt.py            # smoke test: login + llamada autenticada
"""
from __future__ import annotations

import argparse

import requests

from ingest.config import (
    EMT_CLIENT_ID,
    EMT_EMAIL,
    EMT_PASSKEY,
    EMT_PASSWORD,
    require,
)

BASE_URL = "https://openapi.emtmadrid.es"
LOGIN_URL = f"{BASE_URL}/v1/mobilitylabs/user/login/"


def _code(r: requests.Response) -> str:
    """Extrae el campo 'code' de la respuesta JSON de EMT (o vacío)."""
    try:
        return str(r.json().get("code", ""))
    except ValueError:
        return ""


def _extract_token(r: requests.Response) -> str | None:
    """Devuelve el accessToken si la respuesta es un login correcto."""
    try:
        data = r.json().get("data") or []
    except ValueError:
        return None
    if data and data[0].get("accessToken"):
        return data[0]["accessToken"]
    return None


def login() -> str:
    """Autentica contra la API EMT y devuelve el accessToken de sesión.

    Estrategia: intenta primero X-ClientId + passKey (amplía la cuota diaria);
    si no están definidos o el sistema los rechaza, cae al login genérico por
    email + password (20.000 llamadas/día, suficiente para el piloto).
    """
    # 1) Login por aplicación (mayor cuota), si hay credenciales de app.
    if EMT_CLIENT_ID and EMT_PASSKEY:
        r = requests.get(
            LOGIN_URL,
            headers={"X-ClientId": EMT_CLIENT_ID, "passKey": EMT_PASSKEY},
            timeout=30,
        )
        token = _extract_token(r)
        if token:
            return token
        print(
            f"[emt.login] clientId/passKey rechazado (code={_code(r)}). "
            f"Cayendo al login genérico por email/password."
        )

    # 2) Login genérico por email + password.
    email = require("EMT_EMAIL", EMT_EMAIL)
    password = require("EMT_PASSWORD", EMT_PASSWORD)
    r = requests.get(
        LOGIN_URL,
        headers={"email": email, "password": password},
        timeout=30,
    )
    token = _extract_token(r)
    if not token:
        raise RuntimeError(
            f"Login EMT fallido (HTTP {r.status_code}, code={_code(r)}): "
            f"{r.text[:300]}"
        )
    return token


def authed_get(path: str, token: str, **kwargs) -> requests.Response:
    """GET a un endpoint de la API con el accessToken en la cabecera.

    `path` puede ser una ruta relativa ('/v1/...') o una URL absoluta.
    """
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    headers = {"accessToken": token}
    headers.update(kwargs.pop("headers", {}))
    return requests.get(url, headers=headers, timeout=30, **kwargs)


def fetch_validations(start: str, end: str) -> None:
    """Descarga de demanda histórica.

    NOTA: la demanda histórica NO está en esta API en tiempo real, sino en el
    portal de datos abiertos (datos.emtmadrid.es), granularidad día x línea.
    La descarga se implementará en `ingest/emt_opendata.py`.
    """
    raise NotImplementedError(
        "La demanda histórica se obtiene del portal de datos abiertos "
        "(datos.emtmadrid.es), no de la API en tiempo real. "
        "Ver ingest/emt_opendata.py (pendiente) y la sonda "
        "scripts/probe_emt_validations.py."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Utilidades de la API EMT Madrid")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Prueba de humo: hace login y una llamada autenticada.",
    )
    args = parser.parse_args()

    print("[emt] Autenticando...")
    token = login()
    print(f"[emt] OK. accessToken: {token[:12]}...")

    if args.smoke:
        # Llamada autenticada de ejemplo: listado de líneas de bus.
        r = authed_get("/v1/transport/busemtmad/lines/", token)
        print(f"[emt] GET lines -> HTTP {r.status_code}, code={_code(r)}")


if __name__ == "__main__":
    main()
