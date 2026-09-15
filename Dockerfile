# ============================================================
# BusFreq Madrid — imagen del dashboard (backend FastAPI + SPA)
# El backend (codigo/back/api/main.py) sirve TAMBIÉN el frontend
# (codigo/front, rutas relativas), así que un único contenedor
# levanta "front + back".
#
# La estructura del repo se replica dentro de la imagen porque el
# código resuelve rutas relativas a la raíz:
#   ingest/config.py -> PROJECT_ROOT = parents[3] = /app
#   api/main.py      -> ROOT = /app/codigo/back ; FRONTEND = /app/codigo/front
# Los datos (datos/) se montan como volumen (ver docker-compose.yml).
# ============================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1: OpenMP requerido por OR-Tools (CP-SAT)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencias de runtime (capa cacheable: solo se reinstala si cambia el fichero)
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Código: backend + frontend (los datos van por volumen, no en la imagen)
COPY codigo/ ./codigo/

# Todos los comandos del proyecto corren desde codigo/back
WORKDIR /app/codigo/back

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
