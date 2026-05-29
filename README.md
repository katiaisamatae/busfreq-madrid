# BusFreq Madrid

**Optimización dinámica de frecuencias de la EMT Madrid basada en predicción de demanda.**

TFM — Máster en Data Science. Predice la demanda de autobús con 24–48h de antelación y
recomienda ajustes de frecuencia que minimizan el tiempo de espera dentro del presupuesto de flota.

> Especificación completa del proyecto: [`tfm_busfreq_madrid.md`](tfm_busfreq_madrid.md)
> Guía para el desarrollo: [`CLAUDE.md`](CLAUDE.md)

---

## Arquitectura

```
EMT API / AEMET / GTFS / Eventos
        ↓
1. Predicción de demanda  →  2. Optimizador de frecuencias  →  3. Dashboard
   (LightGBM/Prophet/LSTM)      (OR-Tools/PuLP ILP)              (React + FastAPI)
```

## Estructura del repositorio

```
.
├── data/
│   ├── raw/            # Datos crudos descargados (NO se versiona)
│   └── processed/      # Datos limpios y features (NO se versiona)
├── ingest/             # Scripts de ingesta: EMT API, AEMET, GTFS
├── demand/             # Módulo 1 — predicción de demanda
├── optimizer/          # Módulo 2 — optimización de frecuencias
├── dashboard/
│   ├── api/            # Backend FastAPI
│   └── frontend/       # Frontend React (se inicializa con npm)
├── notebooks/          # Análisis exploratorio (EDA)
├── configs/            # Configuración de experimentos (YAML)
├── requirements.txt
├── .env.example        # Plantilla de claves API (copiar a .env)
└── README.md
```

## Puesta en marcha

```bash
# 1. Entorno Python (requiere Python 3.11)
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt

# 2. Claves API
copy .env.example .env         # luego rellena tus claves de EMT y AEMET

# 3. Comandos principales (a medida que se implementan)
python ingest/emt.py                          # descarga validaciones EMT
python demand/train.py --config configs/lgbm.yaml   # entrena modelo
python optimizer/optimize.py --date 2026-06-01      # ejecuta optimizador
uvicorn dashboard.api.main:app --reload             # backend
mlflow ui                                            # tracking de experimentos
```

## Estado del proyecto

- [x] Estructura del repositorio
- [ ] Fase 1 — Ingesta de datos y EDA
- [ ] Fase 2 — Modelo de predicción de demanda
- [ ] Fase 3 — Optimizador de frecuencias
- [ ] Fase 4 — Dashboard React + FastAPI
- [ ] Fase 5 — Memoria y presentación

## Fuentes de datos (todas públicas y gratuitas)

| Fuente | Contenido | Acceso |
|---|---|---|
| API EMT Madrid | Validaciones por línea/parada/franja, GPS | Registro gratuito |
| GTFS EMT | Red, paradas, horarios nominales | Descarga directa |
| AEMET OpenData | Clima histórico y previsto | API gratuita |
| Festivos / Eventos | Calendario nacional, autonómico, local | Datos abiertos |
