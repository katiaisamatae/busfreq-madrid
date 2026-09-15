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
├── codigo/
│   ├── back/               # Backend Python (se ejecuta desde aquí)
│   │   ├── ingest/         #   ingesta: EMT open data, AEMET, config .env
│   │   ├── demand/         #   Módulo 1 — predicción de demanda (LightGBM)
│   │   ├── optimizer/      #   Módulo 2 — optimizador de frecuencias (OR-Tools ILP)
│   │   ├── api/            #   Módulo 3 — backend FastAPI (sirve el front)
│   │   ├── configs/        #   configuración de experimentos (YAML)
│   │   ├── scripts/        #   utilidades puntuales (no se versiona)
│   │   └── mlflow.db       #   tracking de experimentos (no se versiona)
│   ├── front/              # SPA React (CDN) + Chart.js — 3 vistas
│   └── notebooks/          # EDA + selección piloto (figuras en figs/)
├── datos/
│   ├── raw/                # datos crudos descargados (NO se versiona)
│   └── processed/          # datos limpios y features (NO se versiona)
├── docs/                   # estado del arte
├── latex/                  # memoria del TFM (pdflatex + biber)
├── requirements.txt
├── .env.example            # plantilla de claves API (copiar a .env)
└── README.md
```

## Puesta en marcha

```bash
# 1. Entorno Python (requiere Python 3.13)
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt

# 2. Claves API
copy .env.example .env         # luego rellena tus claves de EMT y AEMET

# El backend se ejecuta desde codigo/back
cd codigo/back

# 3. Pipeline de datos
python -m ingest.emt_opendata --all           # demanda, oferta, ocupacion, calendario
python -m ingest.aemet --all                  # meteo historica + prevision (AEMET)
python ../notebooks/eda_gtfs.py               # EDA de la red GTFS
python ../notebooks/eda_demanda.py            # EDA de la demanda
python ../notebooks/eda_saturacion.py         # EDA de saturacion (ocupacion)
python ../notebooks/seleccion_piloto.py       # seleccion de las 12 lineas piloto

# 4. Modulo 1 — demanda
python -m demand.features                                  # dataset de features
python demand/train.py   --config configs/lgbm.yaml        # baseline (walk-forward)
python demand/compare.py --config configs/lgbm.yaml        # comparacion 4 modelos
python demand/tune.py    --config configs/lgbm.yaml        # tuning → lgbm_tuned.yaml
python demand/predict.py --config configs/lgbm_tuned.yaml  # predicciones (dashboard)
python -m demand.intraday                                  # perfiles intradia
mlflow ui --backend-store-uri sqlite:///mlflow.db          # tracking de experimentos

# 5. Modulo 2 — optimizador
python optimizer/optimize.py --date 2026-04-30

# 6. Modulo 3 — dashboard → http://127.0.0.1:8000/
uvicorn api.main:app --reload
```

## Estado del proyecto

- [x] **Fase 1 — Ingesta de datos y EDA** (demanda, oferta, ocupación, calendario, GTFS, AEMET; 12 líneas piloto)
- [x] **Fase 2 — Modelo de predicción de demanda** (LightGBM elegido, MAPE ≈ 8,7 %; Prophet, LSTM, SHAP, MLflow)
- [x] **Fase 3 — Optimizador de frecuencias** (ILP OR-Tools; −4,8 % espera a coste cero)
- [x] **Fase 4 — Dashboard React + FastAPI** (3 vistas, simulación de escenarios)
- [ ] Fase 5 — Despliegue (Docker + Azure) y memoria

## Fuentes de datos (todas públicas y gratuitas)

> **Nota de granularidad:** la demanda pública de la EMT es **diaria por línea**
> (no hay desglose por parada ni por franja horaria). La dimensión intradía se
> deriva del perfil de la *oferta* (sí disponible por franja) y de los horarios GTFS.

| Fuente | Contenido | Granularidad | Acceso |
|---|---|---|---|
| Datos abiertos EMT | Demanda de viajeros | día × línea (2019–2026) | Descarga directa |
| Datos abiertos EMT | Oferta (coches en servicio) | línea × día × franja | Descarga directa |
| Datos abiertos Madrid | Grado de ocupación (saturación) | línea × mes (2019–2024) | Descarga directa |
| Datos abiertos EMT | Calendario de operación | día (2024–2026) | Descarga directa |
| GTFS EMT | Red, paradas, horarios nominales | — | Descarga directa |
| AEMET OpenData | Clima histórico y previsto | día | API gratuita |
| Festivos / Eventos | Calendario nacional, autonómico, local | día | Datos abiertos |
