# Estado del arte — BusFreq Madrid

Sistema de revisión bibliográfica del TFM. Trabaja aquí mientras esperas las
claves de API: es lo que más tiempo lleva en la memoria y no depende de datos.

## Archivos

- **`referencias.bib`** — bibliografía en BibTeX (impórtala en Zotero/Mendeley/Overleaf).
- **`PLANTILLA_ficha.md`** — plantilla para fichar cada trabajo que leas.
- **`ficha_*.md`** — una ficha por trabajo (ej. `ficha_pereira2015events.md`).

## Flujo de trabajo

1. Busca con las cadenas de abajo y guarda los candidatos en `referencias.bib`.
2. Por cada trabajo prometedor: copia `PLANTILLA_ficha.md` → `ficha_<clave>.md`.
3. Rellena la ficha al leer; lo importante es la sección *Relevancia para BusFreq*.
4. Marca el estado en la tabla de seguimiento.
5. La síntesis de las fichas se convierte en el capítulo *Estado del arte* de la memoria.

> 💡 Recomendación: usa **Zotero** (gratis) con el conector del navegador. Importa
> `referencias.bib`, y exporta de vuelta para mantener todo sincronizado.

## Bloques temáticos

| Bloque | Tema | Para qué sección del TFM |
|---|---|---|
| A | Modelos ML / forecasting (LightGBM, Prophet, LSTM, SHAP) | Justificar elección de modelos |
| B | Validación de series temporales (walk-forward) | Justificar metodología de evaluación |
| C | Predicción de demanda de transporte público | Contexto y trabajos previos |
| D | Optimización de frecuencias / headways | Núcleo del Módulo 2 |
| E | Variables exógenas (clima, eventos) | Justificar el feature engineering |
| F | Datos abiertos, GTFS, herramientas | Marco técnico |

## Fuentes de búsqueda

- **Google Scholar** — https://scholar.google.com (empieza aquí; usa "Citado por" para seguir hilos)
- **Scopus** / **Web of Science** — acceso vía biblioteca de tu universidad
- **IEEE Xplore** — https://ieeexplore.ieee.org (mucho de ITS y transporte inteligente)
- **ScienceDirect** — https://www.sciencedirect.com (Transportation Research Part A/B/C)
- **arXiv** — https://arxiv.org (preprints de ML)
- **TRID** — https://trid.trb.org (base de datos específica de transporte)

## Cadenas de búsqueda sugeridas

**Demanda (Bloque C):**
- `public transport demand forecasting machine learning`
- `bus ridership prediction LightGBM gradient boosting`
- `short-term passenger flow prediction LSTM`
- `transit smart card validation demand prediction`

**Optimización (Bloque D):**
- `bus frequency setting optimization`
- `transit headway optimization integer programming`
- `service frequency allocation transit waiting time`
- `transit network frequency setting problem ILP`

**Exógenas (Bloque E):**
- `weather impact public transit ridership`
- `special events public transport demand prediction`
- `event-driven transit demand forecasting`

**Contexto local (valioso para diferenciarte):**
- `EMT Madrid demand prediction`
- `Madrid public transport open data`
- `dynamic bus frequency demand responsive`

## Criterios de inclusión / exclusión

- ✅ Preferir 2015–2026, salvo clásicos fundacionales (Furth & Wilson 1981, Ceder).
- ✅ Priorizar trabajos con datos reales y métricas reproducibles.
- ✅ Priorizar revistas/conferencias de transporte e ITS de impacto.
- ❌ Descartar trabajos puramente teóricos sin aplicación, o con datos sintéticos
  (recuerda: tu diferenciador es usar datos reales de la EMT).

## Tabla de seguimiento

| Clave | Bloque | Estado | Relevancia | Ficha |
|---|---|---|---|---|
| ke2017lightgbm | A | por leer | ⭐⭐⭐⭐ | — |
| chen2016xgboost | A | por leer | ⭐⭐⭐ | — |
| taylor2018prophet | A | por leer | ⭐⭐⭐⭐ | — |
| hochreiter1997lstm | A | por leer | ⭐⭐⭐ | — |
| lundberg2017shap | A | por leer | ⭐⭐⭐⭐ | — |
| bergmeir2012crossval | B | por leer | ⭐⭐⭐⭐⭐ | — |
| hyndman2021fpp | B | por leer | ⭐⭐⭐⭐ | — |
| ma2015lstm | C | por leer | ⭐⭐⭐ | — |
| toque2017forecasting | C | por leer | ⭐⭐⭐ | — |
| furth1981frequencies | D | por leer | ⭐⭐⭐⭐⭐ | — |
| ceder2007transit | D | por leer | ⭐⭐⭐⭐ | — |
| verbas2013frequency | D | por leer | ⭐⭐⭐⭐ | — |
| farahani2013review | D | por leer | ⭐⭐⭐ | — |
| singhal2014weather | E | por leer | ⭐⭐⭐⭐ | — |
| pereira2015events | E | por leer | ⭐⭐⭐⭐⭐ | ✅ |
| rodrigues2019events | E | por leer | ⭐⭐⭐⭐ | — |

> Objetivo orientativo para un TFM: **20–35 referencias** bien fichadas.
> Empiezas con 16; añade las que encuentres en las cadenas de búsqueda.
