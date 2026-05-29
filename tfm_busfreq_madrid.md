# BusFreq Madrid
### Optimización dinámica de frecuencias de la EMT Madrid basada en predicción de demanda

---

## Resumen ejecutivo

Los horarios de la EMT Madrid son estáticos: la línea 27 opera con la misma frecuencia un martes lluvioso de enero que un viernes de puente. El resultado son buses vacíos de madrugada y buses colapsados a las 8:30 que no recogen a nadie.

Este TFM propone un sistema end-to-end que predice la demanda de autobús con 24–48h de antelación y recomienda ajustes de frecuencia que minimizan el tiempo de espera medio dentro del presupuesto de flota disponible. La propuesta se cierra con un dashboard interactivo donde cualquier técnico de la EMT puede simular el impacto de un cambio antes de aplicarlo.

**Estimación de horas:** ~130h  
**Datos requeridos:** 100% públicos y gratuitos  
**Viabilidad:** individual, sin dependencias externas

---

## El problema

La EMT publica sus datos de validaciones por línea, parada y franja horaria en `datos.madrid.es`. Nadie los ha usado todavía para optimizar las propias frecuencias de forma dinámica.

Los tres síntomas del problema actual son:

- **Infrautilización en horas valle:** buses circulando con 3–5 pasajeros con la misma cadencia que en hora punta.
- **Saturación predecible pero no gestionada:** eventos recurrentes como partidos en el Bernabéu o el Wanda Metropolitano generan demanda conocida con días de antelación que el sistema no absorbe.
- **Frecuencias uniformes por día de semana:** no se distingue entre un lunes festivo y un lunes laborable, ni entre lluvia y sol.

---

## Objetivos del proyecto

1. Construir un modelo de predicción de demanda por línea, parada y franja horaria con horizonte de 24–48h.
2. Formular y resolver el problema de asignación óptima de frecuencias dado un presupuesto de flota.
3. Desarrollar un dashboard interactivo que permita simular escenarios y comparar frecuencias actuales vs. recomendadas.
4. Validar el sistema sobre datos históricos reales de la EMT Madrid.

---

## Arquitectura del sistema

El sistema se compone de tres módulos principales:

### Módulo 1 — Predicción de demanda

Series temporales por línea y franja horaria entrenadas con historial de validaciones más variables exógenas:

- Día de la semana y hora
- Festivos nacionales, regionales y locales (Madrid)
- Clima histórico y previsto (AEMET API)
- Eventos en el calendario público (partidos, conciertos, ferias)
- Variables de lag y ventanas deslizantes sobre la propia demanda

Se entrena y compara un conjunto de modelos: baseline naive (misma semana del año anterior), LightGBM con features de calendario, Prophet con regresores externos, y un LSTM ligero para capturar dependencias temporales largas.

La evaluación se realiza con validación walk-forward para respetar la causalidad temporal. Métrica principal: MAE de pasajeros por franja horaria.

### Módulo 2 — Optimizador de frecuencias

Dado el forecast de demanda y la flota disponible, el optimizador resuelve: ¿qué frecuencia mínima por línea garantiza que el tiempo de espera no supere N minutos en el percentil 90 de la demanda?

Se formula como un problema de programación lineal entera con restricciones de presupuesto de buses y turnos de conductor. La solución se obtiene con OR-Tools o PuLP.

El output es un vector de frecuencias recomendadas por línea y franja horaria para las próximas 24–48h, junto con el delta de coste estimado respecto al plan vigente.

### Módulo 3 — Dashboard React + API

Interfaz desarrollada en React que consume una API FastAPI con los resultados del optimizador. Funcionalidades principales:

- Selector de línea, fecha y escenario (día normal, lluvia, evento deportivo)
- Comparativa visual frecuencia actual vs. frecuencia recomendada
- Estimación de tiempo de espera medio antes y después del ajuste
- Mapa de Madrid con capa de calor de demanda por parada
- Coste incremental en buses necesarios para el ajuste

---

## Datos y fuentes

| Fuente | Contenido | Acceso |
|---|---|---|
| API EMT Madrid (`opendata.emtmadrid.es`) | Validaciones por línea y parada, posiciones GPS en tiempo real | API pública, registro gratuito |
| GTFS EMT Madrid (`datos.madrid.es`) | Red completa: paradas, trayectos, horarios nominales | Descarga directa |
| AEMET OpenData | Clima histórico y previsto por municipio | API pública gratuita |
| Calendario de festivos | Festivos nacionales, autonómicos y locales | Datos abiertos INE / Gobierno |
| Eventos públicos Madrid | Partidos, conciertos, ferias (parcial) | Scraping / calendario oficial Ayuntamiento |

> La flota disponible por línea es el dato menos accesible. Se puede estimar a partir de las posiciones GPS históricas (número de vehículos únicos por línea y turno) como proxy razonable.

---

## Plan de trabajo

### Fase 1 — Datos y análisis exploratorio (~25h)

- Ingesta y normalización de validaciones históricas (mínimo 12 meses)
- Cruce con GTFS para enriquecer con información geográfica de paradas
- Descarga y alineación temporal de datos de clima y festivos
- Análisis exploratorio: patrones de demanda por línea, hora y día; identificación de las 10–15 líneas con mayor variabilidad (candidatas al piloto)
- Detección de outliers y períodos atípicos (COVID, obras, huelgas)

### Fase 2 — Modelo de predicción de demanda (~35h)

- Feature engineering: variables de lag, ventanas deslizantes, codificación cíclica de hora y día, variables de festivos y eventos
- Entrenamiento de modelos baseline y avanzados
- Evaluación walk-forward sobre el último trimestre disponible
- Análisis de importancia de features con SHAP
- Selección del modelo final y serialización con MLflow

### Fase 3 — Optimizador de frecuencias (~25h)

- Formalización del modelo de tiempo de espera en función de frecuencia y demanda
- Implementación del problema de optimización lineal con OR-Tools
- Definición de restricciones: presupuesto de flota, frecuencia mínima por línea, turnos
- Validación sobre escenarios históricos conocidos (días de lluvia, festivos, partidos)
- Análisis de sensibilidad: impacto de relajar restricciones de presupuesto

### Fase 4 — Dashboard React + FastAPI (~30h)

- API REST con FastAPI: endpoints de predicción, optimización y comparativa de escenarios
- Frontend React con Recharts para gráficas de series temporales
- Integración de Leaflet para mapa de Madrid con capa de demanda
- Selector de escenarios y visualización de recomendaciones
- Despliegue en Azure (App Service + contenedor Docker)

### Fase 5 — Memoria y presentación (~15h)

- Redacción de la memoria siguiendo estructura estándar de TFM
- Análisis crítico de resultados y limitaciones
- Conclusiones y líneas de trabajo futuro
- Preparación de la presentación oral con demo en vivo

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Ingesta y procesamiento | Python, pandas, requests |
| Análisis exploratorio | pandas, seaborn, matplotlib |
| Forecasting | LightGBM, Prophet, PyTorch (LSTM ligero) |
| Interpretabilidad | SHAP |
| Tracking de experimentos | MLflow |
| Optimización | OR-Tools / PuLP |
| API backend | FastAPI |
| Frontend | React, Recharts, Leaflet |
| Despliegue | Azure App Service, Docker |

---

## Alcance del piloto

Para mantener el proyecto dentro de las 130h estimadas, el piloto se limita a un subconjunto representativo de **10–15 líneas** de la EMT seleccionadas por:

- Alta variabilidad de demanda entre franjas horarias
- Paso por zonas de alta afluencia (Bernabéu, estadios, grandes superficies)
- Representatividad geográfica (norte, sur, centro, periferia)

La generalización al resto de la red se deja como trabajo futuro y se documenta como tal en la memoria.

---

## Propuesta de valor y diferenciación

Este TFM se diferencia de trabajos similares en tres aspectos:

**Problema real con datos reales.** No se usan datasets de Kaggle ni datos sintéticos. Todas las validaciones son históricas reales de la EMT Madrid, lo que añade valor académico y credibilidad.

**Cierre del ciclo completo.** La mayoría de trabajos de predicción de demanda de transporte se quedan en el forecasting. Este cierra el ciclo hasta la recomendación accionable (qué frecuencia poner) y la simulación de impacto.

**Demo en vivo en la presentación.** El dashboard permite seleccionar en tiempo real "línea 27, viernes 19:00, partido en el Bernabéu" y mostrar cuántos buses extra se recomiendan y cuánto baja el tiempo de espera estimado. Esto hace la presentación memorable.

---

## Trabajo futuro

- Extensión a la red completa de la EMT (200+ líneas)
- Incorporación de datos de Citymapper / Google Maps para estimar demanda latente (usuarios que abandonaron la espera)
- Integración con el sistema de asignación de turnos de conductor
- Modelo de reoptimización en tiempo real ante incidencias (accidente, desvío de línea)
- Estudio de impacto medioambiental: reducción de emisiones por eliminación de servicios vacíos

---

*Propuesta elaborada como TFM en Data Science — Universidad [nombre]. Curso 2025–2026.*
