# Ficha de lectura — Predicción de transporte en eventos especiales

> ⚠️ EJEMPLO de ficha rellenada (a partir del resumen/abstract). Verifica los
> detalles leyendo el paper completo y ajusta lo que haga falta.

| Campo | Contenido |
|---|---|
| **Clave BibTeX** | `pereira2015events` |
| **Bloque temático** | E · Variables exógenas (eventos) |
| **Estado** | por leer |
| **Relevancia (1-5)** | ⭐⭐⭐⭐⭐ |

## Referencia completa
Pereira, F. C., Rodrigues, F., & Ben-Akiva, M. (2015). *Using Data from the Web
to Predict Public Transport Arrivals under Special Events Scenarios*. Journal of
Intelligent Transportation Systems, 19(3), 273–288.

## Problema que aborda
Cómo anticipar picos de demanda de transporte público causados por **eventos
especiales** (conciertos, partidos), que rompen los patrones habituales y que
los modelos de series temporales estándar no capturan.

## Datos utilizados
*(A completar al leer: red de transporte usada, periodo, y cómo recogen los
datos de eventos desde la web.)*

## Método / enfoque
Combinan datos históricos de demanda con **información textual de eventos
extraída de la web**, para mejorar la predicción en escenarios atípicos.

## Resultados y métricas
*(A completar: qué mejora obtienen frente al baseline sin información de eventos.)*

## Limitaciones reconocidas
*(A completar al leer.)*

## 🎯 Relevancia para BusFreq Madrid
**Muy alta.** Es la justificación académica directa de mi variable de eventos
(partidos en Bernabéu / Metropolitano). Demuestra que incorporar eventos mejora
la predicción en escenarios atípicos → respalda mi feature `is_match_day`.
Posible baseline conceptual: comparar mi modelo con y sin variable de eventos.

## Cita textual clave
> "…" (p. X) — *(extraer al leer)*

## Conexión con otros trabajos
- Extiende la idea con deep learning en [[ficha_rodrigues2019events]] (mismos autores).
- Complementa el efecto del clima de `singhal2014weather`.
