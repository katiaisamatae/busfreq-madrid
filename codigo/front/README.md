# Frontend — BusFreq Madrid

Cuadro de mando del sistema (Fase 4). Implementado como **SPA React** en un único
`index.html` (React + Leaflet + Chart.js vía CDN, sin paso de compilación), que
sirve el propio backend FastAPI. Esto permite ejecutarlo con solo levantar la API.

## Cómo ejecutarlo

```bash
cd codigo/back && uvicorn api.main:app --reload
# abrir http://127.0.0.1:8000/
```

El frontend llama a la API del mismo origen (`/api/...`).

## Funcionalidades

- Selector de **fecha** y de **escenario** (día normal, lluvia, evento deportivo,
  festivo) más sliders de factor de demanda y presupuesto de flota.
- **KPIs**: tiempo de espera medio actual vs. óptimo, reducción de espera,
  incremento de coste (flota) y viajeros del escenario.
- **Gráfico** (Chart.js) de autobuses por línea en la franja punta: actual vs.
  recomendado.
- **Mapa de Madrid** (Leaflet) con las líneas piloto: tamaño ∝ demanda, color
  según la mejora de espera.
- **Tabla** de detalle por línea.

## Nota

Para una versión productiva con empaquetado (y la posibilidad de usar Recharts y
react-leaflet como componentes), se puede migrar a un proyecto Vite:

```bash
npm create vite@latest . -- --template react
npm install && npm install recharts leaflet react-leaflet
npm run dev
```
