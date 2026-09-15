"""API FastAPI del dashboard — Módulo 3 (Fase 4).

Expone las tres capas del sistema para el frontend:
  * Predicción de demanda (Módulo 1): predicho vs real y MAPE por línea, más los
    factores de demanda medidos en el histórico (tipo de día, lluvia).
  * Optimización (Módulo 2): comparación actual vs recomendada, detalle por línea
    y franja, saturación por línea y curva coste/beneficio.
También sirve el propio frontend (SPA en codigo/front).

Uso (desarrollo):
    uvicorn api.main:app --reload  # desde codigo/back   # -> http://127.0.0.1:8000/
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]  # codigo/back
sys.path.insert(0, str(ROOT))

from ingest.config import DATA_PROCESSED  # noqa: E402
from ingest.contexto import contexto_dias  # noqa: E402
from demand.futuro import demanda_futura_por_franja, mape_walkforward  # noqa: E402
from optimizer.optimize import (  # noqa: E402
    demanda_por_franja, espera_serie, oferta_del_dia, optimizar,
    tipo_dia_calendario)

FRONTEND = ROOT.parent / "front"

app = FastAPI(title="BusFreq Madrid API", version="2.0.0",
              description="Predicción de demanda y optimización de frecuencias — EMT Madrid")
# CORS parametrizable por entorno (abierto en desarrollo; en Azure se fija
# CORS_ORIGINS al dominio del front).
_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_ORIGINS, allow_methods=["*"], allow_headers=["*"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("busfreq")

# Serializa las resoluciones CP-SAT: evita que peticiones concurrentes disparen N
# solves simultáneos (un plan puede lanzar 6). Las cargas de datos van cacheadas.
_CPSAT_LOCK = threading.Lock()


def _optimizar(tabla, escala: float = 1.0):
    """optimizar() serializado por _CPSAT_LOCK (protege el servicio ante ráfagas)."""
    with _CPSAT_LOCK:
        return optimizar(tabla, escala=escala)


# --------------------------------------------------------------------------- #
# Cargadores cacheados
# --------------------------------------------------------------------------- #
def _piloto() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "lineas_piloto.csv", dtype={"linea": str})


@lru_cache(maxsize=1)
def _predicciones() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "predicciones.parquet")


@lru_cache(maxsize=1)
def _saturacion() -> dict:
    """Saturación media reciente (2023+) por línea = 100 - % calidad ocupación."""
    oc = pd.read_parquet(DATA_PROCESSED / "ocupacion_linea_mes.parquet")
    oc["sat"] = 100 - oc["pct_calidad_ocupacion"]
    s = oc[oc["anio"] >= 2023].groupby("linea")["sat"].mean()
    return {k: round(float(v), 2) for k, v in s.items()}


@lru_cache(maxsize=1)
def _factores() -> dict:
    """Factores de demanda MEDIDOS en el histórico (post-COVID)."""
    df = pd.read_parquet(DATA_PROCESSED / "dataset_features.parquet")
    df = df[df["fecha"] >= "2022-01-01"]
    g = df.groupby("fecha").agg(v=("viajeros", "sum"), fest=("is_festivo", "first"),
                                dow=("dia_semana", "first"), prec=("prec", "first"))
    g = g.reset_index()
    base = g[(g.dow < 5) & (g.fest == 0)]
    return {
        "sabado": round(float(g[g.dow == 5]["v"].mean() / base["v"].mean()), 2),
        "festivo": round(float(g[(g.fest == 1) | (g.dow == 6)]["v"].mean() / base["v"].mean()), 2),
        "lluvia": round(_factor_lluvia(base), 2),
    }


def _factor_lluvia(base: pd.DataFrame, min_dias: int = 3) -> float:
    """Cociente lluvia/seco promediado DENTRO de cada mes-año.

    Comparar sin más la media de los días lluviosos con la de los secos da 1,06,
    pero es un artefacto: en Madrid llueve en los meses de demanda alta y no
    llueve en julio y agosto, el valle del año, de modo que el grupo «seco»
    arrastra el verano. Contrastando solo días del mismo mes el efecto se
    desvanece (véase la calibración del contexto en la memoria).
    """
    ratios, pesos = [], []
    for _, sub in base.groupby([base["fecha"].dt.year, base["fecha"].dt.month]):
        lluv, seco = sub[sub.prec > 1.0], sub[sub.prec <= 0.1]
        if len(lluv) >= min_dias and len(seco) >= min_dias:
            ratios.append(float(lluv["v"].mean() / seco["v"].mean()))
            pesos.append(len(lluv))
    if not ratios:
        return 1.0
    return sum(r * p for r, p in zip(ratios, pesos)) / sum(pesos)


def _tabla_escenario(fecha_str: str, factor: float,
                     linea_evento: str = "", factor_evento: float = 1.0):
    try:
        fecha = pd.Timestamp(fecha_str).normalize()
    except ValueError:
        raise HTTPException(400, "Fecha inválida (YYYY-MM-DD)")
    piloto = _piloto()["linea"].tolist()
    try:
        dem = demanda_por_franja(fecha, piloto).copy()
    except (AssertionError, ValueError) as e:
        raise HTTPException(400, str(e))
    dem["pax"] = dem["pax"] * factor
    # Evento localizado: dispara la demanda de UNA línea (no uniforme), por lo que
    # sí altera el reparto óptimo entre líneas.
    if linea_evento:
        dem.loc[dem["linea"] == linea_evento, "pax"] *= factor_evento
    ofe = oferta_del_dia(fecha, piloto)
    tabla = ofe.merge(dem, on=["linea", "franja"], how="inner")
    tabla = tabla[tabla["pax"].notna() & (tabla["pax"] > 0)].reset_index(drop=True)
    if tabla.empty:
        raise HTTPException(400, "Sin datos para ese día/escenario")
    pax_tot = float(tabla["pax"].sum())
    w_act = float((tabla["pax"] * espera_serie(tabla, "buses_actual")).sum())
    return tabla, fecha, w_act, pax_tot


def _por_linea_y_detalle(res: pd.DataFrame, base: pd.DataFrame | None = None):
    """Agregados por línea y detalle línea-franja, comunes a los dos planes.

    `res` ya trae las columnas espera_actual/espera_optima. Si se pasa `base` (la
    tabla con pax_base de la ruta futura), se añade la demanda sin contexto y su
    variación, que es lo único que distingue a un plan futuro de uno histórico.
    """
    nombres = _piloto().set_index("linea")
    sat = _saturacion()
    por_linea = []
    for linea, gg in res.groupby("linea"):
        px = float(gg["pax"].sum())
        fila = {
            "linea": linea,
            "recorrido": str(nombres.loc[linea, "route_long_name"])
            if "route_long_name" in nombres.columns else linea,
            "viajeros": round(px),
            "buses_actual": int(gg["buses_actual"].sum()),
            "buses_optimo": int(gg["buses_optimo"].sum()),
            "espera_actual": round(float((gg["pax"] * gg["espera_actual"]).sum() / px), 2),
            "espera_optima": round(float((gg["pax"] * gg["espera_optima"]).sum() / px), 2),
            "saturacion": sat.get(linea),
        }
        if base is not None:
            base_l = float(base.loc[base["linea"] == linea, "pax_base"].sum())
            fila["viajeros_base"] = round(base_l)
            fila["delta_pct"] = round((px / base_l - 1) * 100, 1) if base_l else 0.0
        fila["mejora_pct"] = round((1 - fila["espera_optima"] / fila["espera_actual"]) * 100, 1)
        por_linea.append(fila)
    por_linea.sort(key=lambda d: d["viajeros"], reverse=True)

    detalle = [{"linea": r["linea"], "franja": int(r["franja"]),
                "pax": round(float(r["pax"]), 1),
                "buses_actual": int(r["buses_actual"]),
                "buses_optimo": int(r["buses_optimo"])}
               for _, r in res.sort_values(["linea", "franja"]).iterrows()]
    return por_linea, detalle


class Escenario(BaseModel):
    fecha: str = "2026-04-30"
    factor_demanda: float = 1.0
    delta_flota_pct: float = 0.0
    linea_evento: str = ""
    factor_evento: float = 1.28   # partido de ~60.000 espectadores (calibrado)


class EscenarioBase(BaseModel):
    fecha: str = "2026-04-30"
    factor_demanda: float = 1.0
    linea_evento: str = ""
    factor_evento: float = 1.28   # partido de ~60.000 espectadores (calibrado)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    """Comprueba existencia y frescura de los parquet clave (útil en Azure)."""
    claves = {"predicciones": "predicciones.parquet",
              "perfil_intradia": "perfil_intradia.parquet",
              "ciclos_linea": "ciclos_linea.parquet",
              "demanda_diaria": "demanda_diaria_linea.parquet"}
    datos, ok = {}, True
    for nombre, fichero in claves.items():
        ruta = DATA_PROCESSED / fichero
        existe = ruta.exists()
        ok = ok and existe
        datos[nombre] = {"existe": existe,
                         "modificado": (datetime.fromtimestamp(ruta.stat().st_mtime)
                                        .isoformat(timespec="seconds") if existe else None)}
    return {"status": "ok" if ok else "degradado", "datos": datos}


@app.get("/api/lineas")
def lineas() -> list[dict]:
    df = _piloto()
    cols = {"linea": "linea", "route_long_name": "recorrido", "sector": "sector",
            "lat": "lat", "lon": "lon", "media": "viajeros_dia"}
    df = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    sat = _saturacion()
    out = df.round({"viajeros_dia": 0, "lat": 5, "lon": 5}).to_dict("records")
    for r in out:
        r["saturacion"] = sat.get(r["linea"])
    return out


@app.get("/api/prediccion")
def prediccion() -> dict:
    """Predicho vs real por línea (walk-forward), MAPE, banda p10-p90 y factores."""
    pred = _predicciones()
    lineas = {}
    for linea, g in pred.groupby("linea"):
        g = g.sort_values("fecha")
        real = g["real"].to_numpy()
        pr = g["pred"].to_numpy()
        mape = float((abs(pr - real)[real > 0] / real[real > 0]).mean() * 100)
        # cobertura real del intervalo: es del 57,7 % frente al 80 % nominal, así
        # que la banda se pinta pero se etiqueta como demasiado estrecha
        dentro = float(((real >= g["pred_p10"]) & (real <= g["pred_p90"])).mean() * 100)
        lineas[linea] = {
            "mape": round(mape, 1),
            "cobertura_pct": round(dentro, 1),
            "serie": [{"fecha": f.strftime("%Y-%m-%d"), "real": int(r), "pred": int(p),
                       "p10": int(lo), "p90": int(hi)}
                      for f, r, p, lo, hi in zip(g["fecha"], g["real"], g["pred"],
                                                 g["pred_p10"], g["pred_p90"])],
        }
    real = pred["real"].to_numpy()
    pr = pred["pred"].to_numpy()
    mape_g = float((abs(pr - real)[real > 0] / real[real > 0]).mean() * 100)
    cob_g = float(((real >= pred["pred_p10"]) & (real <= pred["pred_p90"])).mean() * 100)
    return {"mape_global": round(mape_g, 1), "cobertura_global_pct": round(cob_g, 1),
            "lineas": lineas, "factores": _factores()}


@app.post("/api/optimizar")
def api_optimizar(esc: Escenario) -> dict:
    tabla, fecha, w_act, pax_tot = _tabla_escenario(
        esc.fecha, esc.factor_demanda, esc.linea_evento, esc.factor_evento)
    res = _optimizar(tabla, escala=1 + esc.delta_flota_pct / 100)
    res["espera_actual"] = espera_serie(res, "buses_actual")
    res["espera_optima"] = espera_serie(res, "buses_optimo")
    w_opt = float((res["pax"] * res["espera_optima"]).sum())

    por_linea, detalle = _por_linea_y_detalle(res)
    punta = int(res.groupby("franja")["pax"].sum().idxmax())

    return {
        "fecha": str(fecha.date()), "tipo_dia": tipo_dia_calendario(fecha),
        "franja_punta": punta,
        "kpis": {
            "espera_actual": round(w_act / pax_tot, 2),
            "espera_optima": round(w_opt / pax_tot, 2),
            "reduccion_pct": round((1 - w_opt / w_act) * 100, 1),
            "flota_actual": int(res["buses_actual"].sum()),
            "flota_optima": int(res["buses_optimo"].sum()),
            "delta_flota_pct": esc.delta_flota_pct,
            "viajeros_dia": round(pax_tot),
        },
        "por_linea": por_linea, "detalle": detalle,
    }


# --------------------------------------------------------------------------- #
# Predicción a futuro con contexto automático (clima + eventos, sin entrada manual)
# --------------------------------------------------------------------------- #
@app.get("/api/contexto")
def api_contexto(dias: int = 7, forzar: bool = False) -> dict:
    """Clima y eventos de los próximos días, detectados en las APIs externas."""
    try:
        return contexto_dias(dias, forzar=forzar)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"No se pudo construir el contexto: {e}")


class PlanFuturo(BaseModel):
    fecha: str
    delta_flota_pct: float = 0.0


@app.post("/api/plan_futuro")
def api_plan_futuro(req: PlanFuturo) -> dict:
    """Predicción + plan de flota para una fecha futura.

    Demanda base = patrón reciente del mismo tipo de día (trazable), sobre la
    que se aplican los factores DETECTADOS en el contexto (lluvia, eventos).
    """
    try:
        fecha = pd.Timestamp(req.fecha).normalize()
    except ValueError:
        raise HTTPException(400, "Fecha inválida (YYYY-MM-DD)")

    ctx = contexto_dias(7)
    dia_ctx = next((d for d in ctx["dias"] if d["fecha"] == str(fecha.date())), None)

    piloto = _piloto()["linea"].tolist()
    try:
        dem, td, metodo = demanda_futura_por_franja(fecha, piloto)
    except (AssertionError, ValueError) as e:
        raise HTTPException(400, str(e))

    dem = dem.copy()
    dem["pax_base"] = dem["pax"]
    aplicados: list[str] = []
    if dia_ctx:
        clima = dia_ctx["clima"]
        if clima["lluvia"]:
            dem["pax"] = dem["pax"] * clima["factor_red"]
            aplicados.append(
                f"lluvia (prob. {clima['prob_prec']} %) → demanda de red "
                f"×{clima['factor_red']}")
        for ev in dia_ctx["eventos"]:
            # el evento se detecta y se muestra siempre, pero no toca la demanda
            # si su aforo queda bajo el umbral calibrado (factor 1,0) o si no
            # afecta a ninguna línea piloto (p. ej. el Metropolitano)
            if ev["factor"] <= 1.0 or not ev["lineas"]:
                continue
            m = dem["linea"].isin(ev["lineas"]) & dem["franja"].isin(ev["franjas"])
            dem.loc[m, "pax"] = dem.loc[m, "pax"] * ev["factor"]
            aplicados.append(
                f"{ev['nombre']} ({ev['lugar']}) → líneas {', '.join(ev['lineas'])} "
                f"≈ +{ev['efecto_pct']} % en franjas {ev['ventana']}")

    ofe = oferta_del_dia(fecha, piloto)  # sin programación futura → media del tipo de día
    tabla = ofe.merge(dem, on=["linea", "franja"], how="inner")
    tabla = tabla[tabla["pax"].notna() & (tabla["pax"] > 0)].reset_index(drop=True)
    if tabla.empty:
        raise HTTPException(400, "Sin datos suficientes para estimar ese día")

    pax_tot = float(tabla["pax"].sum())
    w_act = float((tabla["pax"] * espera_serie(tabla, "buses_actual")).sum())
    res = _optimizar(tabla[["linea", "franja", "pax", "buses_actual"]],
                     escala=1 + req.delta_flota_pct / 100)
    res["espera_actual"] = espera_serie(res, "buses_actual")
    res["espera_optima"] = espera_serie(res, "buses_optimo")
    w_opt = float((res["pax"] * res["espera_optima"]).sum())

    por_linea, detalle = _por_linea_y_detalle(res, base=tabla)

    franjas = (tabla.groupby("franja")
               .agg(pax=("pax", "sum"), pax_base=("pax_base", "sum"),
                    buses_actual=("buses_actual", "sum")).reset_index())
    fr_opt = res.groupby("franja")["buses_optimo"].sum()
    franjas["buses_optimo"] = franjas["franja"].map(fr_opt).fillna(0).astype(int)
    franjas_out = [{"franja": int(r["franja"]), "pax": round(float(r["pax"])),
                    "pax_base": round(float(r["pax_base"])),
                    "buses_actual": int(r["buses_actual"]),
                    "buses_optimo": int(r["buses_optimo"])}
                   for _, r in franjas.sort_values("franja").iterrows()]

    curva = []
    for d in (0, 5, 10, 15, 20):
        r = _optimizar(tabla[["linea", "franja", "pax", "buses_actual"]], escala=1 + d / 100)
        w = float((r["pax"] * espera_serie(r, "buses_optimo")).sum())
        curva.append({"delta_flota_pct": d, "espera_media": round(w / pax_tot, 2),
                      "reduccion_pct": round((1 - w / w_act) * 100, 1)})

    return {
        "fecha": str(fecha.date()), "tipo_dia": td,
        "metodo": metodo,
        "mape_modelo": mape_walkforward(),
        "contexto": dia_ctx, "factores_aplicados": aplicados,
        "franja_punta": int(res.groupby("franja")["pax"].sum().idxmax()),
        "kpis": {
            "viajeros_dia": round(pax_tot),
            "viajeros_base": round(float(tabla["pax_base"].sum())),
            "espera_actual": round(w_act / pax_tot, 2),
            "espera_optima": round(w_opt / pax_tot, 2),
            "reduccion_pct": round((1 - w_opt / w_act) * 100, 1),
            "flota_actual": int(res["buses_actual"].sum()),
            "flota_optima": int(res["buses_optimo"].sum()),
            "buses_movidos": int((res["buses_optimo"] - res["buses_actual"]).clip(lower=0).sum()),
            # denominador del KPI: sin él, "57 buses reasignados" no dice nada
            "pares": int(len(res)),
            "lineas_ganan": int((res["buses_optimo"] > res["buses_actual"]).sum()),
            "lineas_ceden": int((res["buses_optimo"] < res["buses_actual"]).sum()),
            "delta_flota_pct": req.delta_flota_pct,
        },
        "por_linea": por_linea, "detalle": detalle,
        "franjas": franjas_out, "curva": curva,
    }


# --------------------------------------------------------------------------- #
# Diagnóstico retrospectivo — qué se hizo mal en un mes ya observado
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=6)
def _diagnostico_mes(mes: str) -> dict:
    dem = pd.read_parquet(DATA_PROCESSED / "demanda_diaria_linea.parquet")
    fechas = sorted({f for f in dem["fecha"].unique()
                     if str(pd.Timestamp(f))[:7] == mes})
    if not fechas:
        raise HTTPException(404, f"Sin demanda para el mes {mes}")

    feats = (pd.read_parquet(DATA_PROCESSED / "dataset_features.parquet")
             [["fecha", "prec", "is_festivo"]].groupby("fecha").first())
    dias, casos = [], []
    tot_evitable_min = 0.0
    for f in fechas:
        ts = pd.Timestamp(f)
        try:
            tabla, _, w_act, pax_tot = _tabla_escenario(str(ts.date()), 1.0)
        except HTTPException:
            continue
        res = _optimizar(tabla)
        res["esp_a"] = espera_serie(res, "buses_actual")
        res["esp_o"] = espera_serie(res, "buses_optimo")
        res["gan_min"] = res["pax"] * (res["esp_a"] - res["esp_o"])
        w_opt = float((res["pax"] * res["esp_o"]).sum())
        red = (1 - w_opt / w_act) * 100
        evitable_min = float(res["gan_min"].clip(lower=0).sum())
        tot_evitable_min += evitable_min

        prec = float(feats.loc[ts, "prec"]) if ts in feats.index else 0.0
        festivo = bool(feats.loc[ts, "is_festivo"]) if ts in feats.index else False
        td = tipo_dia_calendario(ts)
        dias.append({"fecha": str(ts.date()), "tipo_dia": td,
                     "viajeros": round(pax_tot), "reduccion_pct": round(red, 1),
                     "lluvia_mm": round(prec, 1), "festivo": festivo})

        top = res.loc[res["gan_min"].idxmax()]
        if float(top["gan_min"]) > 0:
            if prec > 1.0:
                causa = f"lluvia ({prec:.0f} mm) no considerada en la programación"
            elif festivo or td == "FE":
                causa = "festivo/domingo con oferta de día tipo"
            elif td == "SA":
                causa = "patrón de sábado mal ajustado"
            else:
                causa = "pico no anticipado (posible evento)"
            casos.append({
                "fecha": str(ts.date()), "linea": str(top["linea"]),
                "franja": int(top["franja"]), "causa": causa,
                "buses_actual": int(top["buses_actual"]),
                "buses_optimo": int(top["buses_optimo"]),
                "espera_actual": round(float(top["esp_a"]), 1),
                "espera_optima": round(float(top["esp_o"]), 1),
                "mejora_pct": round((1 - float(top["esp_o"]) / float(top["esp_a"])) * 100, 1),
                "evitable_h": round(float(top["gan_min"]) / 60, 1),
            })

    casos.sort(key=lambda c: -c["evitable_h"])
    meses = sorted({str(pd.Timestamp(f))[:7] for f in dem["fecha"].unique()})[-6:]
    return {
        "mes": mes, "meses_disponibles": meses,
        "kpis": {
            "viajeros": sum(d["viajeros"] for d in dias),
            "dias_analizados": len(dias),
            "dias_desajuste": sum(1 for d in dias if d["reduccion_pct"] >= 3),
            "espera_evitable_h": round(tot_evitable_min / 60),
        },
        "dias": dias, "casos": casos[:8],
    }


@app.get("/api/diagnostico")
def api_diagnostico(mes: str = "") -> dict:
    """Desajustes oferta-demanda de un mes (ILP retrospectivo, cacheado)."""
    if not mes:
        dem = pd.read_parquet(DATA_PROCESSED / "demanda_diaria_linea.parquet")
        mes = str(pd.Timestamp(dem["fecha"].max()))[:7]
    return _diagnostico_mes(mes)


@app.post("/api/coste_beneficio")
def api_coste_beneficio(esc: EscenarioBase) -> dict:
    tabla, fecha, w_act, pax_tot = _tabla_escenario(
        esc.fecha, esc.factor_demanda, esc.linea_evento, esc.factor_evento)
    puntos = []
    for d in (0, 5, 10, 15, 20):
        r = _optimizar(tabla, escala=1 + d / 100)
        w = float((r["pax"] * espera_serie(r, "buses_optimo")).sum())
        puntos.append({"delta_flota_pct": d, "espera_media": round(w / pax_tot, 2),
                       "reduccion_pct": round((1 - w / w_act) * 100, 1)})
    return {"fecha": str(fecha.date()), "puntos": puntos}


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
