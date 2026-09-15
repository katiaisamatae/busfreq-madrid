"""Optimizador de frecuencias — Módulo 2 (Fase 3).

Dada la demanda por línea y franja horaria (demanda diaria prevista por el
Módulo 1, repartida intradía con el perfil de la oferta) y la flota que la EMT
pone en circulación en cada franja, decide cómo repartir esa MISMA flota entre
las líneas para minimizar el tiempo de espera total de los usuarios.

Formulación como programación lineal entera (ILP) con OR-Tools (CP-SAT):

  variables:   n_{l,f} = nº de autobuses de la línea l en la franja f (entero)
               (linealizado con binarias x_{l,f,n}, una por valor candidato de n)
  objetivo:    min  Σ  pax_{l,f} · T/(2·n_{l,f})         (espera = intervalo/2)
  sujeto a:    Σ_l n_{l,f} = flota_actual_f   (misma flota por franja)
               1 ≤ n_{l,f} ≤ tope             (frecuencia mínima y máxima)

Salida: frecuencias/autobuses recomendados por línea y franja, y comparación de
tiempo de espera frente a la asignación actual (a coste de flota constante).

Uso:
    python optimizer/optimize.py --date 2026-04-15
"""
from __future__ import annotations

import argparse
import math
import sys
import zipfile
from functools import lru_cache
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import DATA_PROCESSED, DATA_RAW

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT.parent / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

T_FRANJA = 60          # duración de franja (min): 1 franja = 1 hora
TIPO = {5: "SA", 6: "FE"}   # dayofweek -> tipo de día (resto = LA)

CICLO_DEFECTO = 60.0   # min: ciclo de respaldo si una línea no está en ciclos_linea
CAP_BUS = 90           # plazas por autobús (capacidad estándar EMT, parametrizable)
HEADWAY_MAX_MIN = 30   # intervalo máximo de política: una línea en servicio no debe
#                        superar este headway (evita "ceder" una línea hasta n=1)


@lru_cache(maxsize=1)
def _ciclos() -> dict:
    """Tiempo de ciclo (ida+vuelta, min) por línea (de optimizer/ciclos.py).

    La espera del usuario es headway/2 = ciclo/(2·n); usar el ciclo real de cada
    línea (no un T=60 uniforme) es lo que hace defendible el reparto óptimo.
    """
    p = DATA_PROCESSED / "ciclos_linea.parquet"
    if not p.exists():
        return {}
    d = pd.read_parquet(p)
    return {str(k): float(v) for k, v in zip(d["linea"], d["ciclo_min"])}


def espera_serie(df: pd.DataFrame, col_buses: str) -> pd.Series:
    """Espera media (min) por fila = ciclo_linea/(2·buses)."""
    ciclo = df["linea"].astype(str).map(_ciclos()).fillna(CICLO_DEFECTO)
    return ciclo / (2 * df[col_buses])


@lru_cache(maxsize=1)
def mapa_codigo_a_linea() -> dict[str, str]:
    with zipfile.ZipFile(DATA_RAW / "gtfs_emt.zip") as z, z.open("routes.txt") as f:
        r = pd.read_csv(f, dtype=str)
    return dict(zip(r["route_id"], r["route_short_name"]))


@lru_cache(maxsize=1)
def _oferta_cruda() -> pd.DataFrame:
    """Carga (y cachea) todos los CSV de oferta con línea y fecha normalizadas."""
    dfs = []
    for f in sorted(glob(str(DATA_RAW / "oferta" / "*.csv"))):
        dfs.append(pd.read_csv(f, sep=";", encoding="utf-8-sig",
                               dtype={"Linea": str, "TIPODIAMO": str,
                                      "CodFranja": int, "NumCCReal": float}))
    of = pd.concat(dfs, ignore_index=True)
    of["linea"] = of["Linea"].map(mapa_codigo_a_linea())
    of["fecha"] = pd.to_datetime(of["FECHASER"]).dt.normalize()
    return of


def oferta_del_dia(fecha: pd.Timestamp, piloto: list[str]) -> pd.DataFrame:
    """Autobuses actuales por línea y franja para la fecha (o media del tipo de día)."""
    of = _oferta_cruda()
    of = of[of["linea"].isin(piloto)]

    dia = of[of["fecha"] == fecha]
    if dia.empty:  # respaldo: media del tipo de día (festivo-aware)
        td = tipo_dia_calendario(fecha)
        dia = (of[of["TIPODIAMO"] == td]
               .groupby(["linea", "CodFranja"])["NumCCReal"].mean().round().reset_index())
    dia = dia.rename(columns={"CodFranja": "franja", "NumCCReal": "buses_actual"})
    dia = dia[dia["buses_actual"] >= 1]
    return dia[["linea", "franja", "buses_actual"]].astype({"buses_actual": int})


def demanda_por_franja(fecha: pd.Timestamp, piloto: list[str]) -> pd.DataFrame:
    """Demanda por línea y franja = demanda diaria × perfil intradía de la oferta."""
    dem = pd.read_parquet(DATA_PROCESSED / "demanda_diaria_linea.parquet")
    dia = dem[(dem["fecha"] == fecha) & (dem["linea"].isin(piloto))]
    if dia.empty:
        raise ValueError(f"Sin demanda para {fecha.date()} (rango hasta 2026-04-30)")
    perfil = pd.read_parquet(DATA_PROCESSED / "perfil_intradia.parquet")
    td = tipo_dia_calendario(fecha)   # festivo-aware (un lunes festivo != laborable)
    perfil = perfil[perfil["tipo_dia"] == td]
    d = dia.merge(perfil, on="linea", how="left")
    d["pax"] = d["viajeros"] * d["share"]
    return d[["linea", "franja", "pax"]]


@lru_cache(maxsize=1)
def _festivos_madrid():
    import holidays
    return holidays.country_holidays("ES", subdiv="MD", years=range(2019, 2031))


def tipo_dia_calendario(fecha: pd.Timestamp) -> str:
    """Tipo de día considerando festivos (FE), no solo el día de la semana."""
    if fecha.date() in _festivos_madrid() or fecha.dayofweek == 6:
        return "FE"
    return TIPO.get(fecha.dayofweek, "LA")


# La predicción de demanda para fechas futuras vive en demand/futuro.py
# (LightGBM sin lags). Antes había aquí una "media de 8 días tipo", que era lógica
# de predicción en la capa equivocada y no usaba ningún modelo.


def optimizar(tabla: pd.DataFrame, escala: float = 1.0) -> pd.DataFrame:
    """ILP: reparte la flota de cada franja entre líneas minimizando la espera.

    `escala` multiplica el presupuesto de flota por franja (1.0 = misma flota;
    1.1 = +10% de autobuses, para el análisis coste/beneficio).

    Cotas por (línea, franja): 1 ≤ n ≤ tope, con dos cotas INFERIORES de servicio:
      * capacidad: n · CAP_BUS · viajes_ida ≥ pax  (no dejar una línea saturada);
      * headway máximo: ciclo/n ≤ HEADWAY_MAX_MIN  (no ceder una línea hasta n=1).
    Si en una franja la suma de mínimos supera la flota disponible, se relaja esa
    franja (no cabe garantizar la capacidad con flota fija; queda documentado).
    """
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    ciclos = _ciclos()
    pool = {f: int(round(c * escala))
            for f, c in tabla.groupby("franja")["buses_actual"].sum().items()}

    # 1) Topes y cotas inferiores (capacidad y headway máximo) por línea y franja.
    n_max_lf, n_lo_lf, ciclo_lf = {}, {}, {}
    for _, row in tabla.iterrows():
        l, f, cur, pax = row["linea"], row["franja"], row["buses_actual"], row["pax"]
        c_l = ciclos.get(str(l), CICLO_DEFECTO)
        ciclo_lf[(l, f)] = c_l
        nmax = int(min(pool[f], max(int(2 * cur * escala) + 1, 12)))
        # viajes (un sentido) por bus y franja = 2·T_FRANJA/ciclo -> plazas = CAP_BUS·viajes
        n_cap = math.ceil(pax * c_l / (2 * CAP_BUS * T_FRANJA)) if pax > 0 else 1
        n_head = math.ceil(c_l / HEADWAY_MAX_MIN)
        n_max_lf[(l, f)] = nmax
        n_lo_lf[(l, f)] = max(1, min(max(n_cap, n_head), nmax))

    # 2) Relajación de factibilidad: si los mínimos de una franja superan su flota.
    franjas_relajadas = 0
    for f, cap in pool.items():
        claves = [k for k in n_lo_lf if k[1] == f]
        if sum(n_lo_lf[k] for k in claves) > cap:
            for k in claves:
                n_lo_lf[k] = 1
            franjas_relajadas += 1

    # 3) Binarias por valor de n en [n_lo, n_max]; exactamente una activa.
    x = {}
    for (l, f), nmax in n_max_lf.items():
        for n in range(n_lo_lf[(l, f)], nmax + 1):
            x[(l, f, n)] = model.NewBoolVar(f"x_{l}_{f}_{n}")
        model.AddExactlyOne(x[(l, f, n)] for n in range(n_lo_lf[(l, f)], nmax + 1))

    # 4) Presupuesto de flota por franja (escalado).
    for f, cap in pool.items():
        buses_f = []
        for (l, ff) in n_max_lf:
            if ff == f:
                buses_f += [n * x[(l, f, n)]
                            for n in range(n_lo_lf[(l, f)], n_max_lf[(l, f)] + 1)]
        model.Add(sum(buses_f) == int(cap))

    # 5) Objetivo: espera total (pax · ciclo_linea/(2n)), escalada a entero.
    terminos = []
    for _, row in tabla.iterrows():
        l, f, pax = row["linea"], row["franja"], row["pax"]
        c_l = ciclo_lf[(l, f)]
        for n in range(n_lo_lf[(l, f)], n_max_lf[(l, f)] + 1):
            terminos.append(int(round(pax * c_l / (2 * n))) * x[(l, f, n)])
    model.Minimize(sum(terminos))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    # Reproducibilidad: 1 worker + semilla fija (con empates puede haber planes
    # alternativos con el mismo óptimo; se documenta en la memoria).
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    estado = solver.Solve(model)
    if estado not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("El ILP no encontró solución factible")

    filas = []
    for _, row in tabla.iterrows():
        l, f = row["linea"], row["franja"]
        n_opt = next(n for n in range(n_lo_lf[(l, f)], n_max_lf[(l, f)] + 1)
                     if solver.Value(x[(l, f, n)]) == 1)
        filas.append({"linea": l, "franja": f, "pax": row["pax"],
                      "buses_actual": row["buses_actual"], "buses_optimo": n_opt})
    out = pd.DataFrame(filas)
    out.attrs["franjas_sin_capacidad"] = franjas_relajadas
    return out


def _reparto_entero(peso, total: int):
    """Reparte `total` enteros proporcional a `peso` (mínimo 1, suma exacta)."""
    import numpy as np
    peso = np.asarray(peso, dtype=float)
    if len(peso) == 0:
        return peso.astype(int)
    if peso.sum() <= 0:
        peso = np.ones_like(peso)
    crudo = peso / peso.sum() * total
    n = np.clip(np.round(crudo), 1, None).astype(int)
    dif = int(total - n.sum())
    orden = list(np.argsort(-(crudo - n))) if dif > 0 else list(np.argsort(crudo - n))
    i = guard = 0
    while dif != 0 and orden and guard < 10 * len(n) + 10:
        k = orden[i % len(orden)]
        if dif > 0:
            n[k] += 1
            dif -= 1
        elif n[k] > 1:
            n[k] -= 1
            dif += 1
        i += 1
        guard += 1
    return n


def asignacion_sqrt(tabla: pd.DataFrame) -> pd.DataFrame:
    """Baseline analítico (regla de la raíz cuadrada de Mohring): reparte la flota
    de cada franja proporcional a sqrt(pax·ciclo), redondeada a enteros que suman la
    MISMA flota. Contextualiza el ILP, que debe igualarlo o mejorarlo."""
    ciclos = _ciclos()
    out = []
    for f, g in tabla.groupby("franja"):
        pool = int(round(g["buses_actual"].sum()))
        c = g["linea"].astype(str).map(ciclos).fillna(CICLO_DEFECTO).to_numpy()
        peso = (g["pax"].clip(lower=0).to_numpy() * c) ** 0.5
        gg = g.copy()
        gg["buses_sqrt"] = _reparto_entero(peso, pool)
        out.append(gg[["linea", "franja", "pax", "buses_actual", "buses_sqrt"]])
    return pd.concat(out, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimiza frecuencias de bus")
    parser.add_argument("--date", default="2026-04-15", help="Fecha objetivo YYYY-MM-DD")
    args = parser.parse_args()
    fecha = pd.Timestamp(args.date).normalize()
    print(f"[optimizer] Optimizando frecuencias para {fecha.date()} "
          f"(tipo de día {tipo_dia_calendario(fecha)})")

    piloto = pd.read_csv(DATA_PROCESSED / "lineas_piloto.csv",
                         dtype={"linea": str})["linea"].tolist()

    dem = demanda_por_franja(fecha, piloto)
    ofe = oferta_del_dia(fecha, piloto)
    tabla = ofe.merge(dem, on=["linea", "franja"], how="inner")
    tabla = tabla[tabla["pax"].notna() & (tabla["pax"] > 0)].reset_index(drop=True)
    print(f"[optimizer] {tabla['linea'].nunique()} líneas, "
          f"{tabla['franja'].nunique()} franjas, {len(tabla)} pares línea-franja")

    res = optimizar(tabla)

    # Espera (min) = ciclo_linea/(2·buses); espera total ponderada por pax.
    res["espera_actual"] = espera_serie(res, "buses_actual")
    res["espera_optima"] = espera_serie(res, "buses_optimo")
    w_act = (res["pax"] * res["espera_actual"]).sum()
    w_opt = (res["pax"] * res["espera_optima"]).sum()
    mejora = (1 - w_opt / w_act) * 100
    espera_media_act = w_act / res["pax"].sum()
    espera_media_opt = w_opt / res["pax"].sum()

    print("\n=== Resultado de la optimización (misma flota por franja) ===")
    print(f"Flota total reasignada: {res['buses_actual'].sum():,} autobuses·franja")
    print(f"Espera media ponderada  ACTUAL : {espera_media_act:.2f} min")
    print(f"Espera media ponderada  ÓPTIMA : {espera_media_opt:.2f} min")
    print(f"Reducción del tiempo de espera : {mejora:.1f}%  (a coste de flota constante)")
    n_cambios = (res["buses_actual"] != res["buses_optimo"]).sum()
    print(f"Pares línea-franja reasignados : {n_cambios} de {len(res)}")
    if res.attrs.get("franjas_sin_capacidad"):
        print(f"Franjas sin capacidad garantizable con flota fija: "
              f"{res.attrs['franjas_sin_capacidad']} (cotas relajadas)")

    # Validación externa: regla de la raíz cuadrada de Mohring (n ∝ sqrt(pax·ciclo)).
    sq = asignacion_sqrt(tabla)
    w_sqrt = float((sq["pax"] * espera_serie(sq, "buses_sqrt")).sum())
    print(f"Baseline Mohring (raíz cuadrada): espera {w_sqrt / res['pax'].sum():.2f} min "
          f"(−{(1 - w_sqrt / w_act) * 100:.1f}% vs actual)  |  ILP: −{mejora:.1f}%")

    salida = DATA_PROCESSED / f"optimizacion_{fecha.date()}.csv"
    res.round(3).to_csv(salida, index=False)
    print(f"[optimizer] recomendación guardada en {salida}")

    # --- Análisis coste/beneficio: espera vs presupuesto de flota ---
    print("\n=== Escenarios coste/beneficio (Δ flota → Δ espera) ===")
    escenarios = []
    for esc in (1.0, 1.05, 1.10, 1.15, 1.20):
        r = optimizar(tabla, escala=esc)
        w = (r["pax"] * espera_serie(r, "buses_optimo")).sum()
        red = (1 - w / w_act) * 100
        escenarios.append({"delta_flota_%": round((esc - 1) * 100),
                           "espera_media_min": w / r["pax"].sum(),
                           "reduccion_espera_%": red})
        print(f"  flota {(esc-1)*100:+4.0f}%  ->  espera media {w/r['pax'].sum():.2f} min "
              f"(−{red:.1f}% vs actual)")
    esc_df = pd.DataFrame(escenarios)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(esc_df["delta_flota_%"], esc_df["reduccion_espera_%"],
            marker="o", color="#0178BC", lw=2)
    ax.set_title("Coste/beneficio: reducción de espera según presupuesto de flota")
    ax.set_xlabel("Δ flota respecto a la actual (%)  →  coste")
    ax.set_ylabel("Reducción del tiempo de espera (%)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "optimizacion_coste_beneficio.png", dpi=120)
    plt.close()
    print(f"[optimizer] figura coste/beneficio en "
          f"{FIGS / 'optimizacion_coste_beneficio.png'}")

    # --- Figura: actual vs recomendado en la franja punta ---
    punta = res.groupby("franja")["pax"].sum().idxmax()
    sub = res[res["franja"] == punta].sort_values("pax", ascending=False)
    x = range(len(sub))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - 0.2 for i in x], sub["buses_actual"], width=0.4,
           label="actual", color="#9aa7b1")
    ax.bar([i + 0.2 for i in x], sub["buses_optimo"], width=0.4,
           label="recomendado", color="#0178BC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["linea"])
    ax.set_title(f"Autobuses por línea en la franja punta ({punta}) — "
                 f"{fecha.date()} (misma flota total)")
    ax.set_xlabel("Línea")
    ax.set_ylabel("Nº de autobuses")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "optimizacion_franja_punta.png", dpi=120)
    plt.close()
    print(f"[optimizer] figura en {FIGS / 'optimizacion_franja_punta.png'}")


if __name__ == "__main__":
    main()
