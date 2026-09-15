"""El ILP cumple la restricción de flota, respeta la capacidad y su óptimo
coincide con la fuerza bruta en una instancia pequeña (Fase 2)."""
import math

import pandas as pd

from optimizer.optimize import (CAP_BUS, CICLO_DEFECTO, HEADWAY_MAX_MIN,
                                 T_FRANJA, optimizar)


def _instancia():
    # 1 franja, 3 líneas sintéticas (no están en ciclos_linea -> ciclo por defecto).
    return pd.DataFrame({
        "linea": ["A", "B", "C"],
        "franja": [8, 8, 8],
        "buses_actual": [6, 5, 4],     # pool = 15 (con holgura para elegir)
        "pax": [900.0, 300.0, 120.0],
    })


def _cotas(pax, cur, pool, escala=1.0):
    c = CICLO_DEFECTO
    nmax = int(min(pool, max(int(2 * cur * escala) + 1, 12)))
    n_cap = math.ceil(pax * c / (2 * CAP_BUS * T_FRANJA)) if pax > 0 else 1
    n_head = math.ceil(c / HEADWAY_MAX_MIN)
    return max(1, min(max(n_cap, n_head), nmax)), nmax


def _espera_total(asig, pax):
    return sum(pax[l] * CICLO_DEFECTO / (2 * n) for l, n in asig.items())


def test_restriccion_de_flota():
    tabla = _instancia()
    res = optimizar(tabla)
    assert int(res["buses_optimo"].sum()) == int(tabla["buses_actual"].sum())


def test_capacidad_respetada():
    tabla = _instancia()
    pool = int(tabla["buses_actual"].sum())
    res = optimizar(tabla).set_index("linea")
    for _, r in tabla.iterrows():
        lo, _ = _cotas(r["pax"], r["buses_actual"], pool)
        assert res.loc[r["linea"], "buses_optimo"] >= lo


def test_ilp_igual_a_fuerza_bruta():
    tabla = _instancia()
    pool = int(tabla["buses_actual"].sum())
    pax = dict(zip(tabla["linea"], tabla["pax"]))
    # Fuerza bruta explícita sobre el conjunto factible (instancia pequeña).
    cotas = {r["linea"]: _cotas(r["pax"], r["buses_actual"], pool)
             for _, r in tabla.iterrows()}
    mejor = None
    for na in range(cotas["A"][0], cotas["A"][1] + 1):
        for nb in range(cotas["B"][0], cotas["B"][1] + 1):
            for nc in range(cotas["C"][0], cotas["C"][1] + 1):
                if na + nb + nc != pool:
                    continue
                w = _espera_total({"A": na, "B": nb, "C": nc}, pax)
                mejor = w if mejor is None else min(mejor, w)
    res = optimizar(tabla).set_index("linea")["buses_optimo"].to_dict()
    assert mejor is not None
    assert abs(_espera_total(res, pax) - mejor) < 1e-6
