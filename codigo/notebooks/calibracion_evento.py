"""Calibración empírica del factor de evento (tarea 3.2 del plan de mejora).

`ingest/contexto.py::_factor_evento` usaba `1 + min(0,9; asistencia/75000)`, una
suposición sin respaldo: para un partido de 60.000 espectadores implicaba +80 %
de demanda en las líneas cercanas. Aquí medimos el efecto real con el histórico
de demanda diaria (día x línea, 2022-2026) y los partidos de LaLiga disputados
en Madrid.

Diseño (diferencias en diferencias con efectos fijos):

    log(viajeros_{l,t}) = a_{l,dow} + c_{l,año} + d_t + b_v · partido_{v,l,t} + e

  * a_{l,dow}  línea x día de la semana -> el perfil semanal PROPIO de cada
                línea. Es el término decisivo: los partidos caen en fin de
                semana y las líneas de oficinas de la Castellana pierden en
                sábado mucho más que la línea media, así que sin este término el
                efecto del Bernabéu salía NEGATIVO (-9 %). El placebo lo detectó.
  * c_{l,año}  línea x año -> deriva propia de cada línea (refuerzos, obras).
  * d_t        día -> todo lo común a la ciudad esa jornada (tipo de día, meteo,
                huelgas, agosto). El contrafactual son las otras ~280 líneas de
                la red el mismo día.
  * partido_{v,l,t} = 1 si la línea l pasa a menos de RADIO_M del recinto v y
    ese día hubo partido en v. Se estima un b_v por recinto (tres aforos muy
    distintos: Bernabéu, Metropolitano y Vallecas), lo que permite ajustar el
    efecto por espectador en lugar de un único número.

Los efectos fijos se absorben por *demeaning* iterativo (proyecciones
alternadas, equivalente a Frisch-Waugh-Lovell); los errores estándar se agrupan
por día, porque el tratamiento y los choques son comunes a todas las líneas de
una misma jornada. No hace falta statsmodels: la OLS va con numpy.

Comprobaciones incluidas: placebo con las fechas desplazadas 7 días (mismo día
de la semana, descartando las que sean partido de verdad), sensibilidad al radio
(600/800/1000 m) y variante ponderada por el tamaño de la línea.

Limitaciones (van a la memoria):
  * Solo LaLiga. Los partidos europeos y de Copa en casa (~8-12 por temporada y
    equipo) quedan en el grupo de control, lo que ATENÚA el efecto estimado: la
    cifra es un límite inferior. Igual ocurre con los conciertos del Bernabéu y
    del Metropolitano.
  * Las asistencias reales no son públicas por partido en una fuente abierta y
    reutilizable; se usa la asistencia media estimada de cada recinto
    (ASISTENCIA), no el dato partido a partido.
  * La granularidad de la demanda es diaria. El paso de efecto diario a factor
    por franja se hace con el perfil intradía (ver `_factor_por_franja`), no
    está medido directamente.

Fuente de partidos: openfootball (dominio público), temporadas 2022-23 a 2025-26,
con fecha y hora de comienzo. La descarga, el parseo y las constantes de los
recintos se comparten con `ingest/contexto.py`, que usa el mismo calendario para
DETECTAR los partidos futuros del cuadro de mando.

Uso:
    python notebooks/calibracion_evento.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # raíz del repo (TFM/)
PROCESSED = ROOT / "datos" / "processed"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "codigo" / "back"))
from ingest.contexto import (ESTADIOS_LALIGA, _descargar_fixtures,  # noqa: E402
                             _franjas_evento, _norm, _parsear_fixtures)

AZUL = "#0178BC"
NARANJA = "#F39200"
GRIS = "#8A9099"

TEMPORADAS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# Los recintos, sus coordenadas y su asistencia estimada viven en
# `ingest/contexto.py` (ESTADIOS_LALIGA): producción y calibración usan las
# mismas constantes, así no pueden divergir.
RECINTOS = ESTADIOS_LALIGA

RADIO_M = 800            # radio principal (el Metropolitano solo tiene 1 línea a 600 m)
RADIOS_SENSIBILIDAD = [600, 800, 1000]
INICIO = "2022-07-01"    # post-COVID y comienzo de la temporada 2022-23
VIAJEROS_MIN = 50        # descarta líneas testimoniales (log inestable)


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def partidos() -> pd.DataFrame:
    """Partidos de LaLiga en Madrid: fecha, hora y recinto.

    Descarga y parseo compartidos con producción (`ingest/contexto.py`), que
    cachea en `datos/raw/partidos/`.
    """
    equipos = {clave: _norm(cfg["equipo"]) for clave, cfg in RECINTOS.items()}
    filas = []
    for t in TEMPORADAS:
        crudo = _descargar_fixtures(t)
        if not crudo:
            raise RuntimeError(f"No hay calendario para la temporada {t}")
        for fecha, hora, local, _visitante in _parsear_fixtures(crudo):
            local_n = _norm(local)
            recinto = next((k for k, eq in equipos.items() if eq in local_n), None)
            if recinto:
                filas.append({"fecha": pd.Timestamp(fecha), "hora": hora,
                              "recinto": recinto, "temporada": t})
    df = pd.DataFrame(filas).drop_duplicates(["fecha", "recinto"])
    return df.sort_values("fecha").reset_index(drop=True)


def lineas_por_recinto(radio_m: float) -> dict[str, list[str]]:
    """Líneas EMT con alguna parada a menos de radio_m de cada recinto."""
    p = pd.read_parquet(PROCESSED / "paradas_todas.parquet")
    lat = np.radians(p["stop_lat"].to_numpy())
    lon = p["stop_lon"].to_numpy()
    out = {}
    for recinto, cfg in RECINTOS.items():
        la, lo = np.radians(cfg["lat"]), cfg["lon"]
        dlat, dlon = lat - la, np.radians(lon - lo)
        a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(lat) * np.sin(dlon / 2) ** 2
        d = 6371000.0 * 2 * np.arcsin(np.sqrt(a))
        out[recinto] = sorted(p.loc[d < radio_m, "linea"].unique().tolist())
    return out


def panel() -> pd.DataFrame:
    """Panel día x línea de demanda diaria, post-COVID."""
    d = pd.read_parquet(PROCESSED / "demanda_diaria_linea.parquet")
    d = d[(d["fecha"] >= INICIO) & (d["viajeros"] >= VIAJEROS_MIN)].copy()
    d["log_viajeros"] = np.log(d["viajeros"])
    # celdas para los efectos fijos: cada línea tiene su propio perfil semanal
    # (una línea de oficinas cae en sábado mucho más que la media de la red) y
    # su propia deriva anual. Sin absorberlos, el estimador confunde "hay
    # partido" con "es fin de semana en esta línea" (lo detecta el placebo).
    d["linea_dow"] = d["linea"] + "_" + d["fecha"].dt.dayofweek.astype(str)
    d["linea_anio"] = d["linea"] + "_" + d["fecha"].dt.year.astype(str)
    return d


# --------------------------------------------------------------------------- #
# Estimación: efectos fijos de dos vías + errores agrupados por día
# --------------------------------------------------------------------------- #
EF = ("fecha", "linea_dow", "linea_anio")   # efectos fijos absorbidos


def _absorber(df: pd.DataFrame, cols: list[str], grupos=EF,
              max_iter: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Quita los efectos fijos de `grupos` por proyecciones alternadas."""
    X = df[cols].to_numpy(dtype=float)
    claves = [df[g].to_numpy() for g in grupos]
    for _ in range(max_iter):
        cambio = 0.0
        for k in claves:
            medias = pd.DataFrame(X).groupby(k).transform("mean").to_numpy()
            X -= medias
            cambio = max(cambio, float(np.abs(medias).max()))
        if cambio < tol:
            break
    return X


def estimar(df: pd.DataFrame, tratamientos: list[str],
            peso: str | None = None) -> pd.DataFrame:
    """OLS/WLS con efectos fijos de línea y día; ES agrupados por día."""
    y = _absorber(df, ["log_viajeros"] + tratamientos)
    Y, X = y[:, :1], y[:, 1:]
    if peso:
        w = np.sqrt(df[peso].to_numpy(dtype=float))[:, None]
        Y, X = Y * w, X * w
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = (XtX_inv @ X.T @ Y).ravel()
    u = (Y - X @ beta[:, None]).ravel()

    # varianza agrupada por día (los choques diarios son comunes a la red)
    meat = np.zeros((X.shape[1], X.shape[1]))
    codigos = pd.factorize(df["fecha"])[0]
    for g in range(codigos.max() + 1):
        m = codigos == g
        s = (X[m] * u[m, None]).sum(axis=0)[:, None]
        meat += s @ s.T
    G = codigos.max() + 1
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1))
    se = np.sqrt(np.diag(V))

    return pd.DataFrame({
        "termino": tratamientos,
        "coef": beta,
        "se": se,
        "t": beta / se,
        "efecto_pct": (np.exp(beta) - 1) * 100,
        "ic95_inf_pct": (np.exp(beta - 1.96 * se) - 1) * 100,
        "ic95_sup_pct": (np.exp(beta + 1.96 * se) - 1) * 100,
        "n_dias_tratados": [int(df.loc[df[t] == 1, "fecha"].nunique())
                            for t in tratamientos],
        "n_obs_tratadas": [int((df[t] == 1).sum()) for t in tratamientos],
    })


def preparar(dem: pd.DataFrame, part: pd.DataFrame, radio_m: float,
             desplazamiento_dias: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Añade al panel una columna dummy por recinto.

    Con `desplazamiento_dias` se construye el placebo: las mismas líneas en la
    misma fecha corrida n días. Se descartan las fechas desplazadas que sean
    partido de verdad en ese recinto (la liga es semanal y el desplazamiento
    puede caer en otra jornada en casa, contaminando el placebo).
    """
    lineas = lineas_por_recinto(radio_m)
    df = dem.copy()
    cols = []
    for recinto, ls in lineas.items():
        reales = set(part.loc[part["recinto"] == recinto, "fecha"])
        fechas = {f + pd.Timedelta(days=desplazamiento_dias) for f in reales}
        if desplazamiento_dias:
            fechas -= reales
        col = f"partido_{recinto}"
        df[col] = (df["linea"].isin(ls) & df["fecha"].isin(fechas)).astype(float)
        cols.append(col)
    return df, cols


# --------------------------------------------------------------------------- #
# De efecto diario a factor por franja
# --------------------------------------------------------------------------- #
def _factor_por_franja(efecto_diario: float, lineas: list[str],
                       horas: pd.Series) -> tuple[float, float]:
    """Reparte el efecto diario en las franjas del evento.

    El factor de `contexto.py` se aplica a la demanda de las franjas afectadas,
    no al día completo. Si esas franjas concentran una fracción `s` de la
    demanda diaria, un efecto diario `u` equivale a un factor `1 + u/s` sobre
    ellas.

    `s` se calcula POR HORA de comienzo (que es como se aplica en producción) y
    se promedia con la frecuencia observada de cada horario; usar la unión de
    todas las franjas posibles infravaloraría el factor. Devuelve (s, factor).
    """
    perfil = pd.read_parquet(PROCESSED / "perfil_intradia.parquet")
    perfil = perfil[perfil["tipo_dia"].isin(["LA", "SA", "FE"])]
    # el perfil usa la línea con ceros por delante ('014'), el GTFS no ('14')
    codigos = {l.zfill(3) for l in lineas} | set(lineas)
    perfil = perfil[perfil["linea"].isin(codigos)]
    if perfil.empty:
        return float("nan"), float("nan")
    frecuencias = horas.value_counts(normalize=True)
    pesos = []
    for hora, frec in frecuencias.items():
        franjas = _franjas_evento(int(hora))
        s_h = (perfil[perfil["franja"].isin(franjas)]
               .groupby(["linea", "tipo_dia"])["share"].sum().mean())
        pesos.append(frec * s_h)
    peso = float(sum(pesos))
    return peso, 1 + efecto_diario / peso


# --------------------------------------------------------------------------- #
def main() -> None:
    part = partidos()
    dem = panel()
    print(f"Panel: {len(dem):,} obs · {dem['linea'].nunique()} líneas · "
          f"{dem['fecha'].nunique()} días ({dem['fecha'].min().date()} → "
          f"{dem['fecha'].max().date()})")
    print("\nPartidos por recinto y temporada:")
    print(part.pivot_table(index="recinto", columns="temporada",
                           values="fecha", aggfunc="count").to_string())
    for r, ls in lineas_por_recinto(RADIO_M).items():
        print(f"\nLíneas a <{RADIO_M} m de {RECINTOS[r]['nombre']} "
              f"({len(ls)}): {', '.join(ls)}")

    # --- estimación principal --------------------------------------------- #
    df, cols = preparar(dem, part, RADIO_M)
    res = estimar(df, cols)
    print("\n=== Efecto de un partido sobre la demanda diaria de las líneas "
          "cercanas (DiD, EF de línea y día) ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # --- placebo: mismas líneas, fechas +7 días --------------------------- #
    df_p, cols_p = preparar(dem, part, RADIO_M, desplazamiento_dias=7)
    res_p = estimar(df_p, cols_p)
    print("\n=== Placebo (fechas desplazadas 7 días) ===")
    print(res_p[["termino", "efecto_pct", "ic95_inf_pct", "ic95_sup_pct", "t"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # --- sensibilidad al radio y ponderación ------------------------------ #
    print("\n=== Sensibilidad al radio ===")
    sens = []
    for r in RADIOS_SENSIBILIDAD:
        d_r, c_r = preparar(dem, part, r)
        e = estimar(d_r, c_r)[["termino", "efecto_pct", "t"]]
        e["radio_m"] = r
        sens.append(e)
    sens = pd.concat(sens)
    print(sens.pivot(index="termino", columns="radio_m",
                     values="efecto_pct").to_string(float_format=lambda x: f"{x:6.2f}"))

    medias = dem.groupby("linea")["viajeros"].mean().rename("peso")
    df_w = df.merge(medias, on="linea")
    res_w = estimar(df_w, cols, peso="peso")
    print("\n=== Ponderado por tamaño de línea ===")
    print(res_w[["termino", "efecto_pct", "t"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # --- calibración del factor ------------------------------------------- #
    # Lectura conservadora: al efecto se le descuenta el placebo positivo. Un
    # placebo distinto de cero indica que el grupo de control contiene días de
    # evento que no están en la fuente (Champions, Copa, conciertos), así que
    # parte del "efecto" podría ser suyo.
    lineas = lineas_por_recinto(RADIO_M)
    placebo = dict(zip(res_p["termino"], res_p["efecto_pct"]))
    filas = []
    for _, r in res.iterrows():
        recinto = r["termino"].replace("partido_", "")
        horas = part.loc[part["recinto"] == recinto, "hora"]
        signif = abs(r["t"]) > 1.96
        pl = placebo[r["termino"]]
        conservador = max(0.0, r["efecto_pct"] - max(0.0, pl)) if signif else 0.0
        peso, factor = _factor_por_franja(conservador / 100, lineas[recinto], horas)
        filas.append({
            "recinto": RECINTOS[recinto]["nombre"],
            "asistencia": RECINTOS[recinto]["asistencia"],
            "efecto_diario_pct": r["efecto_pct"],
            "ic95_inf_pct": r["ic95_inf_pct"],
            "ic95_sup_pct": r["ic95_sup_pct"],
            "t": r["t"],
            "significativo": signif,
            "placebo_pct": pl,
            "efecto_conservador_pct": conservador,
            "peso_franjas_evento": peso,
            "factor_franja": factor,
            "n_partidos": int(r["n_dias_tratados"]),
            "n_lineas": len(lineas[recinto]),
        })
    cal = pd.DataFrame(filas)
    print("\n=== De efecto diario a factor por franja (lectura conservadora) ===")
    print(cal[["recinto", "asistencia", "efecto_diario_pct", "placebo_pct",
               "efecto_conservador_pct", "peso_franjas_evento", "factor_franja"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # Ley calibrada: factor = 1 + max(0, asistencia - UMBRAL) / DIVISOR
    #   UMBRAL  = mayor aforo con efecto NO detectable (por debajo, no se aplica
    #             nada: ni un concierto pequeño ni una actividad municipal).
    #   DIVISOR = recta por el origen sobre los recintos con efecto detectable.
    no_sig = cal[~cal["significativo"]]
    umbral = float(no_sig["asistencia"].max()) if not no_sig.empty else 0.0
    sig = cal[cal["significativo"]]
    a = (sig["asistencia"] - umbral).to_numpy(dtype=float)
    u = (sig["factor_franja"] - 1).to_numpy(dtype=float)
    k = float((a * u).sum() / (a * a).sum())
    divisor = 1 / k
    tope = float(sig["factor_franja"].max())
    print(f"\nUmbral (mayor aforo sin efecto detectable): {umbral:,.0f}")
    print(f"Ley calibrada: factor = 1 + max(0, asistencia - {umbral:,.0f}) / "
          f"{divisor:,.0f}   (tope observado x{tope:.2f})")
    print("\nAsistencia →  factor calibrado   vs   factor sin calibrar")
    for asis in (3000, 13500, 20000, 40000, 57000, 60000, 70000, 78000):
        nuevo = min(1 + max(0.0, asis - umbral) / divisor, tope)
        viejo = 1 + min(0.9, asis / 75000)
        print(f"  {asis:>6,}      x{nuevo:.3f}              x{viejo:.2f}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(PROCESSED / "calibracion_evento.parquet", index=False)
    res.assign(tipo="principal").to_csv(
        PROCESSED / "calibracion_evento_regresion.csv", index=False)
    (PROCESSED / "calibracion_evento_ley.json").write_text(
        json.dumps({"umbral": umbral, "divisor": divisor, "tope": tope,
                    "radio_m": RADIO_M, "n_temporadas": len(TEMPORADAS)},
                   indent=2), encoding="utf-8")
    print(f"\nGuardado {PROCESSED / 'calibracion_evento.parquet'} (+ .csv, ley .json)")

    # --- figura ------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(cal))
    err = [(cal["efecto_diario_pct"] - cal["ic95_inf_pct"]).to_numpy(),
           (cal["ic95_sup_pct"] - cal["efecto_diario_pct"]).to_numpy()]
    ax.barh(y + 0.18, cal["efecto_diario_pct"], color=AZUL, height=0.34,
            xerr=err, error_kw={"ecolor": GRIS, "capsize": 3, "lw": 1},
            label="Día de partido")
    ax.barh(y - 0.18, cal["placebo_pct"], color=NARANJA, height=0.34,
            label="Placebo (mismo día, 7 días después)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.recinto}\n{r.asistencia:,} espect. · "
                        f"{r.n_partidos} partidos · {r.n_lineas} líneas"
                        .replace(",", ".") for r in cal.itertuples()], fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Efecto sobre la demanda diaria de las líneas cercanas (%)")
    ax.set_title("Impacto medido de un partido en las líneas de bus cercanas\n"
                 "Diferencias en diferencias, efectos fijos de línea×día-semana,\n"
                 "línea×año y día · IC 95 % agrupado por día", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "calibracion_evento.png", dpi=150)
    print(f"Figura: {FIGS / 'calibracion_evento.png'}")


if __name__ == "__main__":
    main()
