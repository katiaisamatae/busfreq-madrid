"""Mini-EDA de saturación de líneas — grado de ocupación EMT (Fase 1).

Usa datos/processed/ocupacion_linea_mes.parquet (ingest/emt_opendata.py
--ocupacion). El dato es el % de viajes que cumplen el criterio de calidad de
ocupación (≈100 = holgado). Definimos:

    saturacion = 100 - pct_calidad_ocupacion    (mayor = más aglomeración)

Responde: ¿qué líneas van más llenas?, ¿cómo evoluciona por año (efecto COVID)?,
¿las líneas piloto capturan líneas saturadas?

Uso:
    python notebooks/eda_saturacion.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # raíz del repo (TFM/)
OCUP = ROOT / "datos" / "processed" / "ocupacion_linea_mes.parquet"
PILOTO = ROOT / "datos" / "processed" / "lineas_piloto.csv"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

AZUL = "#0178BC"
NARANJA = "#F39200"


def main() -> None:
    assert OCUP.exists(), (
        f"No encuentro {OCUP}. Ejecuta: python -m ingest.emt_opendata --ocupacion")
    df = pd.read_parquet(OCUP)
    df["saturacion"] = 100 - df["pct_calidad_ocupacion"]

    print("=" * 66)
    print("MINI-EDA — SATURACIÓN DE LÍNEAS (grado de ocupación EMT)")
    print("=" * 66)
    print(f"Filas:    {len(df):>10,}")
    print(f"Líneas:   {df['linea'].nunique():>10}")
    print(f"Años:     {sorted(df['anio'].unique())}")
    print(f"Saturación media global: {df['saturacion'].mean():.2f} "
          f"(0 = nunca lleno, 100 = siempre lleno)")

    # --- Evolución anual del % de calidad (efecto COVID) ---
    por_anio = df.groupby("anio")["pct_calidad_ocupacion"].mean()
    print("\n% medio de viajes SIN aglomeración por año:")
    print(por_anio.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    por_anio.plot(kind="bar", ax=ax, color=AZUL)
    ax.set_title("% medio de viajes que cumplen el criterio de ocupación, por año")
    ax.set_ylabel("% calidad (mayor = más holgado)")
    ax.set_xlabel("Año")
    ax.set_ylim(90, 100)
    plt.tight_layout()
    plt.savefig(FIGS / "saturacion_por_anio.png", dpi=120)
    plt.close()

    # --- Top líneas más saturadas (media años recientes 2023-2024) ---
    recientes = df[df["anio"] >= 2023]
    ranking = (recientes.groupby(["linea", "denominacion"])["saturacion"]
               .mean().sort_values(ascending=False))
    top = ranking.head(20)
    print("\nTop 20 líneas MÁS saturadas (media 2023-2024, mayor = más llena):")
    for (linea, denom), sat in top.items():
        print(f"  {linea:>4}  sat={sat:5.2f}  {denom[:44]}")

    fig, ax = plt.subplots(figsize=(9, 7))
    etiquetas = [f"{l}  {d[:26]}" for (l, d) in top.index[::-1]]
    ax.barh(etiquetas, top.values[::-1], color=NARANJA)
    ax.set_title("Top 20 líneas más saturadas (media 2023-2024)")
    ax.set_xlabel("Saturación = 100 − % calidad ocupación")
    plt.tight_layout()
    plt.savefig(FIGS / "saturacion_top_lineas.png", dpi=120)
    plt.close()

    # --- Cruce con líneas piloto ---
    if PILOTO.exists():
        piloto = pd.read_csv(PILOTO, dtype={"linea": str})["linea"].tolist()
        sat_linea = recientes.groupby("linea")["saturacion"].mean()
        media = sat_linea.mean()
        print(f"\nSaturación media (todas las líneas): {media:.2f}")
        print("Saturación de las 12 líneas piloto:")
        tabla = (sat_linea[sat_linea.index.isin(piloto)]
                 .sort_values(ascending=False))
        for linea, sat in tabla.items():
            marca = "↑ por encima de la media" if sat > media else ""
            print(f"  {linea:>4}  sat={sat:5.2f}  {marca}")
        n_arriba = (tabla > media).sum()
        print(f"→ {n_arriba}/{len(tabla)} líneas piloto están por encima de la "
              f"saturación media de la red.")

    print(f"\nFiguras guardadas en {FIGS}:")
    for f in ["saturacion_por_anio.png", "saturacion_top_lineas.png"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
