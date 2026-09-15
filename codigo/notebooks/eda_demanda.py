"""EDA de la demanda diaria de viajeros por línea — EMT Madrid (Fase 1).

Explora datos/processed/demanda_diaria_linea.parquet (generado por
ingest/emt_opendata.py) para:
  - describir el histórico (rango, cobertura, calidad),
  - caracterizar estacionalidad (día de semana, mes, efecto COVID),
  - proponer 10-15 líneas piloto por volumen y variabilidad de demanda.

Imprime un resumen por consola y guarda figuras en notebooks/figs/.

Uso:
    python notebooks/eda_demanda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # raíz del repo (TFM/)
PARQUET = ROOT / "datos" / "processed" / "demanda_diaria_linea.parquet"
RAW_CSV = ROOT / "datos" / "raw" / "demandadialinea.csv"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

AZUL = "#0178BC"
NARANJA = "#F39200"

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def cargar() -> pd.DataFrame:
    """Carga el parquet normalizado; si no existe, lee el CSV crudo."""
    if PARQUET.exists():
        return pd.read_parquet(PARQUET)
    assert RAW_CSV.exists(), (
        f"No encuentro {PARQUET} ni {RAW_CSV}. "
        f"Ejecuta antes: python ingest/emt_opendata.py --demanda"
    )
    df = pd.read_csv(RAW_CSV, sep=";", encoding="utf-8-sig", dtype=str)
    df = df.rename(columns={"Fecha": "fecha", "Linea": "linea", "TotalViajeros": "viajeros"})
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df["viajeros"] = pd.to_numeric(df["viajeros"], errors="coerce")
    return df.dropna(subset=["fecha", "linea", "viajeros"])


def main() -> None:
    df = cargar()
    df["viajeros"] = df["viajeros"].astype(int)

    print("=" * 64)
    print("EDA — DEMANDA DIARIA DE VIAJEROS POR LÍNEA (EMT MADRID)")
    print("=" * 64)
    print(f"Filas:            {len(df):>12,}")
    print(f"Líneas distintas: {df['linea'].nunique():>12}")
    print(f"Rango temporal:   {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    dias_cubiertos = df['fecha'].nunique()
    print(f"Días distintos:   {dias_cubiertos:>12,}")
    print(f"Viajeros/día (media global): {df.groupby('fecha')['viajeros'].sum().mean():>15,.0f}")

    # --- Serie temporal agregada (toda la red) ---
    total_diario = df.groupby("fecha")["viajeros"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    total_diario.plot(ax=ax, color=AZUL, lw=0.8)
    total_diario.rolling(30, min_periods=7).mean().plot(ax=ax, color=NARANJA, lw=1.8,
                                                         label="media móvil 30d")
    ax.set_title("Demanda diaria total de la red EMT (todas las líneas)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Viajeros/día")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "demanda_total_diaria.png", dpi=120)
    plt.close()

    # --- Estacionalidad: día de semana y mes ---
    df["dow"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    total_por_dia = df.groupby(["fecha"]).agg(viajeros=("viajeros", "sum"),
                                              dow=("dow", "first"),
                                              mes=("mes", "first"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    total_por_dia.groupby("dow")["viajeros"].mean().reindex(range(7)).plot(
        kind="bar", ax=axes[0], color=AZUL)
    axes[0].set_title("Demanda media por día de la semana")
    axes[0].set_xticklabels(DIAS, rotation=0)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Viajeros/día (media)")
    total_por_dia.groupby("mes")["viajeros"].mean().reindex(range(1, 13)).plot(
        kind="bar", ax=axes[1], color=AZUL)
    axes[1].set_title("Demanda media por mes")
    axes[1].set_xlabel("Mes")
    plt.tight_layout()
    plt.savefig(FIGS / "demanda_estacionalidad.png", dpi=120)
    plt.close()

    # --- Selección de líneas piloto (ventana reciente, evita COVID) ---
    corte = df["fecha"].max() - pd.Timedelta(days=730)  # últimos ~24 meses
    reciente = df[df["fecha"] >= corte]
    print(f"\nVentana para selección piloto: {corte.date()} → {df['fecha'].max().date()}")

    stats = (
        reciente.groupby("linea")["viajeros"]
        .agg(dias="count", media="mean", std="std", p95=lambda s: s.quantile(0.95))
        .assign(cv=lambda x: x["std"] / x["media"])
        .query("dias >= 300")  # líneas con cobertura suficiente
    )
    # Nos interesan líneas relevantes (volumen alto) y con demanda variable
    # (donde optimizar la frecuencia tiene más impacto).
    umbral_vol = stats["media"].quantile(0.60)
    candidatas = (
        stats[stats["media"] >= umbral_vol]
        .sort_values("cv", ascending=False)
        .head(15)
    )

    print("\nTop 15 líneas CANDIDATAS A PILOTO "
          "(volumen alto + mayor variabilidad de demanda):")
    print(candidatas[["media", "std", "cv", "p95", "dias"]]
          .round({"media": 0, "std": 0, "cv": 3, "p95": 0})
          .to_string())

    # --- Gráfica: volumen vs variabilidad, con candidatas resaltadas ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(stats["media"], stats["cv"], s=12, alpha=0.4, color="grey",
               label="resto de líneas")
    ax.scatter(candidatas["media"], candidatas["cv"], s=40, color=NARANJA,
               label="candidatas piloto")
    for linea, row in candidatas.iterrows():
        ax.annotate(linea, (row["media"], row["cv"]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title("Selección de líneas piloto: volumen vs variabilidad (24 meses)")
    ax.set_xlabel("Demanda media diaria (viajeros)")
    ax.set_ylabel("Coeficiente de variación (std/media)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "seleccion_piloto.png", dpi=120)
    plt.close()

    print(f"\nFiguras guardadas en: {FIGS}")
    for f in ["demanda_total_diaria.png", "demanda_estacionalidad.png",
              "seleccion_piloto.png"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
