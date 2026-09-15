"""Entrenamiento del baseline de predicción de demanda — Módulo 1.

Modelo LightGBM sobre el dataset diario por línea (datos/processed/
dataset_features.parquet), evaluado con validación WALK-FORWARD (nunca se usan
datos futuros para entrenar). Se compara contra un baseline naive (misma
demanda que el mismo día de la semana anterior, lag_7).

Métricas: MAE, RMSE y MAPE por fold y agregadas.
Artefactos: figuras de predicción vs real e importancia de features en
notebooks/figs/.

Uso:
    python demand/train.py --config configs/lgbm.yaml
    python -m demand.train --config configs/lgbm.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fuente única de verdad en common.py (antes se duplicaban aquí y divergían).
from demand.common import (folds_walk_forward, load_config, make_lgbm,
                           metricas)

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT.parent / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena baseline de demanda")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"[train] Experimento: {cfg['experiment']['name']}")

    df = pd.read_parquet(ROOT.parents[1] / cfg["data"]["processed_path"])
    fmin = cfg["data"].get("fecha_min")
    if fmin:
        df = df[df["fecha"] >= pd.Timestamp(fmin)]
    df = df.sort_values("fecha").reset_index(drop=True)
    target = cfg["data"]["target"]

    f = cfg["features"]
    cat = f["categorical"]
    feat_cols = (f["calendar"] + f["weather"] + f["line_attr"]
                 + f["lags"] + f["rolling"] + cat)
    feat_cols = [c for c in feat_cols if c in df.columns]
    for c in cat:
        df[c] = df[c].astype("category")

    print(f"[train] {len(df):,} filas | {df['fecha'].min().date()} → "
          f"{df['fecha'].max().date()} | {len(feat_cols)} features")

    cortes = folds_walk_forward(df["fecha"], cfg["split"]["n_folds"],
                                cfg["split"]["test_period_days"])
    m = cfg["model"]
    resultados = []
    ult_pred = None
    for i, (t_ini, t_fin) in enumerate(cortes, 1):
        tr = df[df["fecha"] < t_ini]
        te = df[(df["fecha"] >= t_ini) & (df["fecha"] <= t_fin)]
        if te.empty:
            continue
        modelo = make_lgbm(m)
        modelo.fit(tr[feat_cols], tr[target], categorical_feature=cat)
        pred = modelo.predict(te[feat_cols])

        mk = metricas(te[target].to_numpy(), pred)
        mk_naive = metricas(te[target].to_numpy(), te["lag_7"].to_numpy())
        resultados.append({"fold": i, "test": f"{t_ini.date()}→{t_fin.date()}",
                           "n": len(te), **{k: round(v, 2) for k, v in mk.items()},
                           "MAE_naive": round(mk_naive["MAE"], 2)})
        ult_pred = (te.assign(pred=pred), modelo)

    res = pd.DataFrame(resultados)
    print("\n=== Walk-forward (LightGBM vs baseline naive lag_7) ===")
    print(res.to_string(index=False))
    print("\nMedia LightGBM  -> "
          f"MAE={res['MAE'].mean():.1f} | RMSE={res['RMSE'].mean():.1f} | "
          f"MAPE={res['MAPE'].mean():.1f}%")
    print(f"Media naive lag_7 -> MAE={res['MAE_naive'].mean():.1f}")
    mejora = (1 - res["MAE"].mean() / res["MAE_naive"].mean()) * 100
    print(f"Mejora del LightGBM sobre el naive: {mejora:.1f}% en MAE")

    # --- Figura 1: predicción vs real (última ventana, una línea de ejemplo) ---
    te_pred, modelo = ult_pred
    linea_ej = te_pred.groupby("linea")["viajeros"].mean().idxmax()
    sub = te_pred[te_pred["linea"] == linea_ej].sort_values("fecha")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sub["fecha"], sub["viajeros"], label="real", color="#0178BC", lw=1.6)
    ax.plot(sub["fecha"], sub["pred"], label="predicción LightGBM", color="#F39200",
            lw=1.6, ls="--")
    ax.set_title(f"Demanda real vs predicha — línea {linea_ej} (última ventana test)")
    ax.set_ylabel("Viajeros/día")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "forecast_pred_vs_real.png", dpi=120)
    plt.close()

    # --- Figura 2: importancia de features ---
    imp = pd.Series(modelo.feature_importances_, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="#0178BC")
    ax.set_title("Importancia de features (LightGBM)")
    plt.tight_layout()
    plt.savefig(FIGS / "forecast_feature_importance.png", dpi=120)
    plt.close()

    print(f"\nFiguras en {FIGS}: forecast_pred_vs_real.png, "
          f"forecast_feature_importance.png")


if __name__ == "__main__":
    main()
