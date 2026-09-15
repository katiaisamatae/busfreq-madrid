"""Predicciones out-of-sample del Módulo 1 para el cuadro de mando.

Genera, con validación walk-forward, las predicciones de demanda diaria por
línea sobre los folds de prueba y las guarda para que el dashboard muestre
"predicho vs real" y el error por línea. Añade una banda de incertidumbre
(cuantiles 0,1 y 0,9 con LightGBM quantile).

Separa la métrica de SELECCIÓN (folds de tuning) de la métrica HONESTA
out-of-sample (test_fold intocado), y guarda un desglose citable en la memoria:
MAE/RMSE/MAPE/WAPE agregado, por fold (con desviación), por tipo de día y por
línea.

Salidas:
  datos/processed/predicciones.parquet
      [fecha, linea, real, pred, pred_p10, pred_p90, fold, tipo_dia]
  datos/processed/metricas_finales.parquet  (+ .csv)

Uso:
    python demand/predict.py --config configs/lgbm_tuned.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demand.common import (clasificar_tipo_dia, feature_cols,
                           folds_walk_forward, load_config, load_dataset,
                           make_lgbm, metricas)

ROOT = Path(__file__).resolve().parent.parent


def _fit_pred(m: dict, tr, te, feat, cat) -> np.ndarray:
    modelo = make_lgbm(m)
    modelo.fit(tr[feat], tr["viajeros"], categorical_feature=cat)
    return modelo.predict(te[feat])


def _tabla(pred: pd.DataFrame, col_grupo: str | None = None) -> pd.DataFrame:
    """MAE/RMSE/MAPE/WAPE (+ n) agregadas o desglosadas por `col_grupo`."""
    if col_grupo is None:
        d = metricas(pred["real"].to_numpy(), pred["pred"].to_numpy())
        return pd.DataFrame([{**d, "n": len(pred)}])
    filas = []
    for val, g in pred.groupby(col_grupo, observed=True):
        d = metricas(g["real"].to_numpy(), g["pred"].to_numpy())
        filas.append({col_grupo: val, **d, "n": len(g)})
    return pd.DataFrame(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicciones walk-forward")
    parser.add_argument("--config", default="configs/lgbm_tuned.yaml")
    args = parser.parse_args()

    cfg = load_config(str(ROOT / args.config) if not Path(args.config).is_absolute()
                      else args.config)
    df = load_dataset(cfg)
    feat = feature_cols(cfg, df)
    cat = cfg["features"]["categorical"]
    for c in cat:
        df[c] = df[c].astype("category")
    cortes = folds_walk_forward(df["fecha"], cfg["split"]["n_folds"],
                                cfg["split"]["test_period_days"])
    m = cfg["model"]

    partes = []
    for i, (t_ini, t_fin) in enumerate(cortes, 1):
        tr = df[df["fecha"] < t_ini]
        te = df[(df["fecha"] >= t_ini) & (df["fecha"] <= t_fin)]
        if te.empty:
            continue
        out = te[["fecha", "linea", "viajeros"]].copy()
        out["pred"] = np.round(_fit_pred(m, tr, te, feat, cat)).astype(int)
        # Banda de incertidumbre: LightGBM quantile 0,1 y 0,9 (misma config).
        for alpha, col in ((0.1, "pred_p10"), (0.9, "pred_p90")):
            mq = {**m, "objective": "quantile", "alpha": alpha, "metric": "quantile"}
            out[col] = np.round(_fit_pred(mq, tr, te, feat, cat)).astype(int)
        out["fold"] = i
        partes.append(out)
        print(f"[predict] fold {i} ({t_ini.date()}→{t_fin.date()}): {len(te)} filas")

    pred = pd.concat(partes, ignore_index=True).rename(columns={"viajeros": "real"})
    # Evita cruces de cuantiles: p10 <= pred <= p90.
    pred["pred_p10"] = np.minimum(pred["pred_p10"], pred["pred"])
    pred["pred_p90"] = np.maximum(pred["pred_p90"], pred["pred"])
    pred["tipo_dia"] = clasificar_tipo_dia(pred["fecha"])
    pred = pred.sort_values(["linea", "fecha"]).reset_index(drop=True)

    destino = ROOT.parents[1] / "datos" / "processed" / "predicciones.parquet"
    pred.to_parquet(destino, index=False)

    # --- Métricas: selección (folds de tuning) vs honesta (test_fold) ---
    n_folds = cfg["split"]["n_folds"]
    test_fold = cfg["split"].get("test_fold", n_folds)
    tune_folds = cfg["split"].get("tune_folds", list(range(1, n_folds + 1)))

    agg = _tabla(pred)
    seleccion = _tabla(pred[pred["fold"].isin(tune_folds)])
    honesta = _tabla(pred[pred["fold"] == test_fold])
    por_fold = _tabla(pred, "fold")
    por_tipo = _tabla(pred, "tipo_dia")
    por_linea = _tabla(pred, "linea").sort_values("MAPE", ascending=False)

    dentro = (pred["real"] >= pred["pred_p10"]) & (pred["real"] <= pred["pred_p90"])
    cobertura = float(dentro.mean() * 100)

    def _fmt(r):
        return (f"MAE={r['MAE']:.1f}  RMSE={r['RMSE']:.1f}  MAPE={r['MAPE']:.2f}%  "
                f"WAPE={r['WAPE']:.2f}%  (n={int(r['n'])})")

    print(f"\n[predict] {len(pred):,} predicciones | {pred['linea'].nunique()} líneas | "
          f"{pred['fecha'].min().date()} → {pred['fecha'].max().date()}")
    print(f"\n=== SELECCIÓN (folds {tune_folds}, NO out-of-sample) ===\n  "
          + _fmt(seleccion.iloc[0]))
    print(f"\n=== HONESTA out-of-sample (test_fold {test_fold}, intocado) ===\n  "
          + _fmt(honesta.iloc[0]))
    print("\n--- Por fold ---")
    print(por_fold.round(2).to_string(index=False))
    s = por_fold[["MAE", "MAPE", "WAPE"]].std()
    print(f"  desviación entre folds: MAE±{s['MAE']:.1f}  MAPE±{s['MAPE']:.2f}%  "
          f"WAPE±{s['WAPE']:.2f}%")
    print("\n--- Por tipo de día ---")
    print(por_tipo.round(2).to_string(index=False))
    print(f"\n[predict] cobertura del intervalo p10-p90: {cobertura:.1f}% (objetivo ~80%)")

    # --- Desglose citable (tidy) ---
    bloques = []
    for nivel, t, grp in (("agregado", agg, None), ("seleccion", seleccion, None),
                          ("test_fold", honesta, None), ("fold", por_fold, "fold"),
                          ("tipo_dia", por_tipo, "tipo_dia"), ("linea", por_linea, "linea")):
        t = t.copy()
        t["nivel"] = nivel
        t["grupo"] = t[grp].astype(str) if grp else nivel
        bloques.append(t[["nivel", "grupo", "MAE", "RMSE", "MAPE", "WAPE", "n"]])
    met = pd.concat(bloques, ignore_index=True)
    met_dest = ROOT.parents[1] / "datos" / "processed" / "metricas_finales.parquet"
    met.to_parquet(met_dest, index=False)
    met.to_csv(met_dest.with_suffix(".csv"), index=False, encoding="utf-8")

    print(f"\n[predict] predicciones en {destino}")
    print(f"[predict] métricas finales en {met_dest}")


if __name__ == "__main__":
    main()
