"""Ajuste de hiperparámetros de LightGBM — Fase 2 (Módulo 1).

Búsqueda aleatoria evaluada con los MISMOS folds walk-forward que el resto del
modelado (sin filtrar información futura). El criterio de selección es el MAE
medio sobre los bloques de prueba.

Escribe el mejor config en configs/lgbm_tuned.yaml, registra los ensayos en
MLflow y guarda notebooks/figs/tuning_lgbm.png.

Uso:
    python demand/tune.py --config configs/lgbm.yaml --trials 30
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demand.common import (feature_cols, folds_walk_forward, load_config,
                           load_dataset, make_lgbm, metricas, seleccionar_folds)

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT.parent / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

ESPACIO = {
    "num_leaves": [15, 31, 47, 63, 95, 127],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "n_estimators": [600, 1000, 1500, 2000],
    "min_child_samples": [10, 20, 30, 50, 80],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.0, 0.1, 0.5, 1.0],
}


def evaluar(df, feat, cat, cortes, m) -> float:
    """MAE medio walk-forward para una configuración de modelo."""
    maes = []
    for t_ini, t_fin in cortes:
        tr = df[df["fecha"] < t_ini]
        te = df[(df["fecha"] >= t_ini) & (df["fecha"] <= t_fin)]
        if te.empty:
            continue
        modelo = make_lgbm(m)
        modelo.fit(tr[feat], tr["viajeros"], categorical_feature=cat)
        maes.append(metricas(te["viajeros"].to_numpy(), modelo.predict(te[feat]))["MAE"])
    return float(np.mean(maes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ajuste de hiperparámetros LightGBM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_dataset(cfg)
    feat = feature_cols(cfg, df)
    cat = cfg["features"]["categorical"]
    for c in cat:
        df[c] = df[c].astype("category")
    cortes = folds_walk_forward(df["fecha"], cfg["split"]["n_folds"],
                                cfg["split"]["test_period_days"])
    # Test final intocado: se ajusta SOLO con tune_folds; el test_fold queda
    # reservado para la métrica out-of-sample honesta (demand/predict.py).
    tune_idx = cfg["split"].get("tune_folds")
    if tune_idx:
        cortes = seleccionar_folds(cortes, tune_idx)
        print(f"[tune] folds de selección: {tune_idx} | test_fold reservado: "
              f"{cfg['split'].get('test_fold')}")

    base_m = cfg["model"]
    fijos = {"objective": base_m["objective"], "metric": base_m.get("metric", "mae"),
             "subsample_freq": 1, "random_state": base_m["random_state"]}
    base_mae = evaluar(df, feat, cat, cortes, base_m)
    print(f"[tune] MAE base (config actual): {base_mae:.2f}")
    print(f"[tune] {args.trials} ensayos de búsqueda aleatoria...")

    rng = np.random.default_rng(base_m["random_state"])
    ensayos = []
    for i in range(args.trials):
        muestra = {k: rng.choice(v).item() for k, v in ESPACIO.items()}
        m = {**fijos, **muestra}
        mae = evaluar(df, feat, cat, cortes, m)
        ensayos.append({"trial": i + 1, "mae": mae, **muestra})
        mejor = min(e["mae"] for e in ensayos)
        print(f"  ensayo {i + 1:>2}/{args.trials}  MAE={mae:8.2f}  (mejor={mejor:8.2f})")

    res = pd.DataFrame(ensayos).sort_values("mae").reset_index(drop=True)
    best = res.iloc[0]
    print("\n=== Top 5 configuraciones ===")
    print(res.head(5).round(3).to_string(index=False))
    mejora = (1 - best["mae"] / base_mae) * 100
    print(f"\n[tune] mejor MAE={best['mae']:.2f} vs base {base_mae:.2f} "
          f"({mejora:+.1f}%)")

    # --- Guarda el config afinado ---
    cfg_tuned = copy.deepcopy(cfg)
    cfg_tuned["experiment"]["name"] = "demand_lgbm_tuned"
    cfg_tuned["model"] = {**fijos,
                          **{k: (int(best[k]) if k in ("num_leaves", "n_estimators",
                                                       "min_child_samples")
                                 else float(best[k])) for k in ESPACIO}}
    salida_cfg = ROOT / "configs" / "lgbm_tuned.yaml"
    with open(salida_cfg, "w", encoding="utf-8") as f:
        f.write("# Config de LightGBM con hiperparámetros afinados (demand/tune.py).\n")
        f.write(f"# MAE de SELECCIÓN sobre folds de tuning: {best['mae']:.2f} "
                f"(base {base_mae:.2f}, {mejora:+.1f}%). La métrica HONESTA\n"
                f"# out-of-sample es la del test_fold intocado (ver demand/predict.py).\n")
        yaml.safe_dump(cfg_tuned, f, sort_keys=False, allow_unicode=True)
    print(f"[tune] config afinado en {salida_cfg}")

    # --- MLflow ---
    try:
        import mlflow
        dbpath = (ROOT / "mlflow.db").resolve().as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{dbpath}")
        mlflow.set_experiment("demand_lgbm_tuning")
        for e in ensayos:
            with mlflow.start_run(run_name=f"trial_{e['trial']}"):
                mlflow.log_params({k: e[k] for k in ESPACIO})
                mlflow.log_metric("MAE", e["mae"])
        print("[tune] ensayos registrados en MLflow")
    except Exception as ex:  # noqa: BLE001
        print(f"[tune] MLflow omitido ({ex.__class__.__name__})")

    # --- Figura: evolución de la búsqueda ---
    fig, ax = plt.subplots(figsize=(9, 4))
    orden = pd.DataFrame(ensayos)
    mejor_hist = orden["mae"].cummin()
    ax.scatter(orden["trial"], orden["mae"], s=25, color="#0178BC", label="ensayo")
    ax.plot(orden["trial"], mejor_hist, color="#F39200", lw=2, label="mejor hasta el momento")
    ax.axhline(base_mae, color="grey", ls="--", label=f"base ({base_mae:.0f})")
    ax.set_title("Ajuste de hiperparámetros LightGBM (búsqueda aleatoria walk-forward)")
    ax.set_xlabel("Ensayo")
    ax.set_ylabel("MAE medio (viajeros)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "tuning_lgbm.png", dpi=120)
    plt.close()
    print(f"[tune] figura en {FIGS / 'tuning_lgbm.png'}")


if __name__ == "__main__":
    main()
