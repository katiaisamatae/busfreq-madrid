"""Comparación de modelos de predicción de demanda — Fase 2 (Módulo 1).

Evalúa varios modelos con EXACTAMENTE los mismos folds walk-forward y métricas:
  - naive      : demanda del mismo día de la semana anterior (lag_7).
  - lightgbm   : gradient boosting con todas las features.
  - prophet    : modelo aditivo por línea (tendencia + estacionalidad + regresores).
  - lstm       : red recurrente por línea (si PyTorch está disponible).

Asimetría de información (se documenta para no sobrevender el resultado):
LightGBM usa todas las features; Prophet solo tres regresores; la LSTM es
univariante. Todos respetan el mismo horizonte: la LSTM predice cada día de test
con información de al menos 7 días de antigüedad (lag >= 7), igual que los lags
de LightGBM, sin usar el valor del día anterior.

Registra los resultados en MLflow, imprime una tabla comparativa y guarda:
  - notebooks/figs/comparacion_modelos.png
  - notebooks/figs/shap_summary.png   (interpretabilidad del LightGBM)

Uso:
    python demand/compare.py --config configs/lgbm.yaml
    python -m demand.compare --config configs/lgbm.yaml --no-lstm
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demand.common import (feature_cols, folds_walk_forward, load_config,
                           load_dataset, metricas)

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT.parent / "notebooks" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
# Acotado: solo el ruido conocido (deprecaciones y avisos de Prophet/cmdstanpy),
# no un silenciado global que ocultaría problemas reales.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
for lg in ("cmdstanpy", "prophet"):
    logging.getLogger(lg).setLevel(logging.ERROR)

REGRESORES_PROPHET = ["tmed", "prec", "is_festivo"]


# --------------------------------------------------------------------------- #
# Modelos: cada uno recibe (tr, te, ...) y devuelve predicciones alineadas a te.
# --------------------------------------------------------------------------- #
def pred_naive(tr, te, **_):
    return te["lag_7"].to_numpy()


def pred_lightgbm(tr, te, feat, cat, m, **_):
    from demand.common import make_lgbm
    tr, te = tr.copy(), te.copy()
    for c in cat:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    modelo = make_lgbm(m)
    modelo.fit(tr[feat], tr["viajeros"], categorical_feature=cat)
    return modelo.predict(te[feat])


def pred_prophet(tr, te, **_):
    from prophet import Prophet
    pred = pd.Series(index=te.index, dtype=float)
    for linea, te_l in te.groupby("linea", observed=True):
        tr_l = tr[tr["linea"] == linea]
        if len(tr_l) < 90:
            pred[te_l.index] = te_l["lag_7"].to_numpy()
            continue
        dfp = tr_l[["fecha", "viajeros"] + REGRESORES_PROPHET].rename(
            columns={"fecha": "ds", "viajeros": "y"})
        medias = {r: dfp[r].mean() for r in REGRESORES_PROPHET}
        dfp = dfp.fillna(medias)
        modelo = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                         daily_seasonality=False)
        for r in REGRESORES_PROPHET:
            modelo.add_regressor(r)
        modelo.fit(dfp)
        fut = te_l[["fecha"] + REGRESORES_PROPHET].rename(columns={"fecha": "ds"})
        fut = fut.fillna(medias)
        fc = modelo.predict(fut)
        pred[te_l.index] = fc["yhat"].clip(lower=0).to_numpy()
    return pred.to_numpy()


def pred_lstm(tr, te, lookback=28, epochs=150, batch=64, **_):
    LAG = 7          # horizonte real del TFM: la ventana termina 7 días antes
    import torch
    import torch.nn as nn
    torch.manual_seed(42)

    class LSTMNet(nn.Module):
        def __init__(self, hidden=48):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    pred = pd.Series(index=te.index, dtype=float)
    for linea, te_l in te.groupby("linea", observed=True):
        serie_tr = tr[tr["linea"] == linea].sort_values("fecha")["viajeros"].to_numpy(float)
        if len(serie_tr) < lookback + 30:
            pred[te_l.index] = te_l["lag_7"].to_numpy()
            continue
        mu, sd = serie_tr.mean(), serie_tr.std() + 1e-6
        s = (serie_tr - mu) / sd
        # La red debe entrenarse al MISMO horizonte al que se la evalúa. Antes se
        # entrenaba a un paso (y = s[lookback:]) y luego se le pedía predecir con una
        # ventana que termina 7 días antes: devolvía el día t-6 y ese valor se usaba
        # como día t, desalineando el ciclo semanal. De ahí el MAPE del 58 %.
        top = len(s) - lookback - (LAG - 1)
        X = np.stack([s[i:i + lookback] for i in range(top)])
        y = np.array([s[i + lookback + LAG - 1] for i in range(top)])
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        net = LSTMNet()
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=60, gamma=0.5)
        lossf = nn.L1Loss()
        n = len(Xt)
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for j in range(0, n, batch):
                idx = perm[j:j + batch]
                opt.zero_grad()
                loss = lossf(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            sched.step()
        # Predicción con lag >= 7 (mismo horizonte 24-48 h que el resto): la
        # ventana de entrada de cada día de test TERMINA 7 días antes y no usa el
        # valor del día anterior. Antes se hacía hist.append(real) -> lag_1,
        # prohibido para el horizonte del TFM y ventaja injusta frente a los demás.
        te_l = te_l.sort_values("fecha")
        te_vals = te_l["viajeros"].to_numpy(float)
        serie = np.concatenate([serie_tr, te_vals])   # train + test reales
        base = len(serie_tr)                           # 1er día de test en `serie`
        lag7 = te_l["lag_7"].to_numpy(float)
        out = []
        net.eval()
        with torch.no_grad():
            for j in range(len(te_vals)):
                fin = base + j - (LAG - 1)   # fin exclusivo -> último índice usado = t-7
                ini = fin - lookback
                if ini < 0:
                    out.append(float(lag7[j]))
                    continue
                ventana = (serie[ini:fin] - mu) / sd
                xv = torch.tensor(ventana, dtype=torch.float32).view(1, lookback, 1)
                p = net(xv).item() * sd + mu
                out.append(max(p, 0.0))
        pred[te_l.index] = out
    return pred.to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparación de modelos de demanda")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-lstm", action="store_true", help="Omite el LSTM")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_dataset(cfg)
    feat = feature_cols(cfg, df)
    cat = cfg["features"]["categorical"]
    m = cfg["model"]
    cortes = folds_walk_forward(df["fecha"], cfg["split"]["n_folds"],
                                cfg["split"]["test_period_days"])
    print(f"[compare] {len(df):,} filas | {len(feat)} features | "
          f"{len(cortes)} folds walk-forward")

    # Modelos disponibles según dependencias instaladas.
    modelos = {"naive": pred_naive}
    try:
        import lightgbm  # noqa: F401
        modelos["lightgbm"] = pred_lightgbm
    except ImportError:
        print("[compare] lightgbm no disponible")
    try:
        import prophet  # noqa: F401
        modelos["prophet"] = pred_prophet
    except ImportError:
        print("[compare] prophet no disponible")
    if not args.no_lstm:
        try:
            import torch  # noqa: F401
            modelos["lstm"] = pred_lstm
        except ImportError:
            print("[compare] torch no disponible; se omite LSTM")

    filas = []
    for i, (t_ini, t_fin) in enumerate(cortes, 1):
        tr = df[df["fecha"] < t_ini]
        te = df[(df["fecha"] >= t_ini) & (df["fecha"] <= t_fin)]
        if te.empty:
            continue
        for nombre, fn in modelos.items():
            pred = fn(tr, te, feat=feat, cat=cat, m=m)
            mk = metricas(te["viajeros"].to_numpy(), pred)
            filas.append({"modelo": nombre, "fold": i, **mk})
        print(f"[compare] fold {i} ({t_ini.date()}→{t_fin.date()}) evaluado")

    res = pd.DataFrame(filas)
    cols = ["MAE", "RMSE", "MAPE", "WAPE"]
    resumen = res.groupby("modelo")[cols].mean().sort_values("MAE")
    desv = res.groupby("modelo")[cols].std().reindex(resumen.index)
    print("\n=== Comparación de modelos (media de folds walk-forward) ===")
    print(resumen.round(2).to_string())
    print("\n--- Desviación entre folds (±) ---")
    print(desv.round(2).to_string())

    # --- MLflow ---
    try:
        import mlflow
        uri = cfg["experiment"]["mlflow_tracking_uri"]
        if uri.startswith("sqlite"):
            # DB SQLite en la raíz del proyecto (ruta absoluta, robusto en Windows).
            dbpath = (ROOT / "mlflow.db").resolve().as_posix()
            mlflow.set_tracking_uri(f"sqlite:///{dbpath}")
        else:
            p = Path(uri) if Path(uri).is_absolute() else (ROOT / uri)
            p.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(p.resolve().as_uri())
        mlflow.set_experiment(cfg["experiment"]["name"])
        for nombre, fila in resumen.iterrows():
            with mlflow.start_run(run_name=nombre):
                mlflow.log_params({"modelo": nombre,
                                   "n_folds": cfg["split"]["n_folds"],
                                   "horizon_days": cfg["split"]["horizon_days"]})
                mlflow.log_metrics({k: float(fila[k]) for k in ["MAE", "RMSE", "MAPE", "WAPE"]})
        print(f"[compare] métricas registradas en MLflow ({mlflow.get_tracking_uri()})")
    except Exception as e:  # noqa: BLE001
        print(f"[compare] MLflow no disponible ({e.__class__.__name__}); se omite")

    # --- Figura comparativa (MAE y MAPE) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    resumen["MAE"].plot(kind="bar", ax=axes[0], color="#0178BC")
    axes[0].set_title("MAE por modelo (menor = mejor)")
    axes[0].set_ylabel("MAE (viajeros)")
    resumen["MAPE"].plot(kind="bar", ax=axes[1], color="#F39200")
    axes[1].set_title("MAPE por modelo (%)")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIGS / "comparacion_modelos.png", dpi=120)
    plt.close()
    print(f"[compare] figura en {FIGS / 'comparacion_modelos.png'}")

    # --- SHAP sobre LightGBM (interpretabilidad) ---
    if "lightgbm" in modelos:
        try:
            _shap_lightgbm(df, cortes, feat, cat, m)
        except Exception as e:  # noqa: BLE001
            print(f"[compare] SHAP omitido ({e.__class__.__name__}: {e})")


def _shap_lightgbm(df, cortes, feat, cat, m):
    import shap

    from demand.common import make_lgbm
    t_ini = cortes[-1][0]
    tr = df[df["fecha"] < t_ini].copy()
    te = df[(df["fecha"] >= t_ini)].copy()
    for c in cat:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    modelo = make_lgbm(m)
    modelo.fit(tr[feat], tr["viajeros"], categorical_feature=cat)
    muestra = te[feat].sample(min(800, len(te)), random_state=42)
    sv = shap.TreeExplainer(modelo).shap_values(muestra)
    shap.summary_plot(sv, muestra, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGS / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[compare] SHAP en {FIGS / 'shap_summary.png'}")


if __name__ == "__main__":
    main()
