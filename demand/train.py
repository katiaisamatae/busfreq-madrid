"""Entrenamiento de modelos de predicción de demanda — Módulo 1.

Fase 2 del plan de trabajo. Compara baseline naive, LightGBM, Prophet y LSTM.
Evaluación con validación WALK-FORWARD (nunca datos futuros en entrenamiento).
Métrica principal: MAE de validaciones por franja horaria.
Tracking de experimentos con MLflow; importancia de features con SHAP.

Uso:
    python demand/train.py --config configs/lgbm.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena modelo de demanda")
    parser.add_argument("--config", required=True, help="Ruta al YAML de configuración")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[demand.train] Experimento: {cfg['experiment']['name']}")
    # TODO Fase 2:
    #   1. Cargar features de data/processed/
    #   2. Generar folds walk-forward
    #   3. Entrenar y evaluar (MAE), loguear en MLflow
    #   4. SHAP sobre el modelo final y serializar
    raise NotImplementedError("Pendiente: implementar pipeline de entrenamiento")


if __name__ == "__main__":
    main()
