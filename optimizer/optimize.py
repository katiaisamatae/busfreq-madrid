"""Optimizador de frecuencias — Módulo 2.

Fase 3 del plan de trabajo. Dado el forecast de demanda y la flota disponible,
resuelve qué frecuencia mínima por línea mantiene el tiempo de espera por debajo
de un umbral en el percentil 90 de la demanda.

Programación lineal entera (ILP) con OR-Tools:
  minimizar  tiempo de espera medio
  sujeto a:  presupuesto de flota, frecuencia mínima por línea, turnos de conductor

Output: vector de frecuencias recomendadas por línea/franja (24-48h)
        + delta de coste respecto al plan vigente.

Uso:
    python optimizer/optimize.py --date 2026-06-01
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimiza frecuencias de bus")
    parser.add_argument("--date", required=True, help="Fecha objetivo YYYY-MM-DD")
    args = parser.parse_args()

    print(f"[optimizer] Optimizando frecuencias para {args.date}")
    # TODO Fase 3:
    #   1. Cargar forecast de demanda del Módulo 1
    #   2. Definir variables (frecuencia por línea/franja) y restricciones
    #   3. Resolver ILP con OR-Tools
    #   4. Exportar frecuencias recomendadas + delta de coste
    raise NotImplementedError("Pendiente: implementar modelo de optimización")


if __name__ == "__main__":
    main()
