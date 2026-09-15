"""tipo_dia clasifica bien laborables, sábados y festivos (Fase 2)."""
import pandas as pd

from demand.common import clasificar_tipo_dia
from optimizer.optimize import tipo_dia_calendario


def test_clasificar_tipo_dia():
    fechas = ["2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03", "2026-01-01"]
    t = list(clasificar_tipo_dia(fechas))
    assert t[0] == "laborable"   # jueves normal
    assert t[1] == "festivo"     # 1 de mayo (nacional)
    assert t[2] == "festivo"     # 2 de mayo (Comunidad de Madrid): prevalece sobre sábado
    assert t[3] == "festivo"     # domingo
    assert t[4] == "festivo"     # Año Nuevo


def test_sabado_normal_es_sabado():
    assert list(clasificar_tipo_dia(["2026-04-25"]))[0] == "sabado"


def test_tipo_dia_calendario_esquema_oferta():
    assert tipo_dia_calendario(pd.Timestamp("2026-04-30")) == "LA"   # jueves
    assert tipo_dia_calendario(pd.Timestamp("2026-04-25")) == "SA"   # sábado
    assert tipo_dia_calendario(pd.Timestamp("2026-01-01")) == "FE"   # festivo
    assert tipo_dia_calendario(pd.Timestamp("2026-05-03")) == "FE"   # domingo
