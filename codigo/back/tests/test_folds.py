"""Los folds walk-forward no filtran futuro ni se solapan (Fase 1, OG2)."""
import pandas as pd

from demand.common import folds_walk_forward, seleccionar_folds


def _fechas(n=720):   # ~2 años: garantiza histórico antes del primer fold
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))


def test_folds_consecutivos_sin_solape():
    cortes = folds_walk_forward(_fechas(360), n_folds=4, test_days=90)
    assert len(cortes) == 4
    for ini, fin in cortes:
        assert (fin - ini).days == 89            # bloque de 90 días
    for (i0, i1), (j0, j1) in zip(cortes, cortes[1:]):
        assert j0 == i1 + pd.Timedelta(days=1)   # sin hueco ni solape


def test_train_anterior_al_test():
    fechas = _fechas()   # 2 años: el primer fold tiene histórico de entrenamiento
    df = pd.DataFrame({"fecha": fechas})
    for t_ini, t_fin in folds_walk_forward(fechas, n_folds=4, test_days=90):
        train = df[df["fecha"] < t_ini]
        test = df[(df["fecha"] >= t_ini) & (df["fecha"] <= t_fin)]
        assert train["fecha"].max() < test["fecha"].min()   # nada del futuro en train


def test_seleccionar_folds_reserva_el_ultimo():
    cortes = folds_walk_forward(_fechas(360), n_folds=4, test_days=90)
    tune = seleccionar_folds(cortes, [1, 2, 3])
    assert len(tune) == 3
    assert cortes[3] not in tune                 # el test_fold queda fuera del tuning
