import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_stats import ks, psi, psi_categorical, psi_status


def test_psi_distribuicoes_identicas_e_zero():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_valor_conhecido_calculado_a_mao():
    # 2 bins por mediana: esperado 50/50, atual 80/20.
    # PSI = (0.8-0.5)*ln(0.8/0.5) + (0.2-0.5)*ln(0.2/0.5) = 0.4165
    expected = np.array([1] * 50 + [2] * 50)
    actual = np.array([1] * 80 + [2] * 20)
    assert psi(expected, actual, bins=2) == pytest.approx(0.4165, abs=1e-3)


def test_psi_e_nao_negativo_e_cresce_com_o_deslocamento():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 5000)
    valores = [psi(base, rng.normal(m, 1, 5000)) for m in (0, 0.2, 0.5, 1.0)]
    assert all(v >= 0 for v in valores)
    assert valores == sorted(valores)
    assert valores[0] < 0.1 and valores[-1] > 0.25


def test_psi_variavel_discreta_com_quantis_repetidos():
    expected = np.array([1] * 90 + [2] * 10)
    assert np.isfinite(psi(expected, expected))


def test_psi_categorical_valor_conhecido():
    e = pd.Series(["a"] * 50 + ["b"] * 50)
    a = pd.Series(["a"] * 80 + ["b"] * 20)
    assert psi_categorical(e, a) == pytest.approx(0.4165, abs=1e-3)


def test_psi_categorical_categoria_nova_nao_quebra():
    assert psi_categorical(pd.Series([1, 1, 2]), pd.Series([1, 3, 3])) > 0.25


def test_ks_detecta_deslocamento():
    rng = np.random.default_rng(2)
    stat, p = ks(rng.normal(0, 1, 500), rng.normal(1, 1, 500))
    assert stat > 0.3 and p < 0.01


@pytest.mark.parametrize("valor,status", [(0.05, "estavel"), (0.1, "atencao"),
                                          (0.25, "atencao"), (0.26, "drift")])
def test_psi_status(valor, status):
    assert psi_status(valor) == status
