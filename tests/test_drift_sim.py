import numpy as np
import pandas as pd
import pytest

from src.config import TARGET, TEST_BASE_PATH
from src.simulation.drift_sim import SCENARIOS, Scenario, generate_batches, make_batch


@pytest.fixture(scope="module")
def base():
    return pd.read_csv(TEST_BASE_PATH)


@pytest.fixture(scope="module")
def batches():
    return generate_batches()


def test_lotes_passam_no_contrato_e_mantem_tamanho(batches, base):
    assert set(batches) == {"t1", "t2", "t3"}
    assert all(len(b) == len(base) for b in batches.values())


def test_valor_credito_cresce_conforme_cenario(batches, base):
    for name, sc in SCENARIOS.items():
        ratio = batches[name]["valor_credito"].sum() / base["valor_credito"].sum()
        assert ratio == pytest.approx(sc.amount_factor, rel=0.01)


def test_duracao_aumenta_e_respeita_teto(batches, base):
    assert batches["t3"]["duracao_credito"].mean() > batches["t2"]["duracao_credito"].mean() \
        > batches["t1"]["duracao_credito"].mean()
    assert batches["t3"]["duracao_credito"].max() <= 72


def test_idade_t2_mais_jovem_t3_mais_velha(batches, base):
    assert batches["t1"]["idade"].equals(base["idade"])
    assert batches["t2"]["idade"].mean() < base["idade"].mean()
    assert (batches["t3"]["idade"] > 55).mean() > (base["idade"] > 55).mean()


def test_concept_drift_so_aumenta_inadimplencia(batches, base):
    taxas = [base[TARGET].mean()] + [batches[n][TARGET].mean() for n in ("t1", "t2", "t3")]
    assert taxas[1] == taxas[0]          # t1 sem concept drift
    assert taxas[2] > taxas[1] and taxas[3] > taxas[2]


def test_simulacao_e_reprodutivel(base):
    sc = Scenario(1.25, "younger", 6, 0.25)
    a = make_batch(base, sc, 2300, np.random.default_rng(1))
    b = make_batch(base, sc, 2300, np.random.default_rng(1))
    pd.testing.assert_frame_equal(a, b)
