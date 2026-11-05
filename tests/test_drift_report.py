import pandas as pd

from src.config import FEATURES, REFERENCE_PATH
from src.monitoring.drift_report import drift_table
from src.simulation.drift_sim import generate_batches


def test_referencia_vs_ela_mesma_nao_tem_drift():
    ref = pd.read_csv(REFERENCE_PATH)
    table = drift_table(ref, ref)
    assert set(table["feature"]) == set(FEATURES)
    assert (table["status"] == "estavel").all()


def test_lote_t3_acusa_drift_nas_variaveis_alteradas():
    ref = pd.read_csv(REFERENCE_PATH)
    table = drift_table(ref, generate_batches()["t3"]).set_index("feature")
    for col in ("valor_credito", "duracao_credito", "idade"):
        assert table.loc[col, "status"] == "drift"
    assert table.loc["telefone", "status"] == "estavel"   # não foi alterada
