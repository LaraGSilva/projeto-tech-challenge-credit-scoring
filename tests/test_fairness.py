import pandas as pd

from src.monitoring.fairness import group_metrics


def test_group_metrics_calcula_aprovacao_fpr_e_fnr():
    df = pd.DataFrame({"inadimplente": [0, 0, 1, 1], "pred_negado": [1, 0, 0, 1]})
    out = group_metrics(df, pd.Series(["a"] * 4), "x").iloc[0]
    assert out["taxa_aprovacao"] == 0.5
    assert out["fpr_bom_negado"] == 0.5     # 1 de 2 bons pagadores negado
    assert out["fnr_mau_aprovado"] == 0.5   # 1 de 2 maus pagadores aprovado


def test_razao_de_aprovacao_normalizada_pelo_maior_grupo():
    df = pd.DataFrame({"inadimplente": [0] * 4, "pred_negado": [0, 0, 1, 1]})
    out = group_metrics(df, pd.Series(["a", "a", "b", "b"]), "x").set_index("grupo")
    assert out.loc["a", "razao_aprovacao_vs_maior"] == 1.0
    assert out.loc["b", "razao_aprovacao_vs_maior"] == 0.0
