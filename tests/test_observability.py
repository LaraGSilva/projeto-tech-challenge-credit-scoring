import pytest

from src.monitoring.observability import evaluate_alerts, step


def test_sem_alerta_dentro_dos_limites():
    assert evaluate_alerts("t1", psi_mean=0.05, auc_drop=0.01) == []


def test_alerta_de_psi_e_de_auc_sao_independentes():
    assert [a["tipo"] for a in evaluate_alerts("t2", 0.2, 0.01)] == ["psi_medio_alto"]
    assert [a["tipo"] for a in evaluate_alerts("t2", 0.05, 0.06)] == ["queda_auc"]
    assert len(evaluate_alerts("t3", 0.3, 0.06)) == 2


def test_limite_exato_nao_dispara():
    assert evaluate_alerts("t", 0.10, 0.05) == []


def test_step_registra_duracao_em_sucesso_e_em_falha():
    timings = {}
    with step("ok", timings):
        pass
    with pytest.raises(ValueError):
        with step("falha", timings):
            raise ValueError("x")
    assert set(timings) == {"ok", "falha"}
