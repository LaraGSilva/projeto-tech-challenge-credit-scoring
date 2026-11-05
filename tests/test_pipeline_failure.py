import json

import mlflow
import pytest

from src import pipeline
from src.contracts.validate import ContractViolation, make_invalid_batch


def test_pipeline_bloqueia_dados_invalidos_e_registra_falha(tmp_path, monkeypatch):
    # Isola MLflow e relatórios para não tocar nos artefatos reais.
    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "setup_mlflow", lambda: (
        mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"),
        mlflow.set_experiment("teste")))
    monkeypatch.setattr(pipeline, "load_raw", make_invalid_batch)

    with pytest.raises(ContractViolation):
        pipeline.run()

    summary = json.loads((tmp_path / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["health"]["pipeline_status"] == 0
    assert summary["health"]["contract_violations"] == 3
    assert summary["batches"] == {}
    assert "falha" in (tmp_path / "dashboard.html").read_text(encoding="utf-8")
