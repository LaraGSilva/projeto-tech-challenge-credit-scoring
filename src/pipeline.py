"""Orquestra o pipeline completo e centraliza logs (JSON) e métricas (MLflow).

Uso: python -m src.pipeline    # depois: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import time
import uuid

import mlflow
import pandas as pd

from src.config import PRODUCTION_DIR, REFERENCE_PATH, REPORTS_DIR
from src.contracts.validate import ContractViolation, validate_batch
from src.data import split
from src.data.load import load_raw
from src.model import train
from src.model.predict import load_model
from src.monitoring.drift_report import drift_table, evidently_report
from src.monitoring.logging_setup import get_logger, set_run_id
from src.monitoring.metrics import BATCHES, metrics_by_batch
from src.monitoring.observability import evaluate_alerts, setup_mlflow, step
from src.simulation import drift_sim

log = get_logger("pipeline")


def run() -> dict:
    run_id = uuid.uuid4().hex[:8]
    set_run_id(run_id)
    setup_mlflow()
    timings: dict[str, float] = {}
    violations = rows = 0
    batches_summary: dict[str, dict] = {}
    start = time.perf_counter()

    with mlflow.start_run(run_name=f"pipeline-{run_id}") as parent:
        status = 0
        try:
            with step("contrato_dados_brutos", timings):
                try:
                    raw = validate_batch(load_raw(), "raw")
                except ContractViolation as exc:
                    violations += len(exc.violations)
                    raise
                rows += len(raw)
            with step("split", timings):
                split.main()
            with step("treino", timings):
                train.main()
            with step("simulacao_drift", timings):
                drift_sim.main()
            with step("monitoramento", timings):
                model = load_model()
                reference = pd.read_csv(REFERENCE_PATH)
                metrics = metrics_by_batch(model).set_index("lote")
                for name in BATCHES:
                    batch = pd.read_csv(PRODUCTION_DIR / f"{name}.csv")
                    rows += len(batch)
                    table = drift_table(reference, batch)
                    html = REPORTS_DIR / f"evidently_{name}.html"
                    evidently_report(reference, batch, html)
                    psi_mean = float(table["psi"].mean())
                    drift_features = table.loc[table["status"] == "drift", "feature"].tolist()
                    auc_drop = float(metrics.loc[name, "queda_auc_vs_teste_base"])
                    alerts = evaluate_alerts(name, psi_mean, auc_drop)
                    batches_summary[name] = {
                        "model_auc": float(metrics.loc[name, "auc"]),
                        "accuracy": float(metrics.loc[name, "accuracy"]),
                        "auc_drop": auc_drop, "psi_mean": round(psi_mean, 4),
                        "drift_features_count": len(drift_features),
                        "drift_features": drift_features,
                        "alerts_triggered": len(alerts), "alerts": alerts}
                    with mlflow.start_run(run_name=name, nested=True):
                        mlflow.set_tag("batch_id", name)
                        mlflow.log_metrics({k: v for k, v in batches_summary[name].items()
                                            if isinstance(v, (int, float))})
                        mlflow.log_artifact(str(html))
                        csv = REPORTS_DIR / f"drift_table_{name}.csv"
                        table.to_csv(csv, index=False)
                        mlflow.log_artifact(str(csv))
                        log.info("lote monitorado", extra={"batch_id": name, **{
                            k: v for k, v in batches_summary[name].items() if k != "alerts"}})
            status = 1
        finally:
            latency = round(time.perf_counter() - start, 3)
            health = {"pipeline_status": status, "pipeline_latency_seconds": latency,
                      "rows_processed": rows, "contract_violations": violations,
                      "alerts_triggered": sum(b["alerts_triggered"] for b in batches_summary.values())}
            mlflow.log_metrics(health)
            log.info("pipeline finalizado", extra={"mlflow_run_id": parent.info.run_id, **health})

    summary = {"run_id": run_id, "health": health, "step_timings_s": timings, "batches": batches_summary}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    run()
