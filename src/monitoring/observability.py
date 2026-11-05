"""Ponto central de observabilidade: etapas cronometradas, alertas e registro no MLflow."""
import time
from contextlib import contextmanager

import mlflow

from src.config import (ALERT_AUC_DROP, ALERT_PSI_MEAN, MLFLOW_ARTIFACTS, MLFLOW_DB_URI,
                        MLFLOW_EXPERIMENT)
from src.monitoring.logging_setup import get_logger

log = get_logger("pipeline")


@contextmanager
def step(name: str, timings: dict):
    """Loga início/fim de uma etapa com duração e status; falhas são logadas e re-levantadas."""
    start = time.perf_counter()
    log.info("etapa iniciada", extra={"step": name})
    try:
        yield
    except Exception as exc:
        timings[name] = round(time.perf_counter() - start, 3)
        log.error("etapa falhou", extra={"step": name, "duration_s": timings[name],
                                         "erro": type(exc).__name__})
        raise
    timings[name] = round(time.perf_counter() - start, 3)
    log.info("etapa concluída", extra={"step": name, "duration_s": timings[name]})


def evaluate_alerts(batch_id: str, psi_mean: float, auc_drop: float) -> list[dict]:
    """Aplica os limiares de alerta e registra cada alerta disparado como WARNING."""
    alerts = []
    if psi_mean > ALERT_PSI_MEAN:
        alerts.append({"tipo": "psi_medio_alto", "valor": round(psi_mean, 4), "limite": ALERT_PSI_MEAN})
    if auc_drop > ALERT_AUC_DROP:
        alerts.append({"tipo": "queda_auc", "valor": round(auc_drop, 4), "limite": ALERT_AUC_DROP})
    for alert in alerts:
        log.warning("alerta de degradação", extra={"batch_id": batch_id, **alert})
    return alerts


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
