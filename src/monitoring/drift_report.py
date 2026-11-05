"""Detecção de drift por lote: tabela PSI/KS por feature (e relatórios Evidently)."""
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, PRODUCTION_DIR, REFERENCE_PATH, REPORTS_DIR
from src.monitoring.drift_stats import ks, psi, psi_categorical, psi_status
from src.monitoring.logging_setup import get_logger
from src.monitoring.metrics import BATCHES

log = get_logger("drift")


def drift_table(reference: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por feature: PSI (todas), KS e p-valor (numéricas) e status."""
    rows = []
    for col in FEATURES:
        if col in NUMERIC_FEATURES:
            stat, p = ks(reference[col], batch[col])
            value = psi(reference[col], batch[col])
        else:
            stat = p = float("nan")
            value = psi_categorical(reference[col], batch[col])
        rows.append({"feature": col, "tipo": "numerica" if col in NUMERIC_FEATURES else "categorica",
                     "psi": round(value, 4), "ks_stat": round(stat, 4), "ks_pvalue": round(p, 4),
                     "status": psi_status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def evidently_report(reference: pd.DataFrame, batch: pd.DataFrame, path) -> dict:
    """Gera o HTML do Evidently (referência x lote) e retorna o resumo de colunas com drift."""
    definition = DataDefinition(numerical_columns=NUMERIC_FEATURES,
                                categorical_columns=CATEGORICAL_FEATURES)
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(batch[FEATURES], data_definition=definition),
        Dataset.from_pandas(reference[FEATURES], data_definition=definition))
    snapshot.save_html(str(path))
    return snapshot.dict()["metrics"][0]["value"]  # {"count": ..., "share": ...}


def main() -> None:
    reference = pd.read_csv(REFERENCE_PATH)
    REPORTS_DIR.mkdir(exist_ok=True)
    for name in BATCHES:
        batch = pd.read_csv(PRODUCTION_DIR / f"{name}.csv")
        table = drift_table(reference, batch)
        table.to_csv(REPORTS_DIR / f"drift_table_{name}.csv", index=False)
        summary = evidently_report(reference, batch, REPORTS_DIR / f"evidently_{name}.html")
        drifted = table.loc[table["status"] == "drift", "feature"].tolist()
        log.info("drift calculado", extra={"batch_id": name, "psi_medio": round(table["psi"].mean(), 4),
                                           "features_com_drift": drifted,
                                           "evidently_colunas_com_drift": int(summary["count"])})
        print(f"\n== {name} ==\n{table.head(8).to_string(index=False)}")


if __name__ == "__main__":
    main()
