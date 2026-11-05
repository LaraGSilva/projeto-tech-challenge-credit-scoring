"""AUC e acurácia por lote: evidência da degradação silenciosa do modelo."""
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import FEATURES, PRODUCTION_DIR, REFERENCE_PATH, REPORTS_DIR, TARGET, TEST_BASE_PATH
from src.model.predict import load_model, predict_proba
from src.monitoring.logging_setup import get_logger

log = get_logger("metrics")
BATCHES = ["t1", "t2", "t3"]


def evaluate(model, df: pd.DataFrame) -> dict:
    proba = model.predict_proba(df[FEATURES])[:, 1]
    return {"auc": round(roc_auc_score(df[TARGET], proba), 4),
            "accuracy": round(accuracy_score(df[TARGET], proba >= 0.5), 4)}


def metrics_by_batch(model) -> pd.DataFrame:
    sources = {"referencia": REFERENCE_PATH, "teste_base": TEST_BASE_PATH,
               **{b: PRODUCTION_DIR / f"{b}.csv" for b in BATCHES}}
    rows = []
    for name, path in sources.items():
        df = pd.read_csv(path)
        rows.append({"lote": name, "rows": len(df), "taxa_inadimplencia": round(df[TARGET].mean(), 4),
                     **evaluate(model, df)})
    out = pd.DataFrame(rows)
    # A referência (treino) tem AUC otimista; a queda justa é medida contra o teste base.
    for col, lote in (("queda_auc_vs_referencia", "referencia"), ("queda_auc_vs_teste_base", "teste_base")):
        out[col] = (out.loc[out["lote"] == lote, "auc"].iloc[0] - out["auc"]).round(4)
    return out


def main() -> None:
    out = metrics_by_batch(load_model())
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "metrics_by_batch.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
