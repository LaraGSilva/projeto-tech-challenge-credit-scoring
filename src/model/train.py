"""Modelo base: Regressão Logística com pré-processamento (one-hot + padronização)."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (CATEGORICAL_FEATURES, FEATURES, MODEL_PATH, NUMERIC_FEATURES,
                        REFERENCE_PATH, SEED, TARGET, TEST_BASE_PATH)
from src.contracts.validate import validate_batch
from src.monitoring.logging_setup import get_logger
from src.monitoring.metrics import evaluate

log = get_logger("model")


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    reference = validate_batch(pd.read_csv(REFERENCE_PATH), "reference")
    test_base = validate_batch(pd.read_csv(TEST_BASE_PATH), "test_base")
    model = build_pipeline().fit(reference[FEATURES], reference[TARGET])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info("modelo treinado", extra={"rows": len(reference),
             "train": evaluate(model, reference), "test_base": evaluate(model, test_base)})


if __name__ == "__main__":
    main()
