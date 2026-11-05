"""Ablação: separa o efeito do data drift e do concept drift sobre a AUC.

Para cada lote, compara (a) só data drift (sem inversão de rótulos) com (b) data + concept
drift, repetindo com 20 seeds para medir a variação.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from src.config import REFERENCE_PATH, REPORTS_DIR, TEST_BASE_PATH
from src.model.predict import load_model
from src.monitoring.metrics import evaluate
from src.simulation.drift_sim import SCENARIOS, make_batch

N_SEEDS = 20


def main() -> None:
    model = load_model()
    base = pd.read_csv(TEST_BASE_PATH)
    median = pd.read_csv(REFERENCE_PATH)["valor_credito"].median()
    rows = [{"lote": "teste_base", "cenario": "sem drift", "auc_medio": evaluate(model, base)["auc"],
             "auc_desvio": 0.0}]
    for name, sc in SCENARIOS.items():
        for label, scenario in (("so_data_drift", replace(sc, concept_flip_prob=0.0)),
                                ("data_mais_concept_drift", sc)):
            aucs = [evaluate(model, make_batch(base, scenario, median, np.random.default_rng(k)))["auc"]
                    for k in range(N_SEEDS)]
            rows.append({"lote": name, "cenario": label, "auc_medio": round(float(np.mean(aucs)), 4),
                         "auc_desvio": round(float(np.std(aucs)), 4)})
    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "ablation_drift.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
