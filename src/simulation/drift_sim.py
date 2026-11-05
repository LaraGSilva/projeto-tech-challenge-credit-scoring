"""Simulação de produção: três lotes temporais (t1, t2, t3) com data e concept drift.

Cada lote parte do teste base (300 linhas) e aplica:
- data drift: alteração de `valor_credito`, `idade` e `duracao_credito`;
- concept drift: inversão de rótulos de bons pagadores em solicitações de valor
  acima da mediana da referência E prazo acima de 24 meses (a relação feature->alvo muda).

O concept drift exige rótulos; ele simula o rótulo real que chegaria depois da decisão.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PRODUCTION_DIR, REFERENCE_PATH, SEED, TARGET, TEST_BASE_PATH
from src.contracts.validate import validate_batch
from src.monitoring.logging_setup import get_logger

log = get_logger("simulation")

AGE_MIN, AGE_MAX = 18, 100
DURATION_MAX = 72
RISKY_DURATION = 24


@dataclass(frozen=True)
class Scenario:
    amount_factor: float      # multiplicador de valor_credito (inflação)
    age_mode: str             # "none" | "younger" | "older"
    duration_add: int         # meses somados a duracao_credito
    concept_flip_prob: float  # prob. de virar mau pagador no segmento de risco


SCENARIOS = {
    "t1": Scenario(1.10, "none", 0, 0.00),
    "t2": Scenario(1.25, "younger", 6, 0.25),
    "t3": Scenario(1.40, "older", 12, 0.60),
}


def _shift_age(age: pd.Series, mode: str, rng: np.random.Generator) -> pd.Series:
    if mode == "younger":
        age = age - rng.integers(0, 6, len(age))
    elif mode == "older":
        # 40% dos clientes passam a ter entre 56 e 70 anos.
        older = rng.random(len(age)) < 0.40
        age = age.where(~older, np.maximum(age, rng.integers(56, 71, len(age))))
    return age.clip(AGE_MIN, AGE_MAX).astype(int)


def make_batch(base: pd.DataFrame, scenario: Scenario, amount_median: float,
               rng: np.random.Generator) -> pd.DataFrame:
    df = base.copy()
    df["valor_credito"] = (df["valor_credito"] * scenario.amount_factor).round().astype(int)
    df["idade"] = _shift_age(df["idade"], scenario.age_mode, rng)
    df["duracao_credito"] = (df["duracao_credito"] + scenario.duration_add).clip(upper=DURATION_MAX)

    risky = (df["valor_credito"] > amount_median) & (df["duracao_credito"] > RISKY_DURATION)
    flip = risky & (df[TARGET] == 0) & (rng.random(len(df)) < scenario.concept_flip_prob)
    df.loc[flip, TARGET] = 1
    return df


def generate_batches() -> dict[str, pd.DataFrame]:
    reference = pd.read_csv(REFERENCE_PATH)
    base = pd.read_csv(TEST_BASE_PATH)
    amount_median = reference["valor_credito"].median()
    rng = np.random.default_rng(SEED)
    batches = {}
    for name, scenario in SCENARIOS.items():
        batch = validate_batch(make_batch(base, scenario, amount_median, rng), name)
        log.info("lote simulado", extra={"batch_id": name, "rows": len(batch),
                                         "taxa_inadimplencia": round(batch[TARGET].mean(), 4)})
        batches[name] = batch
    return batches


def main() -> None:
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    for name, batch in generate_batches().items():
        batch.to_csv(PRODUCTION_DIR / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
