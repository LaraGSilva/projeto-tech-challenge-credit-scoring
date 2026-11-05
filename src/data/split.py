"""Split estratificado: referência (treino) e teste base (origem dos lotes de produção)."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import REFERENCE_PATH, SEED, TARGET, TEST_BASE_PATH, TEST_SIZE
from src.data.load import load_raw


def split_reference_test(df: pd.DataFrame):
    return train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)


def main() -> None:
    reference, test_base = split_reference_test(load_raw())
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(REFERENCE_PATH, index=False)
    test_base.to_csv(TEST_BASE_PATH, index=False)
    print(f"referência={len(reference)} teste_base={len(test_base)}")


if __name__ == "__main__":
    main()
