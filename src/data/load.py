"""Carregamento do dataset bruto com nomes de colunas em português."""
import pandas as pd

from src.config import COLUMN_MAP, RAW_PATH, TARGET, TARGET_RAW


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";").rename(columns=COLUMN_MAP)
    # classe_credito: 1 = bom, 0 = mau  ->  inadimplente: 1 = mau.
    df[TARGET] = 1 - df[TARGET_RAW]
    return df.drop(columns=[TARGET_RAW])
