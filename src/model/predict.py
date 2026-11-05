"""Predição de probabilidade de inadimplência com o modelo treinado."""
import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_proba(model, df: pd.DataFrame):
    return model.predict_proba(df[FEATURES])[:, 1]
