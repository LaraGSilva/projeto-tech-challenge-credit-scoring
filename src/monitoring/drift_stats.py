"""Estatísticas de drift implementadas à mão: PSI e Kolmogorov-Smirnov.

Interpretação do PSI: < 0,1 estável; 0,1 a 0,25 atenção; > 0,25 drift significativo.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN, PSI_DRIFT = 0.1, 0.25
EPS = 1e-6


def _psi_from_props(e: np.ndarray, a: np.ndarray) -> float:
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected, actual, bins: int = 10) -> float:
    """PSI para variável numérica, com bins por quantis da referência (`expected`)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    # unique: variáveis discretas geram quantis repetidos e bins de largura zero.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    return _psi_from_props(e, a)


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI para variável categórica, comparando proporções de cada categoria."""
    cats = sorted(set(expected.unique()) | set(actual.unique()))
    e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi_from_props(e, a)


def ks(expected, actual) -> tuple[float, float]:
    """Estatística KS e p-valor (duas amostras)."""
    res = ks_2samp(expected, actual)
    return float(res.statistic), float(res.pvalue)


def psi_status(value: float) -> str:
    if value > PSI_DRIFT:
        return "drift"
    return "atencao" if value >= PSI_WARN else "estavel"
