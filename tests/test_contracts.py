import pandas as pd
import pytest

from src.contracts.validate import ContractViolation, make_invalid_batch, validate_batch
from src.data.load import load_raw


def test_dataset_original_e_valido():
    df = load_raw()
    assert validate_batch(df, "raw") is df


def test_lote_invalido_e_bloqueado_com_regras_corretas():
    with pytest.raises(ContractViolation) as exc:
        validate_batch(make_invalid_batch(), "teste")
    quebradas = {v["coluna"] for v in exc.value.violations}
    assert {"idade", "valor_credito", "duracao_credito"} <= quebradas


@pytest.mark.parametrize("coluna,valor", [("idade", 17), ("valor_credito", 0),
                                          ("duracao_credito", 73), ("tipo_moradia", 9),
                                          ("estado_civil_sexo", 0)])
def test_cada_regra_rigida_bloqueia(coluna, valor):
    df = load_raw().head(10).copy()
    df.loc[df.index[0], coluna] = valor
    with pytest.raises(ContractViolation):
        validate_batch(df, "teste")


def test_duplicatas_bloqueadas():
    df = load_raw().head(10)
    with pytest.raises(ContractViolation):
        validate_batch(pd.concat([df, df.iloc[[0]]]), "dup")


def test_nulo_bloqueado():
    df = load_raw().head(10).astype({"idade": float})
    df.loc[df.index[0], "idade"] = None
    with pytest.raises(ContractViolation):
        validate_batch(df, "nulo")
