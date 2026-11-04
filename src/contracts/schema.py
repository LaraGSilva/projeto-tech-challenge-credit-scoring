"""Contrato de dados (Pandera) para lotes de solicitações de crédito.

As regras de domínio refletem o dicionário do South German Credit; as faixas de
`idade` e `duracao_credito` seguem o plano técnico (18-100 anos, 4-72 meses).
"""
import pandera.pandas as pa
from pandera.pandas import Check, Column


def _cat(values: list[int]) -> Column:
    return Column(int, Check.isin(values), nullable=False)


# R1-R5: regras rígidas exigidas pelo plano; demais colunas: domínio das categorias.
# R6 (nível de tabela): nenhuma linha duplicada.
schema = pa.DataFrameSchema(
    {
        "idade": Column(int, Check.in_range(18, 100), nullable=False),                # R1
        "valor_credito": Column(int, Check.gt(0), nullable=False),                    # R2
        "duracao_credito": Column(int, Check.in_range(4, 72), nullable=False),        # R3
        "tipo_moradia": _cat([1, 2, 3]),                                              # R4
        "estado_civil_sexo": _cat([1, 2, 3, 4]),                                      # R5
        "profissao": _cat([1, 2, 3, 4]),
        "conta_corrente": _cat([1, 2, 3, 4]),
        "historico_pagamento": _cat([0, 1, 2, 3, 4]),
        "finalidade_credito": _cat([0, 1, 2, 3, 4, 5, 6, 8, 9, 10]),
        "conta_poupanca": _cat([1, 2, 3, 4, 5]),
        "tempo_emprego": _cat([1, 2, 3, 4, 5]),
        "taxa_prestacao": _cat([1, 2, 3, 4]),
        "tempo_residencia": _cat([1, 2, 3, 4]),
        "fiador_garantias": _cat([1, 2, 3]),
        "patrimonio": _cat([1, 2, 3, 4]),
        "outros_creditos": _cat([1, 2, 3]),
        "numero_creditos": _cat([1, 2, 3, 4]),
        "dependentes": _cat([1, 2]),
        "telefone": _cat([1, 2]),
        "trabalhador_estrangeiro": _cat([1, 2]),
    },
    checks=Check(lambda df: ~df.duplicated(), error="linhas duplicadas"),
    strict=False,
)
