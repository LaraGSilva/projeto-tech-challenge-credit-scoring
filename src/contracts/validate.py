"""Bloqueia a ingestão de lotes que violam o contrato de dados."""
import argparse
import sys

import pandas as pd
import pandera.pandas as pa

from src.contracts.schema import schema
from src.data.load import load_raw
from src.monitoring.logging_setup import get_logger

log = get_logger("contracts")


class ContractViolation(Exception):
    """Lote rejeitado: uma ou mais regras do contrato foram quebradas."""

    def __init__(self, violations: list[dict]):
        self.violations = violations
        super().__init__(f"{len(violations)} violação(ões) de contrato")


def validate_batch(df: pd.DataFrame, batch_id: str = "batch") -> pd.DataFrame:
    """Retorna o df se válido; senão loga cada regra quebrada e levanta ContractViolation."""
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        fc = exc.failure_cases
        violations = [
            {
                "coluna": None if pd.isna(col) else col,
                "regra": str(check),
                "linhas_afetadas": int(len(g)),
                "exemplos": g["failure_case"].astype(str).head(3).tolist(),
            }
            for (col, check), g in fc.groupby(["column", "check"], dropna=False)
        ]
        log.error("lote rejeitado", extra={"batch_id": batch_id, "n_violacoes": len(violations),
                                           "violacoes": violations})
        raise ContractViolation(violations) from exc
    log.info("lote válido", extra={"batch_id": batch_id, "rows": len(df)})
    return df


def make_invalid_batch(n: int = 20) -> pd.DataFrame:
    """Lote com erros propositais: idade 15, valor negativo e duração de 200 meses."""
    df = load_raw().head(n).copy()
    df.loc[df.index[0], "idade"] = 15
    df.loc[df.index[1], "valor_credito"] = -500
    df.loc[df.index[2], "duracao_credito"] = 200
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Valida um lote contra o contrato de dados.")
    p.add_argument("csv", nargs="?", help="CSV do lote (omitido com --demo)")
    p.add_argument("--demo", action="store_true", help="valida um lote com erros propositais")
    args = p.parse_args()
    if args.demo:
        df, batch_id = make_invalid_batch(), "demo_invalido"
    elif args.csv:
        df, batch_id = pd.read_csv(args.csv), args.csv
    else:
        p.error("informe um CSV ou --demo")
    try:
        validate_batch(df, batch_id)
    except ContractViolation as exc:
        print(f"\nIngestão BLOQUEADA: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
