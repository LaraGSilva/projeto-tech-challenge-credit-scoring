"""Auditoria de viés: taxa de aprovação e de erro por grupo sensível e por faixa etária.

Decisão do modelo: nega o crédito se P(inadimplente) >= 0,5. Os atributos de grupo
(`estado_civil_sexo`, `idade`) NÃO são features do modelo; são usados só aqui, para auditar.
"""
import pandas as pd

from src.config import PRODUCTION_DIR, REPORTS_DIR, SENSITIVE, TARGET, TEST_BASE_PATH
from src.model.predict import load_model, predict_proba

THRESHOLD = 0.5
DISPARATE_IMPACT_MIN = 0.8   # regra dos 4/5, usada só como referência de triagem

# Código do dataset -> descrição. O código 2 mistura sexos, então não separa homem de mulher.
SEX_LABELS = {1: "1 homem divorciado/separado", 2: "2 mulher não solteira ou homem casado/viúvo",
              3: "3 homem solteiro", 4: "4 mulher solteira"}
AGE_BINS = [17, 24, 34, 49, 120]
AGE_LABELS = ["18-24", "25-34", "35-49", "50+"]


def group_metrics(df: pd.DataFrame, group: pd.Series, name: str) -> pd.DataFrame:
    """Uma linha por grupo: n, aprovação, FPR (bom negado) e FNR (mau aprovado)."""
    rows = []
    for g, part in df.groupby(group, observed=True):
        denied = part["pred_negado"] == 1
        good, bad = part[TARGET] == 0, part[TARGET] == 1
        rows.append({"atributo": name, "grupo": g, "n": len(part),
                     "taxa_aprovacao": round(1 - denied.mean(), 4),
                     "fpr_bom_negado": round(denied[good].mean(), 4) if good.any() else float("nan"),
                     "fnr_mau_aprovado": round((~denied)[bad].mean(), 4) if bad.any() else float("nan")})
    out = pd.DataFrame(rows)
    out["razao_aprovacao_vs_maior"] = (out["taxa_aprovacao"] / out["taxa_aprovacao"].max()).round(3)
    return out


def audit(model, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pred_negado"] = (predict_proba(model, df) >= THRESHOLD).astype(int)
    sex = df[SENSITIVE].map(SEX_LABELS)
    age = pd.cut(df["idade"], AGE_BINS, labels=AGE_LABELS)
    return pd.concat([group_metrics(df, sex, SENSITIVE), group_metrics(df, age, "faixa_etaria")],
                     ignore_index=True)


def main() -> None:
    model = load_model()
    REPORTS_DIR.mkdir(exist_ok=True)
    for name, path in (("teste_base", TEST_BASE_PATH), ("t3", PRODUCTION_DIR / "t3.csv")):
        out = audit(model, pd.read_csv(path))
        out.insert(0, "lote", name)
        out.to_csv(REPORTS_DIR / f"fairness_{name}.csv", index=False)
        print(f"\n== {name} ==\n{out.drop(columns='lote').to_string(index=False)}")


if __name__ == "__main__":
    main()
