"""Configuração central: caminhos, seed, colunas e thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "german_credit_data.csv"
REFERENCE_PATH = ROOT / "data" / "reference" / "reference.csv"
TEST_BASE_PATH = ROOT / "data" / "reference" / "test_base.csv"
PRODUCTION_DIR = ROOT / "data" / "production"
MODEL_PATH = ROOT / "models" / "baseline_logreg.joblib"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

SEED = 42
TEST_SIZE = 0.30

# Nomes originais (alemão) -> nomes em português, iguais aos do notebook de EDA.
COLUMN_MAP = {
    "laufkont": "conta_corrente",
    "laufzeit": "duracao_credito",
    "moral": "historico_pagamento",
    "verw": "finalidade_credito",
    "hoehe": "valor_credito",
    "sparkont": "conta_poupanca",
    "beszeit": "tempo_emprego",
    "rate": "taxa_prestacao",
    "famges": "estado_civil_sexo",
    "buerge": "fiador_garantias",
    "wohnzeit": "tempo_residencia",
    "verm": "patrimonio",
    "alter": "idade",
    "weitkred": "outros_creditos",
    "wohn": "tipo_moradia",
    "bishkred": "numero_creditos",
    "beruf": "profissao",
    "pers": "dependentes",
    "telef": "telefone",
    "gastarb": "trabalhador_estrangeiro",
    "kredit": "classe_credito",
}

# kredit: 1 = bom pagador, 0 = mau pagador. Target do modelo = inadimplência (1 = mau).
TARGET_RAW = "classe_credito"
TARGET = "inadimplente"

NUMERIC_FEATURES = [
    "duracao_credito",
    "valor_credito",
    "idade",
    "taxa_prestacao",
    "tempo_residencia",
    "numero_creditos",
    "dependentes",
]
CATEGORICAL_FEATURES = [
    "conta_corrente",
    "historico_pagamento",
    "finalidade_credito",
    "conta_poupanca",
    "tempo_emprego",
    "fiador_garantias",
    "patrimonio",
    "outros_creditos",
    "tipo_moradia",
    "profissao",
    "telefone",
    "trabalhador_estrangeiro",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Atributo sensível: codifica estado civil + sexo. NÃO entra no modelo (LGPD/viés),
# mas é mantido no dado para auditoria de viés.
SENSITIVE = "estado_civil_sexo"
