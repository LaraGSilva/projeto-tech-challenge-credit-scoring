# Plano técnico: Credit Scoring com observabilidade e drift

## Modelo para predição de score de crédito

O projeto desenvolvido tem o principal objetivo de garantir a sustentação e confiabilidade da pipeline de treino, teste e retreino. Devido a suscetibilidade de mudança nos dados, é importante aplicar um plano de monitoramento da qualidade, quantiadade e conceito dos dados além de ter atenção a uma possível degradação silenciosa também.

## Visão geral do fluxo

```
german_credit_data.csv
   │
   ├─► Split estratificado ─► Referência (treino, ~70%)
   │                       └► Teste base (~30%) ─► Simulação de produção
   │
   ├─► Contrato de dados (Pandera) ─► bloqueia lotes inválidos
   ├─► Modelo base (Logistic Regression / XGBoost)
   ├─► Simulação de drift em 3 lotes temporais (Data + Concept)
   ├─► Evidently AI ─► relatórios HTML + PSI / KS
   ├─► Logs + métricas ─► MLflow + dashboard HTML consolidado
   └─► README: governança, LGPD, causalidade
```

---

## Estrutura do repositório

```
credit-risk-observability/
├── README.md                    # governança, LGPD, análise causal, como rodar
├── requirements.txt             # versões fixadas
├── .gitignore                   # mlruns/, logs/, __pycache__/
├── data/
│   ├── raw/german_credit_data.csv
│   ├── reference/               # split de treino (gerado)
│   └── production/              # lotes simulados t1, t2, t3 (gerado)
├── src/
│   ├── config.py                # caminhos, seed, colunas, thresholds
│   ├── data/
│   │   ├── load.py
│   │   └── split.py
│   ├── contracts/
│   │   ├── schema.py            # Pandera schema (Etapa 1)
│   │   └── validate.py          # bloqueia ingestão se falhar
│   ├── model/
│   │   ├── train.py             # modelo base
│   │   └── predict.py
│   ├── simulation/
│   │   └── drift_sim.py         # cria lotes com drift (Etapa 2)
│   ├── monitoring/
│   │   ├── drift_report.py      # Evidently + PSI + KS
│   │   ├── metrics.py           # AUC, acurácia por lote
│   │   └── logging_setup.py     # logs estruturados do pipeline
│   └── pipeline.py              # orquestra as etapas
├── tests/
│   ├── test_contracts.py
│   ├── test_drift_sim.py
│   └── test_psi.py
├── reports/                     # HTMLs gerados (Evidently e dashboard)
├── docs/
│   ├── governanca_lgpd.md       # opcional: detalhar o README
│   └── analise_causal.md
└── logs/
```

---

## Etapa 1: Validação de dados e contratos

**Objetivo:** impedir que dados ruins cheguem ao modelo.

### Modelo base

- Target: `Risk` (good/bad), convertido para 0/1.
- Features categóricas com one-hot encoding. Numéricas padronizadas.
- Comece com **Logistic Regression**, porque é interpretável e facilita a análise causal. Se quiser, compare com XGBoost e documente a escolha.
- Divida em referência (treino) e teste base com split **estratificado** e `random_state` fixo.

### Contrato de dados com Pandera

Mínimo de 3 regras rígidas; este plano usa 5:

```python
import pandera as pa
from pandera import Column, Check

schema = pa.DataFrameSchema(
    {
        "Age": Column(int, Check.in_range(18, 100), nullable=False),
        "Credit amount": Column(float, Check.gt(0), nullable=False),
        "Duration": Column(int, Check.in_range(4, 72), nullable=False),
        "Housing": Column(str, Check.isin(["own", "rent", "free"])),
        "Sex": Column(str, Check.isin(["male", "female"])),
    },
    unique=False,
    strict=False,
)
```

Regras adicionais recomendadas:

- Nenhuma linha duplicada (verificar com `df.duplicated()`).
- `Job` entre 0 e 3.
- Valores ausentes em `Checking account` são permitidos, mas documentados como categoria `"unknown"`.

### Demonstração de bloqueio

- Crie um lote com erros propositais: idade 15, `Credit amount` negativo, duração 200 meses.
- O script deve **falhar com exceção** e registrar no log quais regras quebraram.
- Mostre a saída no vídeo e no README.

**Entregável:** `src/contracts/validate.py` funcionando, modelo base treinado, log de validação do lote com erros.

---

## Etapa 2: Simulação e detecção de drift

**Objetivo:** mostrar a degradação ao longo do tempo.

### Simulação em três lotes temporais

| Lote | Mudança em `Credit amount` | Mudança em `Age` | Mudança em `Duration` | Concept drift |
|---|---|---|---|---|
| t1 | +10% | sem mudança | sem mudança | não |
| t2 | +25% (inflação) | deslocamento para clientes mais jovens | +6 meses | leve |
| t3 | +40% | mais clientes acima de 55 anos | +12 meses | forte |

**Data drift:** altere a distribuição das features. Exemplo:

```python
prod["Credit amount"] = prod["Credit amount"] * 1.25
prod["Age"] = prod["Age"].clip(18, 100) - rng.integers(0, 5, len(prod))
```

**Concept drift:** altere a relação entre features e target. Exemplo: nos lotes t2 e t3, aumente a probabilidade de `bad` para solicitações com valor acima da mediana **e** prazo acima de 24 meses. Isso exige rótulos, e você deve documentar isso no README.

### Predições e métricas

- Gere predições para cada lote com o modelo treinado na referência.
- Calcule AUC e acurácia por lote. A queda entre t1 e t3 é a evidência de degradação silenciosa.

### Detecção estatística

- Use **Evidently AI** para comparar referência × cada lote, com `DataDriftPreset`.
- Implemente você mesmo o **PSI** para validar o resultado. Interpretação padrão: PSI < 0,1 estável; 0,1 a 0,25 atenção; > 0,25 drift significativo.
- Complemente com **Kolmogorov-Smirnov** para variáveis numéricas.
- Escreva testes unitários para `psi()` com casos conhecidos, para garantir que a implementação está correta.

```python
import numpy as np

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
```

**Entregável:** HTML do Evidently para cada lote, tabela de features com PSI/KS e identificação clara das variáveis que sofreram drift.

> **Atenção à versão:** a API do Evidently mudou entre versões. Fixe a versão em `requirements.txt` e confira a documentação correspondente antes de escrever o código.

---

## Etapa 3: Observabilidade de pipelines e modelos

**Objetivo:** dar visibilidade operacional ao pipeline.

**Escolha de ferramenta:** use **MLflow** (local, gratuito, sem infraestrutura extra) para rastrear experimentos e métricas. Gere também um **dashboard HTML consolidado** com um script. Datadog e Prometheus são opcionais e mais complexos para este prazo.

### O que registrar

- Por execução do pipeline: status (sucesso/falha), tempo de execução, número de linhas processadas, número de regras de contrato quebradas.
- Por lote: AUC, acurácia, PSI médio, número de features com drift, alerta gerado (sim/não).
- Logs estruturados em JSON, com um registro por etapa.

### Métricas de saúde a documentar (no README)

- `pipeline_status`, `pipeline_latency_seconds`, `rows_processed`, `contract_violations`, `drift_features_count`, `model_auc`, `alerts_triggered`.

### Alertas

- Defina limites: por exemplo, alerta se PSI médio > 0,1 ou AUC cair mais de 0,05 em relação à referência.
- Registre cada alerta no log e no dashboard.

**Entregável:** dashboard HTML (ou interface do MLflow) mostrando a saúde do pipeline, a estabilidade do modelo e os alertas de degradação em t1, t2 e t3.

---

## Etapa 4: Governança e LGPD (README)

**Objetivo:** documentar a arquitetura, a conformidade e a análise causal.

### Plano de proteção de dados

- **Dados pessoais (PII):** idade, sexo, emprego e moradia são dados pessoais. Explique como foram tratados (anonimização, não exposição em logs, uso apenas para fins do projeto).
- **Base legal:** explique a base legal para a decisão de crédito. Na LGPD, decisões automatizadas de crédito envolvem o art. 7º (execução de contrato ou legítimo interesse, conforme o caso) e o art. 20 (direito de revisão de decisões automatizadas). Confira a base com seu grupo ou professor, porque isso exige revisão jurídica.
- **Retenção:** defina por quanto tempo os dados e logs são mantidos e como são descartados.
- **Viés:** o atributo `Sex` não deve ser usado como feature. Compare taxas de aprovação e erro entre sexos e faixas etárias, e documente o resultado.

### Análise causal

- Descreva o grafo causal assumido (DAG): por exemplo, inflação → valor médio solicitado → inadimplência.
- Explique por que drift nas features **não** implica automaticamente mudança causal, e como o concept drift se diferencia disso.
- Discuta a limitação: o modelo prevê associação, e não causa.

