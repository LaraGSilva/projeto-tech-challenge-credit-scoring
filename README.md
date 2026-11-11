# projeto-tech-challenge-credit-scoring 📊🔍

## Objetivo principal
O projeto desenvolvido tem o principal objetivo de garantir a sustentação e confiabilidade da pipeline de treino, teste e retreino. Devido a suscetibilidade de mudança nos dados, é importante aplicar um plano de monitoramento da qualidade, quantiadade e conceito dos dados além de ter atenção a uma possível degradação silenciosa também.


✨ Acesse o video star: [Link do video](https://canva.link/pr5qa7bl5uep6ue)


## Visão geral

Pipeline de credit scoring com observabilidade e detecção de drift. Um modelo de Regressão Logística estima a probabilidade de inadimplência; o projeto valida os dados de entrada, simula três lotes de produção com degradação crescente, detecta o drift (PSI, KS e Evidently), rastreia métricas e alertas (logs JSON e MLflow) e consolida tudo em um dashboard HTML.

```
dataset ─► split estratificado ─► referência (700) / teste base (300)
              │
              ├─► contrato de dados (Pandera) ─► bloqueia lotes inválidos
              ├─► modelo base (Regressão Logística)
              ├─► simulação de produção: lotes t1, t2, t3 (data + concept drift)
              ├─► PSI, KS e Evidently por lote ─► alertas
              └─► logs JSON + MLflow + dashboard HTML
```

## Como rodar

```bash
pip install -r requirements.txt

python -m src.pipeline                  # roda tudo: contrato, split, treino, simulação, monitoramento, dashboard
python -m pytest -q                     # testes
mlflow ui --backend-store-uri sqlite:///mlflow.db     # experimentos em http://localhost:5000
```

Saídas em `reports/`: `dashboard.html`, `evidently_t1|t2|t3.html`, `drift_table_*.csv`, `metrics_by_batch.csv`, `pipeline_summary.json`. Logs em `logs/pipeline.jsonl`.

Módulos avulsos:

| Comando | O que faz |
|---|---|
| `python -m src.contracts.validate --demo` | Mostra o bloqueio de um lote com erros propositais |
| `python -m src.monitoring.metrics` | AUC e acurácia por lote |
| `python -m src.monitoring.drift_report` | Tabelas PSI/KS e relatórios Evidently |
| `python -m src.monitoring.fairness` | Auditoria de viés por grupo |
| `python -m src.simulation.ablation` | Separa o efeito do data drift e do concept drift |

## Estrutura do repositório

```
src/
  config.py            caminhos, seed, colunas, limites de alerta
  data/                load.py, split.py
  contracts/           schema.py (Pandera), validate.py (bloqueio)
  model/               train.py, predict.py
  simulation/          drift_sim.py (lotes t1-t3), ablation.py
  monitoring/          metrics, drift_stats (PSI/KS), drift_report (Evidently),
                       observability (etapas, alertas, MLflow), dashboard, fairness, logging_setup
  pipeline.py          orquestra tudo
tests/                 contrato, simulação, PSI, drift, observabilidade, falha do pipeline, viés
docs/                  governanca_lgpd.md, analise_causal.md
```

## Dados

O dataset é o *South German Credit*: **1.000 linhas**, 20 atributos codificados como inteiros e o alvo `classe_credito`, sem valores ausentes. O alvo foi invertido para `inadimplente` (1 = mau pagador; 30% dos casos). Split estratificado 70/30 com `random_state=42`: 700 linhas de referência (treino) e 300 de teste base.

> **Atenção:** o enunciado pede pelo menos 5.000 amostras. Este dataset não atende. Confirmar com o professor se é aceito; as alternativas do enunciado são Home Credit e Lending Club. A estrutura do projeto não depende do dataset.

Adaptações em relação ao enunciado: o dataset **não tem renda**, inflação nem desemprego. A inflação foi simulada como multiplicador do valor do crédito e o limite de renda foi substituído por limites de valor e prazo.

## Etapa 1: contrato de dados

O contrato ([schema.py](src/contracts/schema.py)) bloqueia a ingestão: se qualquer regra é quebrada, `validate_batch` levanta `ContractViolation` e registra no log cada regra e quantas linhas a violaram.

| Regra | Coluna | Condição |
|---|---|---|
| R1 | `idade` | entre 18 e 100, sem nulos |
| R2 | `valor_credito` | maior que 0 |
| R3 | `duracao_credito` | entre 4 e 72 meses |
| R4 | `tipo_moradia` | 1, 2 ou 3 |
| R5 | `estado_civil_sexo` | 1 a 4 |
| R6 | tabela | nenhuma linha duplicada |
| R7 | demais colunas | domínio de cada categoria; sem nulos |

Demonstração (`python -m src.contracts.validate --demo`): um lote com idade 15, valor −500 e duração 200 meses é rejeitado com 3 violações e código de saída 1:

```
{"level": "ERROR", "event": "lote rejeitado", "batch_id": "demo_invalido", "n_violacoes": 3, "violacoes": [
  {"coluna": "duracao_credito", "regra": "in_range(4, 72)", "linhas_afetadas": 1, "exemplos": ["200"]},
  {"coluna": "idade", "regra": "in_range(18, 100)", "linhas_afetadas": 1, "exemplos": ["15"]},
  {"coluna": "valor_credito", "regra": "greater_than(0)", "linhas_afetadas": 1, "exemplos": ["-500"]}]}
```

**Modelo base:** Regressão Logística (`class_weight="balanced"`) com one-hot nas categóricas e padronização nas numéricas. Escolhida por ser interpretável, o que ajuda na análise causal e no direito de revisão (LGPD, art. 20). `estado_civil_sexo` fica fora das features.

## Etapa 2: simulação e detecção de drift

Três lotes de 300 linhas, derivados do teste base:

| Lote | `valor_credito` | `idade` | `duracao_credito` | Concept drift |
|---|---|---|---|---|
| t1 | ×1,10 | sem mudança | sem mudança | nenhum |
| t2 | ×1,25 | até 5 anos mais jovens | +6 meses | 25% |
| t3 | ×1,40 | 40% passam a ter 56-70 anos | +12 meses | 60% |

**Concept drift** (exige rótulos): no segmento de valor acima da mediana da referência **e** prazo acima de 24 meses, bons pagadores passam a maus pagadores com a probabilidade da tabela. A taxa de inadimplência vai de 30% (t1) a 36,7% (t2) e 52,3% (t3).

**Detecção:** PSI implementado no projeto (menos de 0,1 estável; 0,1 a 0,25 atenção; mais de 0,25 drift), teste KS para numéricas e relatório Evidently (`DataDriftPreset`). O PSI tem testes unitários com valores calculados à mão ([test_psi.py](tests/test_psi.py)).

| Lote | AUC | Acurácia | Queda da AUC* | PSI médio | Features com drift (PSI > 0,25) |
|---|---|---|---|---|---|
| teste base | 0,760 | 0,653 | — | — | — |
| t1 | 0,760 | 0,660 | 0,001 | 0,017 | nenhuma |
| t2 | 0,728 | 0,643 | 0,033 | 0,126 | `duracao_credito` (1,73) |
| t3 | 0,709 | 0,670 | 0,051 | 0,332 | `duracao_credito` (5,00), `idade` (0,81), `valor_credito` (0,29) |

\* Queda da AUC contra o teste base (hold-out), não contra a referência: a AUC da referência (0,848) é de treino e otimista.

Pontos para interpretar:

- **A acurácia não cai** (em t3 chega a subir). Com mais maus pagadores no lote, o modelo acerta mais por ter mais casos positivos. A AUC é a métrica que revela a degradação.
- **O Evidently é mais sensível que o PSI** com 300 linhas: marca 4 colunas com drift já em t1, onde o PSI não marca nenhuma.
- **O PSI não enxerga o concept drift.** Ver [a ablação](docs/analise_causal.md): só o data drift quase não muda a AUC; a queda vem do concept drift.

## Etapa 3: observabilidade

`python -m src.pipeline` registra cada execução no MLflow (run pai por execução, run filho por lote), em logs JSON e no dashboard ([dashboard.html](reports/dashboard.html), gerado). Todo log carrega um `run_id`, e cada etapa registra duração e status.

### Métricas de saúde

| Métrica | Nível | Significado |
|---|---|---|
| `pipeline_status` | execução | 1 = sucesso, 0 = falha |
| `pipeline_latency_seconds` | execução | tempo total da execução |
| `rows_processed` | execução | linhas validadas e processadas (bruto + lotes) |
| `contract_violations` | execução | regras do contrato quebradas |
| `model_auc` | lote | AUC do modelo no lote |
| `drift_features_count` | lote | features com PSI > 0,25 |
| `alerts_triggered` | execução e lote | alertas disparados |

Também por lote: `accuracy`, `auc_drop`, `psi_mean`.

### Alertas

| Alerta | Condição |
|---|---|
| `psi_medio_alto` | PSI médio das features > 0,10 |
| `queda_auc` | queda da AUC > 0,05 contra o teste base |

Resultado: t1 sem alertas; t2 dispara `psi_medio_alto`; t3 dispara `psi_medio_alto` e `queda_auc` (0,0514, por pouco acima do limite). Cada alerta vira um log de nível WARNING e aparece no dashboard. Se o contrato bloqueia os dados, o pipeline termina com `pipeline_status = 0` e o dashboard mostra a falha ([teste](tests/test_pipeline_failure.py)).

## Etapa 4: governança, LGPD e causalidade

### Proteção de dados (resumo; detalhes em [docs/governanca_lgpd.md](docs/governanca_lgpd.md))

- **Dados pessoais:** o dataset é público e sem identificadores diretos, mas idade, estado civil/sexo, nacionalidade, profissão e moradia são tratados como dados pessoais. Os logs registram só métricas agregadas, nunca linhas de clientes.
- **Minimização:** `estado_civil_sexo` não é feature do modelo; é usado só na auditoria de viés.
- **Base legal (a confirmar com revisão jurídica):** art. 7º, V (procedimentos preliminares ao contrato), X (proteção do crédito) ou IX (legítimo interesse). Art. 20: direito de revisão de decisões automatizadas, apoiado por um modelo interpretável e por logs rastreáveis; faltaria um canal de revisão humana em produção.
- **Retenção (proposta):** logs e experimentos 90 dias; dados derivados e relatórios são regeneráveis e descartados ao fim do projeto.

### Viés (auditoria sobre o teste base)

- **Sexo/estado civil:** nenhum grupo fica abaixo de 0,8 da maior taxa de aprovação. O dataset não separa sexo com clareza (o código 2 mistura sexos) e dois grupos têm 12 e 31 linhas, então a conclusão é fraca.
- **Idade:** os clientes de 18-24 anos têm 40,4% de aprovação contra 67,7% dos de 35-49 (razão 0,60), e 51,9% dos bons pagadores jovens são negados. No lote t3 a razão cai para 0,32. O modelo usa `idade`, então manter essa feature é uma decisão em aberto.
- `trabalhador_estrangeiro` (proxy de nacionalidade) também é feature e merece a mesma discussão.

### Análise causal (resumo; detalhes em [docs/analise_causal.md](docs/analise_causal.md))

- **DAG assumido:** inflação → valor solicitado → comprometimento da parcela → inadimplência, com prazo e idade como outras causas e renda como variável não observada.
- **Drift nas features não implica mudança causal.** Na ablação, o t3 com **só data drift** tem AUC 0,758 (contra 0,760 sem drift), mesmo com PSI de 5,0 em `duracao_credito`. Com concept drift a AUC cai para 0,707. Alertas só de PSI geram falsos positivos e deixam passar o concept drift, que exige acompanhar o desempenho com rótulos.
- **Limitação:** o modelo prevê associação, não causa. O concept drift foi imposto por construção na simulação, e os dados têm viés de seleção (só clientes aprovados no passado).

## Limitações do projeto

1. Dataset de 1.000 linhas, abaixo dos 5.000 exigidos.
2. Os três lotes usam as mesmas 300 linhas do teste base, com transformações diferentes; não são amostras independentes.
3. O concept drift é sintético e sua intensidade (25% e 60%) é arbitrária.
4. O DAG não foi validado com dados.
5. Base legal e prazos de retenção precisam de revisão jurídica.
