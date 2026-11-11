# Governança e LGPD

> Este documento descreve as decisões técnicas do projeto.

## 1. Dados tratados

O dataset é o *South German Credit*. Ele **não contém identificadores diretos** (nome, CPF, endereço, contato). Mesmo assim, vários atributos são dados pessoais quando associados a um indivíduo (LGPD, art. 5º, I) e foram tratados como tal:

| Atributo | Por que merece cuidado | Tratamento no projeto |
|---|---|---|
| `idade` | Atributo pessoal; risco de discriminação etária | Usado pelo modelo. Auditado por faixa etária (seção 5) |
| `estado_civil_sexo` | Mistura estado civil e sexo | **Excluído das features do modelo.** Usado apenas na auditoria de viés |
| `trabalhador_estrangeiro` | Proxy de nacionalidade | **Ainda é feature do modelo.** Ponto em aberto (seção 6) |
| `profissao`, `tipo_moradia`, `patrimonio`, `telefone` | Informação socioeconômica | Usados pelo modelo; nunca aparecem em logs |

Nenhum desses atributos é dado pessoal *sensível* na lista fechada do art. 5º, II (origem racial/étnica, religião, saúde, biometria etc.), mas o risco de discriminação existe e o princípio da **não discriminação** (art. 6º, IX) se aplica.

## 2. Medidas de proteção implementadas

- **Minimização (art. 6º, III):** `estado_civil_sexo` é removido antes do treino ([config.py](../src/config.py), lista `FEATURES`).
- **Qualidade dos dados (art. 6º, V):** o contrato Pandera bloqueia a ingestão de lotes inválidos ([schema.py](../src/contracts/schema.py)).
- **Sem dados pessoais em logs:** os logs JSON registram apenas identificadores de lote, contagens, métricas agregadas e o valor inválido que quebrou uma regra (ex.: idade `15`). Nenhuma linha de cliente é registrada. A trilha fica em `logs/pipeline.jsonl` e no MLflow.
- **Dados fora do Git:** `data/production/`, `logs/`, `models/`, `mlruns/` e `mlflow.db` estão no `.gitignore`. Os lotes de produção são sintéticos, derivados do dataset público.
- **Finalidade limitada:** os dados são usados apenas para o exercício acadêmico de monitoramento de modelo.

## 3. Base legal

Em um cenário real de concessão de crédito, as bases candidatas são:

- **Art. 7º, V:** execução de contrato ou de procedimentos preliminares, a pedido do titular (a análise de crédito é preliminar ao contrato).
- **Art. 7º, X:** proteção do crédito.
- **Art. 7º, IX:** legítimo interesse, que exige teste de balanceamento e registro.

O projeto usa dados públicos e fins acadêmicos, então não há titular real a quem aplicar uma base legal. A tabela serve para mostrar o raciocínio.

## 4. Decisão automatizada e direito de revisão (art. 20)

O modelo decide sozinho (nega se P(inadimplente) ≥ 0,5). Em produção, o art. 20 dá ao titular o direito de **solicitar revisão** de decisões tomadas unicamente com base em tratamento automatizado que afetem seus interesses, incluindo a definição de perfil de crédito, e de receber **informações claras sobre os critérios e procedimentos** usados. O que o projeto oferece para isso:

- **Modelo interpretável:** Regressão Logística, cujos coeficientes explicam o peso de cada atributo.
- **Rastreabilidade:** `run_id` em todos os logs e experimentos no MLflow permitem reconstruir qual modelo e quais dados geraram uma decisão.
- **Monitoramento de degradação:** o dashboard avisa quando o modelo deixa de ser confiável (ver README).

O que **falta** para atender o art. 20 em produção: canal de revisão humana, registro por decisão (com o devido controle de acesso) e texto de explicação ao titular.

## 5. Auditoria de viés (resultado)

Gerada por `python -m src.monitoring.fairness` sobre o teste base (300 linhas). A decisão é "negar se P(inadimplente) ≥ 0,5". A razão de aprovação compara cada grupo com o de maior aprovação (referência de triagem: 0,8, a "regra dos 4/5").

**Por estado civil/sexo** (códigos do dataset; o código 2 mistura sexos, então **não separa homem de mulher**):

| Grupo | n | Aprovação | Bom pagador negado | Mau pagador aprovado | Razão |
|---|---|---|---|---|---|
| 1 homem divorciado/separado | 12 | 58,3% | 28,6% | 40,0% | 0,95 |
| 2 mulher não solteira ou homem casado/viúvo | 90 | 52,2% | 38,1% | 29,6% | 0,85 |
| 3 homem solteiro | 167 | 60,5% | 31,9% | 41,7% | 0,99 |
| 4 mulher solteira | 31 | 61,3% | 28,6% | 40,0% | 1,00 |

**Por faixa etária:**

| Faixa | n | Aprovação | Bom pagador negado | Mau pagador aprovado | Razão |
|---|---|---|---|---|---|
| 18-24 | 52 | **40,4%** | **51,9%** | 32,0% | **0,60** |
| 25-34 | 122 | 59,0% | 31,4% | 36,1% | 0,87 |
| 35-49 | 93 | 67,7% | 26,4% | 47,6% | 1,00 |
| 50+ | 33 | 54,5% | 40,0% | 37,5% | 0,81 |

**Leitura:**

- **Sexo/estado civil:** nenhum grupo cai abaixo de 0,8. Os grupos 1 e 4 têm 12 e 31 linhas, então as taxas deles são pouco confiáveis e não permitem concluir ausência de viés.
- **Idade:** a faixa 18-24 tem razão 0,60, bem abaixo de 0,8. Mais da metade dos bons pagadores jovens é negada. Como `idade` é feature, o modelo penaliza jovens diretamente. A decisão de manter ou não a idade como feature é do grupo e deve considerar o contexto regulatório.
- **No lote t3** (clientes mais velhos e concept drift), a razão da faixa 18-24 cai para 0,32 (n = 28). Degradação do modelo e viés pioram juntos, o que justifica auditar equidade também em produção, e não só na avaliação inicial.
- O limite de 0,8 é uma heurística de triagem, não um critério legal.


## 6. Retenção e descarte (proposta do projeto)

| Artefato | Conteúdo | Retenção proposta | Descarte |
|---|---|---|---|
| `data/raw/` | Dataset público original | Duração da disciplina | Apagar o diretório |
| `data/reference/`, `data/production/` | Derivados do dataset (reprodutíveis) | Regeneráveis a qualquer momento | Apagar; regenerar com `python -m src.pipeline` |
| `logs/pipeline.jsonl` | Eventos e métricas agregadas | 90 dias | Apagar o arquivo (rotacionar em produção) |
| `mlflow.db`, `mlruns/` | Experimentos e artefatos | 90 dias | Apagar o banco e o diretório |
| `reports/` | Dashboard, HTMLs do Evidently e tabelas | Regeneráveis | Apagar |

