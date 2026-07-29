# Datathon PosTech FIAP — Fase 5 · Associação Passos Mágicos

Análise do PEDE (Pesquisa Extensiva do Desenvolvimento Educacional) de **2022, 2023 e 2024** e **modelo preditivo de risco de defasagem escolar**, com aplicação Streamlit para uso pelas equipes da Passos Mágicos.

## Estrutura do repositório

```
├── src/
│   ├── preparacao_dados.py       # limpeza e construção do painel aluno-ano + pares longitudinais
│   └── analise_exploratoria.py   # gera as figuras que respondem às perguntas de negócio
├── notebooks/
│   └── modelo_risco_defasagem.ipynb  # feature engineering, treino/teste, modelagem e avaliação
├── app/
│   ├── app.py                    # aplicação Streamlit (tema com a paleta FIAP)
│   ├── .streamlit/config.toml    # cores do tema (magenta FIAP #ED145B)
│   ├── modelo_risco.joblib       # modelo final (Gradient Boosting calibrado)
│   └── requirements.txt
├── figures/                      # gráficos da análise
└── apresentacao_datathon_passos_magicos.pptx  # storytelling gerencial
```

## Como reproduzir

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl joblib

# 1. Coloque BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx na raiz do projeto
python src/preparacao_dados.py       # gera data/panel_pede.csv e data/pares_longitudinais.csv
python src/analise_exploratoria.py   # gera as figuras
# 2. Execute notebooks/modelo_risco_defasagem.ipynb (gera modelo_risco.joblib)
```

## Aplicação Streamlit

Rodar local:

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Deploy no **Streamlit Community Cloud**:

1. Suba este repositório no GitHub (a pasta `app/` precisa conter `app.py`, `modelo_risco.joblib` e `requirements.txt`).
2. Em [share.streamlit.io](https://share.streamlit.io), conecte a conta do GitHub → *New app*.
3. Selecione o repositório, branch `main` e o arquivo principal `app/app.py`.
4. *Deploy* — a URL pública gerada pode ser compartilhada com a Passos Mágicos.

## O modelo em resumo

- **Alvo:** probabilidade de o aluno estar em defasagem escolar (`defasagem < 0`) no **ano seguinte**.
- **Dados:** pares longitudinais aluno (ano *t* → ano *t+1*) via `RA`: 2022→2023 e 2023→2024 (1.248 pares válidos).
- **Modelo:** `HistGradientBoostingClassifier` (lida nativamente com valores ausentes) + calibração isotônica.
- **Validação temporal:** treino nos pares 2022→23, teste nos pares 2023→24 → **AUC 0,84**, precisão de 79% na classe de risco (corte 0,5).
- **Principais variáveis:** defasagem atual, IAN, fase, idade, IPP, IPV e INDE.

## Principais achados da análise

1. Alunos **"em fase"** saltaram de **30% (2022) para 54% (2024)**; a defasagem severa caiu de 3,3% para 0,3%.
2. **Topázio dobrou** em dois anos (15% → 31%) e o INDE médio subiu de 7,04 para 7,40 — evidência de efetividade do programa.
3. **IEG (engajamento)** correlaciona forte com IDA (r≈0,54) e IPV (r≈0,53): é o indicador mais alavancável.
4. **IAA (autoavaliação)** descola do desempenho real (r≈0,2) — bom para autoestima, ruim como termômetro acadêmico.
5. **IPS baixo antecede queda acadêmica**: alunos no quartil inferior de IPS têm 34% de chance de perder 1+ ponto de IDA no ano seguinte (vs 24% no grupo intermediário-alto).
6. O **IPP confirma o IAN**: quanto maior a defasagem, menor o indicador psicopedagógico.
7. O IDA cai de forma recorrente nas **fases 3–5** — a transição para a adolescência é o ponto crítico do funil.

---
*Datathon PosTech FIAP — Fase 5 · dados anonimizados fornecidos pela Associação Passos Mágicos.*
