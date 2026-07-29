
"""
Datathon FIAP — Passos Mágicos
App Streamlit: previsão de risco de defasagem escolar.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Passos Mágicos — Risco de Defasagem",
                   page_icon="🎓", layout="wide")


@st.cache_resource
def carrega_modelo():
    return joblib.load(Path(__file__).parent / "modelo_risco.joblib")


artefato = carrega_modelo()
modelo = artefato["model"]
FEATURES = artefato["features"]
AUC = artefato["auc_temporal"]

st.markdown("<h1 style='color:#ED145B;margin-bottom:0'>Passos Mágicos</h1>"
            "<h3 style='color:#1A1A1A;margin-top:0'>Previsão de risco de defasagem escolar</h3>",
            unsafe_allow_html=True)
st.markdown(
    f"Modelo treinado com os PEDEs de **2022, 2023 e 2024** para estimar a "
    f"**probabilidade de um aluno estar em defasagem escolar no próximo ano**, "
    f"a partir dos indicadores do ano atual. "
    f"Desempenho em validação temporal (treino 2022→23, teste 2023→24): **AUC = {AUC:.2f}**."
)



# ---------------------------------------------------------------- dados do repositório
ARQ_BASE = "BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx"


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _extrai_fase(x):
    """'ALFA' -> 0; '7', 'FASE 7', '7A' -> 7."""
    import re
    x = str(x).strip().upper()
    if x == "ALFA":
        return 0
    m = re.search(r"(\d)", x)
    return int(m.group(1)) if m else np.nan


@st.cache_data
def carrega_base_2024():
    """Lê a aba PEDE2024 do Excel bruto do repositório e aplica o tratamento."""
    for base_dir in (Path(__file__).parent, Path(__file__).parent.parent):
        caminho = base_dir / ARQ_BASE
        if caminho.exists():
            break
    else:
        return None
    d = pd.read_excel(caminho, sheet_name="PEDE2024")
    out = pd.DataFrame({"RA": d["RA"]})
    out["fase"] = d["Fase"].apply(_extrai_fase)
    out["genero"] = d["Gênero"]
    out["idade"] = _to_num(d["Idade"])
    out["anos_na_pm"] = 2024 - _to_num(d["Ano ingresso"])
    for k in ["IAA", "IEG", "IPS", "IPP", "IDA", "IPV", "IAN"]:
        out[k] = _to_num(d[k])
    out["INDE"] = _to_num(d["INDE 2024"])          # "INCLUIR" vira NaN
    out["nota_mat"] = _to_num(d["Mat"])
    out["nota_por"] = _to_num(d["Por"])
    out["nota_ing"] = _to_num(d["Ing"])
    out["defasagem"] = _to_num(d["Defasagem"])
    return out.dropna(subset=["INDE", "defasagem"])  # remove alunos sem avaliação

aba1, aba2 = st.tabs(["🔍 Avaliar um aluno", "📄 Avaliar uma turma (CSV)"])

# ---------------------------------------------------------------- aba 1
with aba1:
    st.subheader("Indicadores do aluno no ano atual")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ida = st.slider("IDA — Aprendizagem", 0.0, 10.0, 6.5, 0.1)
        ieg = st.slider("IEG — Engajamento", 0.0, 10.0, 8.0, 0.1)
        iaa = st.slider("IAA — Autoavaliação", 0.0, 10.0, 8.0, 0.1)
        ips = st.slider("IPS — Psicossocial", 0.0, 10.0, 6.5, 0.1)
    with c2:
        ipp = st.slider("IPP — Psicopedagógico", 0.0, 10.0, 7.0, 0.1)
        ipv = st.slider("IPV — Ponto de virada", 0.0, 10.0, 7.5, 0.1)
        ian = st.select_slider("IAN — Adequação ao nível", [2.5, 5.0, 10.0], 5.0)
        inde = st.slider("INDE — Índice geral", 0.0, 10.0, 7.2, 0.05)
    with c3:
        nota_mat = st.slider("Nota Matemática", 0.0, 10.0, 6.0, 0.1)
        nota_por = st.slider("Nota Português", 0.0, 10.0, 6.5, 0.1)
        nota_ing = st.slider("Nota Inglês", 0.0, 10.0, 6.0, 0.1)
    with c4:
        fase = st.selectbox("Fase atual", list(range(0, 9)), index=3,
                            format_func=lambda x: "ALFA" if x == 0 else f"Fase {x}")
        defas = st.selectbox("Defasagem atual (fase real − ideal)",
                             [-5, -4, -3, -2, -1, 0, 1, 2, 3], index=5)
        idade = st.number_input("Idade", 5, 25, 12)
        anos_pm = st.number_input("Anos na Passos Mágicos", 0, 15, 2)
        genero = st.radio("Gênero", ["Feminino", "Masculino"], horizontal=True)

    entrada = pd.DataFrame([{
        "IAA_t": iaa, "IEG_t": ieg, "IPS_t": ips, "IPP_t": ipp, "IDA_t": ida,
        "IPV_t": ipv, "IAN_t": ian, "INDE_t": inde, "defasagem_t": defas,
        "fase_t": fase, "idade_t": idade, "anos_na_pm_t": anos_pm,
        "nota_mat_t": nota_mat, "nota_por_t": nota_por, "nota_ing_t": nota_ing,
        "genero_fem": 1 if genero == "Feminino" else 0,
    }])[FEATURES]

    prob = float(modelo.predict_proba(entrada)[0, 1])

    st.divider()
    e1, e2 = st.columns([1, 2])
    with e1:
        st.metric("Probabilidade de defasagem no próximo ano", f"{prob:.0%}")
        if prob >= 0.7:
            st.error("🔴 **Risco alto** — priorizar acompanhamento individualizado.")
        elif prob >= 0.4:
            st.warning("🟡 **Risco moderado** — monitorar de perto e reforçar apoio.")
        else:
            st.success("🟢 **Risco baixo** — manter acompanhamento regular.")
    with e2:
        st.progress(min(prob, 1.0))
        alertas = []
        if defas < 0:
            alertas.append("aluno já está em defasagem — o histórico é o fator mais persistente")
        if ipp < 6:
            alertas.append("IPP abaixo de 6 — sinal psicopedagógico de atenção")
        if ips < 6:
            alertas.append("IPS abaixo de 6 — associado a quedas futuras de desempenho")
        if ieg < 6:
            alertas.append("engajamento baixo — forte relação com IDA e ponto de virada")
        if alertas:
            st.markdown("**Pontos de atenção:** " + "; ".join(alertas) + ".")

# ---------------------------------------------------------------- aba 2
with aba2:
    st.subheader("Avaliar a base completa")

    def pontua(df):
        """Recebe um DataFrame com os indicadores e devolve o ranking de risco."""
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        mapa = {"iaa": "IAA_t", "ieg": "IEG_t", "ips": "IPS_t", "ipp": "IPP_t",
                "ida": "IDA_t", "ipv": "IPV_t", "ian": "IAN_t", "inde": "INDE_t",
                "defasagem": "defasagem_t", "fase": "fase_t", "idade": "idade_t",
                "anos_na_pm": "anos_na_pm_t", "nota_mat": "nota_mat_t",
                "nota_por": "nota_por_t", "nota_ing": "nota_ing_t"}
        X = pd.DataFrame(index=df.index)
        for orig, dest in mapa.items():
            X[dest] = pd.to_numeric(df.get(orig), errors="coerce")
        X["genero_fem"] = (df.get("genero", pd.Series("", index=df.index))
                           .astype(str).str.strip().str.lower().eq("feminino").astype(int))
        probs = modelo.predict_proba(X[FEATURES])[:, 1]
        saida = pd.DataFrame({
            "aluno": df.get("nome", df.get("ra", pd.Series(range(1, len(df) + 1)))),
            "prob_risco": probs.round(3),
        }).sort_values("prob_risco", ascending=False)
        saida["classificacao"] = np.select(
            [saida.prob_risco >= 0.7, saida.prob_risco >= 0.4],
            ["🔴 Alto", "🟡 Moderado"], default="🟢 Baixo")
        return saida

    fonte = st.radio("Fonte dos dados",
                     ["Base PEDE 2024 (tratada do Excel do repositório)", "Enviar um CSV novo"],
                     horizontal=True)

    df = None
    if fonte.startswith("Base"):
        df = carrega_base_2024()
        if df is not None:
            st.caption(f"Base tratada em tempo real a partir de {ARQ_BASE}: "
                       f"{len(df)} alunos avaliados no PEDE 2024.")
        else:
            st.error(f"{ARQ_BASE} não encontrado na raiz do repositório nem na pasta app/.")
    else:
        st.markdown(
            "Colunas esperadas (uma linha por aluno): "
            "`IAA, IEG, IPS, IPP, IDA, IPV, IAN, INDE, defasagem, fase, idade, anos_na_pm, "
            "nota_mat, nota_por, nota_ing, genero` (Feminino/Masculino). "
            "Uma coluna `nome` ou `RA` é usada como identificador, se existir."
        )
        arq = st.file_uploader("Arquivo CSV", type="csv")
        if arq is not None:
            df = pd.read_csv(arq)

    if df is not None:
        saida = pontua(df)
        st.dataframe(saida, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Baixar resultado (CSV)",
                           saida.to_csv(index=False).encode("utf-8"),
                           "risco_defasagem.csv", "text/csv")
        n_alto = (saida.prob_risco >= 0.7).sum()
        n_mod = ((saida.prob_risco >= 0.4) & (saida.prob_risco < 0.7)).sum()
        st.caption(f"{n_alto} aluno(s) em risco alto | {n_mod} em risco moderado.")

st.divider()
st.caption("Datathon PosTech FIAP — Fase 5 · Associação Passos Mágicos · "
           "Modelo: Gradient Boosting calibrado (scikit-learn), validação temporal 2023→2024.")
