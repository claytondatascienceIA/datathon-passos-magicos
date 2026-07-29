"""
Datathon FIAP - Passos Mágicos
Limpeza e preparação dos dados do PEDE (2022, 2023, 2024).

Gera:
- data/panel_pede.csv          -> painel unificado aluno-ano
- data/pares_longitudinais.csv -> pares aluno (ano t -> ano t+1) p/ modelo
"""
import re
import unicodedata
import numpy as np
import pandas as pd

ARQUIVO_BASE = "BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx"


def norm_pedra(x):
    """Padroniza a pedra-conceito ('Agata' -> 'Ágata'; 'INCLUIR' -> NaN)."""
    if pd.isna(x):
        return np.nan
    x = unicodedata.normalize("NFKD", str(x).strip().title()).encode("ascii", "ignore").decode()
    return {"Agata": "Ágata", "Ametista": "Ametista", "Topazio": "Topázio", "Quartzo": "Quartzo"}.get(x, np.nan)


def extrai_fase(x):
    """Extrai a fase numérica de formatos distintos: '7', 'FASE 7', '7A', 'ALFA'."""
    x = str(x).strip().upper()
    if x == "ALFA":
        return 0
    m = re.search(r"(\d)", x)
    return int(m.group(1)) if m else np.nan


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _monta_ano(df, ano, cmap):
    out = pd.DataFrame()
    out["RA"] = df["RA"]
    out["ano"] = ano
    out["fase"] = df[cmap["fase"]].apply(extrai_fase)
    out["genero"] = df["Gênero"].replace({"Menina": "Feminino", "Menino": "Masculino"})
    out["idade"] = _to_num(df[cmap["idade"]])
    out["ano_ingresso"] = _to_num(df["Ano ingresso"])
    out["anos_na_pm"] = ano - out["ano_ingresso"]
    out["inst_ensino"] = df["Instituição de ensino"].astype(str).str.strip()
    for k in ["IAA", "IEG", "IPS", "IDA", "IPV", "IAN"]:
        out[k] = _to_num(df[k])
    out["IPP"] = _to_num(df["IPP"]) if "IPP" in df.columns else np.nan
    out["INDE"] = _to_num(df[cmap["inde"]])
    out["pedra"] = df[cmap["pedra"]].apply(norm_pedra)
    out["nota_mat"] = _to_num(df[cmap["mat"]])
    out["nota_por"] = _to_num(df[cmap["por"]])
    out["nota_ing"] = _to_num(df[cmap["ing"]])
    out["defasagem"] = _to_num(df[cmap["defas"]])
    out["atingiu_pv"] = (
        df["Atingiu PV"].map({"Sim": 1, "Não": 0}) if df["Atingiu PV"].notna().any() else np.nan
    )
    return out


def classifica_defasagem(d):
    if pd.isna(d):
        return np.nan
    if d >= 0:
        return "Em fase"
    if d >= -2:
        return "Moderada (-1 a -2)"
    return "Severa (≤ -3)"


def carrega_painel(caminho=ARQUIVO_BASE):
    """Lê as 3 abas do Excel e devolve o painel unificado aluno-ano."""
    xl = pd.ExcelFile(caminho)
    d22, d23, d24 = xl.parse("PEDE2022"), xl.parse("PEDE2023"), xl.parse("PEDE2024")

    p22 = _monta_ano(d22, 2022, dict(fase="Fase", idade="Idade 22", inde="INDE 22",
                                     pedra="Pedra 22", mat="Matem", por="Portug",
                                     ing="Inglês", defas="Defas"))
    p23 = _monta_ano(d23, 2023, dict(fase="Fase", idade="Idade", inde="INDE 2023",
                                     pedra="Pedra 2023", mat="Mat", por="Por",
                                     ing="Ing", defas="Defasagem"))
    p24 = _monta_ano(d24, 2024, dict(fase="Fase", idade="Idade", inde="INDE 2024",
                                     pedra="Pedra 2024", mat="Mat", por="Por",
                                     ing="Ing", defas="Defasagem"))

    # A aba 2022 não traz IPP: reconstruímos pela fórmula oficial do INDE (fases 0-7):
    # INDE = 0,1*IAN + 0,2*IDA + 0,2*IEG + 0,1*IAA + 0,1*IPS + 0,1*IPP + 0,2*IPV
    m = p22["fase"] <= 7
    rec = (p22["INDE"] - (0.1 * p22["IAN"] + 0.2 * p22["IDA"] + 0.2 * p22["IEG"]
                          + 0.1 * p22["IAA"] + 0.1 * p22["IPS"] + 0.2 * p22["IPV"])) / 0.1
    p22.loc[m, "IPP"] = rec[m].clip(0, 10).round(3)

    panel = pd.concat([p22, p23, p24], ignore_index=True)
    panel["defasado"] = np.where(panel["defasagem"].isna(), np.nan,
                                 (panel["defasagem"] < 0).astype(float))
    panel["nivel_defasagem"] = panel["defasagem"].apply(classifica_defasagem)
    return panel


def monta_pares(panel):
    """Pares longitudinais (features no ano t, desfecho no ano t+1) via RA."""
    pares = []
    for a0, a1 in [(2022, 2023), (2023, 2024)]:
        m = panel[panel.ano == a0].merge(panel[panel.ano == a1], on="RA",
                                         suffixes=("_t", "_t1"))
        pares.append(m)
    return pd.concat(pares, ignore_index=True)


if __name__ == "__main__":
    panel = carrega_painel()
    panel.to_csv("data/panel_pede.csv", index=False)
    pares = monta_pares(panel)
    pares.to_csv("data/pares_longitudinais.csv", index=False)
    print(f"Painel: {panel.shape} | Pares: {pares.shape}")
