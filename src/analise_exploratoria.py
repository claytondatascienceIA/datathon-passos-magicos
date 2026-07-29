"""
Datathon FIAP - Passos Mágicos
Análise exploratória: gera as figuras que respondem às perguntas de negócio.

Uso: python src/analise_exploratoria.py  (após rodar src/preparacao_dados.py)
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

# Paleta FIAP: magenta + preto + tons de apoio
AZUL, VERDE, LARANJA, VERM, CINZA = "#0D0D0D", "#ED145B", "#F26B8A", "#8A0C35", "#9E9E9E"
CORES_PEDRA = {"Quartzo": "#CFCFCF", "Ágata": "#8C8C8C", "Ametista": "#F26B8A", "Topázio": "#ED145B"}

panel = pd.read_csv("data/panel_pede.csv")
pairs = pd.read_csv("data/pares_longitudinais.csv")

# ---------------------------------------------------------------- Q1: defasagem
tab = panel.groupby(["ano", "nivel_defasagem"]).size().unstack()
tab = tab[["Em fase", "Moderada (-1 a -2)", "Severa (≤ -3)"]]
pct = tab.div(tab.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
bottom = np.zeros(len(pct))
for col, c in zip(pct.columns, [VERDE, LARANJA, VERM]):
    ax.bar(pct.index.astype(str), pct[col], bottom=bottom, label=col, color=c, width=0.55)
    for i, v in enumerate(pct[col]):
        if v > 4:
            ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold")
    bottom += pct[col].values
ax.set_title('Defasagem escolar (IAN): alunos "em fase" saltam de 30% para 54%', fontweight="bold")
ax.set_ylabel("% dos alunos"); ax.legend(loc="lower right", framealpha=0.9)
plt.tight_layout(); plt.savefig("figures/f1_defasagem_ano.png"); plt.close()

# ---------------------------------------------------------------- Q2: indicadores por ano
inds = ["IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "IAN", "INDE"]
med = panel.groupby("ano")[inds].mean()
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(inds)); w = 0.26
for i, (ano, c) in enumerate(zip([2022, 2023, 2024], [CINZA, AZUL, VERDE])):
    ax.bar(x + (i - 1) * w, med.loc[ano], w, label=str(ano), color=c)
ax.set_xticks(x); ax.set_xticklabels(inds); ax.set_ylim(0, 10)
ax.set_title("Indicadores médios por ano — IDA oscila, IAN melhora consistentemente", fontweight="bold")
ax.legend(); plt.tight_layout(); plt.savefig("figures/f2_indicadores_ano.png"); plt.close()

# ---------------------------------------------------------------- Q2: IDA por fase
fig, ax = plt.subplots(figsize=(8, 4.5))
piv = panel[panel.fase <= 7].groupby(["fase", "ano"])["IDA"].mean().unstack()
for ano, c in zip([2022, 2023, 2024], [CINZA, AZUL, VERDE]):
    ax.plot(piv.index, piv[ano], marker="o", label=str(ano), color=c, lw=2)
ax.set_xlabel("Fase"); ax.set_ylabel("IDA médio"); ax.set_ylim(4, 9)
ax.set_title("Desempenho acadêmico (IDA) por fase: queda nas fases intermediárias", fontweight="bold")
ax.legend(); plt.tight_layout(); plt.savefig("figures/f3_ida_fase.png"); plt.close()

# ---------------------------------------------------------------- Q3: IEG x IDA x IPV
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
d = panel.dropna(subset=["IEG", "IDA", "IPV"])
for ax, ycol, tit in [(axes[0], "IDA", "IEG × IDA"), (axes[1], "IPV", "IEG × IPV")]:
    ax.scatter(d.IEG, d[ycol], s=6, alpha=0.25, color=AZUL)
    z = np.polyfit(d.IEG, d[ycol], 1); xs = np.linspace(0, 10, 50)
    ax.plot(xs, np.polyval(z, xs), color=VERM, lw=2)
    ax.set_title(f"{tit}  (r = {d.IEG.corr(d[ycol]):.2f})", fontweight="bold")
    ax.set_xlabel("IEG (engajamento)"); ax.set_ylabel(ycol)
plt.suptitle("Engajamento anda junto com aprendizagem e ponto de virada", y=1.02, fontweight="bold")
plt.tight_layout(); plt.savefig("figures/f4_ieg_ida_ipv.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Q4: IAA x IDA
fig, ax = plt.subplots(figsize=(7, 4.5))
d = panel.dropna(subset=["IAA", "IDA"])
ax.scatter(d.IAA, d.IDA, s=6, alpha=0.25, color=AZUL)
z = np.polyfit(d.IAA, d.IDA, 1); xs = np.linspace(0, 10, 50)
ax.plot(xs, np.polyval(z, xs), color=VERM, lw=2)
ax.set_xlabel("IAA (autoavaliação)"); ax.set_ylabel("IDA (desempenho)")
ax.set_title(f"Autoavaliação pouco reflete o desempenho real (r = {d.IAA.corr(d.IDA):.2f})",
             fontweight="bold")
plt.tight_layout(); plt.savefig("figures/f5_iaa_ida.png"); plt.close()

# ---------------------------------------------------------------- Q5: IPS antecede queda
q = pairs.copy()
q["dIDA"] = q.IDA_t1 - q.IDA_t
q = q.dropna(subset=["IPS_t", "dIDA"])
q["quartil"] = pd.qcut(q.IPS_t, 4, labels=["Q1 (baixo)", "Q2", "Q3", "Q4 (alto)"])
prob = q.groupby("quartil", observed=True).apply(lambda g: (g.dIDA <= -1).mean() * 100,
                                                 include_groups=False)
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.bar(prob.index.astype(str), prob.values, color=[VERM, LARANJA, VERDE, LARANJA], width=0.55)
for i, v in enumerate(prob.values):
    ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontweight="bold")
ax.set_ylabel("% com queda de IDA ≥ 1 ponto no ano seguinte")
ax.set_xlabel("Quartil do IPS no ano anterior")
ax.set_title("IPS baixo hoje antecipa maior risco de queda acadêmica amanhã", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/f6_ips_queda.png"); plt.close()

# ---------------------------------------------------------------- Q6: IPP x defasagem
fig, ax = plt.subplots(figsize=(7, 4.3))
g = panel.dropna(subset=["IPP", "nivel_defasagem"])
ordem = ["Em fase", "Moderada (-1 a -2)", "Severa (≤ -3)"]
bp = ax.boxplot([g[g.nivel_defasagem == o]["IPP"] for o in ordem], tick_labels=ordem,
                patch_artist=True, showfliers=False, medianprops=dict(color="black"))
for patch, c in zip(bp["boxes"], [VERDE, LARANJA, VERM]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_ylabel("IPP (psicopedagógico)")
ax.set_title("IPP confirma o IAN: quanto maior a defasagem, menor o IPP", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/f7_ipp_defasagem.png"); plt.close()

# ---------------------------------------------------------------- Q7: drivers do IPV
fig, ax = plt.subplots(figsize=(7, 4.3))
d = panel[["IPV", "IDA", "IEG", "IPP", "IAA", "IPS", "IAN"]].dropna()
corr = d.corr()["IPV"].drop("IPV").sort_values()
ax.barh(corr.index, corr.values, color=[AZUL if v < 0.5 else VERDE for v in corr.values])
for i, (k, v) in enumerate(corr.items()):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontweight="bold")
ax.set_xlim(0, 1); ax.set_xlabel("Correlação com IPV")
ax.set_title("Ponto de virada: puxado por aprendizagem, engajamento e psicopedagógico",
             fontweight="bold")
plt.tight_layout(); plt.savefig("figures/f8_ipv_drivers.png"); plt.close()

# ---------------------------------------------------------------- Q10: pedras por ano
tab = panel.groupby(["ano", "pedra"]).size().unstack()[["Quartzo", "Ágata", "Ametista", "Topázio"]]
pct = tab.div(tab.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
bottom = np.zeros(len(pct))
for col in pct.columns:
    ax.bar(pct.index.astype(str), pct[col], bottom=bottom, label=col,
           color=CORES_PEDRA[col], width=0.55)
    for i, v in enumerate(pct[col]):
        if v > 5:
            ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold")
    bottom += pct[col].values
ax.set_ylabel("% dos alunos avaliados")
ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.set_title("Topázio dobra em 2 anos (15% → 31%): o programa move alunos para cima",
             fontweight="bold")
plt.tight_layout(); plt.savefig("figures/f9_pedras_ano.png"); plt.close()

# ---------------------------------------------------------------- Q8: combinações -> INDE
d = panel[["INDE", "IDA", "IEG", "IPS", "IPP"]].dropna().copy()
for c in ["IDA", "IEG", "IPS", "IPP"]:
    d[f"{c}_hi"] = d[c] >= d[c].median()
d["n_altos"] = d[[f"{c}_hi" for c in ["IDA", "IEG", "IPS", "IPP"]]].sum(axis=1)
med = d.groupby("n_altos")["INDE"].mean()
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.bar(med.index.astype(str), med.values, color=AZUL, width=0.55)
for i, v in enumerate(med.values):
    ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_xlabel("Nº de indicadores acima da mediana (IDA, IEG, IPS, IPP)")
ax.set_ylabel("INDE médio"); ax.set_ylim(0, 10)
ax.set_title("Efeito cumulativo: cada indicador forte a mais eleva o INDE", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/f10_combinacoes_inde.png"); plt.close()

print("Figuras geradas em figures/")
