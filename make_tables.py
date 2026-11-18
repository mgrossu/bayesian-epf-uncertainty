"""make_tables.py -- LaTeX tables from a results folder (so no numbers are typed by hand)
    python make_tables.py results_synth      (writes table_layers.tex, table_window_eta.tex there)
"""
import sys, json, os
import pandas as pd

d = sys.argv[1]
res = pd.read_csv(os.path.join(d, "test_results.csv"))
ch = json.load(open(os.path.join(d, "chosen_settings.json")))
W, SEL = ch["window"], ch["single_model"]
f = lambda v, k: "" if pd.isna(v) else {"p": f"{100*v:.1f}", "w": f"{v:.2f}", "l": f"{v:.3f}", "e": f"{v:.2f}"}[k]

# ---- table 1: every model x layers 1-3, and the mixture for every eta (chosen window)
r = res[res.window == W]
lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Model & Layer & Cov.\ 80\% & Cov.\ 95\% & Width 80\% & Width 95\% & LPS & RMSE & MAE \\",
         r"\midrule"]
for m in [x for x in r.model.unique() if x != "mixture"]:
    for layer in ("L1 point", "L2 plug-in", "L3 single PPD"):
        q = r[(r.model == m) & (r.layer == layer)].iloc[0]
        name = m + ("$^*$" if m == SEL else "")
        lines.append(f"{name} & {layer.replace('L1 point','1 point').replace('L2 plug-in','2 plug-in').replace('L3 single PPD','3 posterior pred.')} & "
                     f"{f(q.cov80,'p')} & {f(q.cov95,'p')} & {f(q.width80,'w')} & {f(q.width95,'w')} & "
                     f"{f(q.lps,'l')} & {f(q.rmse,'e')} & {f(q.mae,'e')} \\\\")
    lines.append(r"\addlinespace")
for _, q in r[r.layer == "L4 mixture PPD"].iterrows():
    lines.append(f"mixture & 4 BMA, $\\eta={q.eta:g}$ & {f(q.cov80,'p')} & {f(q.cov95,'p')} & {f(q.width80,'w')} & "
                 f"{f(q.width95,'w')} & {f(q.lps,'l')} & {f(q.rmse,'e')} & {f(q.mae,'e')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(d, "table_layers.tex"), "w").write("\n".join(lines) + "\n")

# ---- table 2: mixture LPS by window x eta (test)
m = res[res.layer == "L4 mixture PPD"].copy()
m["eta"] = m["eta"].astype(float)
piv = m.pivot(index="eta", columns="window", values="lps").sort_index(ascending=False)
cols = list(piv.columns)
lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
         "$\\eta$ & " + " & ".join(cols) + r" \\", r"\midrule"]
for eta, row in piv.iterrows():
    lines.append(f"{eta:g} & " + " & ".join(f"{v:.3f}" for v in row.values) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(d, "table_window_eta.tex"), "w").write("\n".join(lines) + "\n")
print("wrote tables in", d, "| window", W, "| validation-chosen single model", SEL)
