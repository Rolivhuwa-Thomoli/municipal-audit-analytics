"""EDA step 1b: chi-square tests of independence between missingness and audit opinion.

H0: whether a record is missing is independent of the municipality's audit opinion
(a necessary condition for MCAR). Effect size: Cramér's V.
Run: python src/eda_02_chisquare_missingness.py [data/raw]
"""
import sys, pathlib, numpy as np, pandas as pd
from scipy.stats import chi2_contingency

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "data" / "raw")
K = ["demarcation.code", "financial_year_end.year"]
a = pd.read_parquet(R / "audit_opinions.parquet")[K + ["opinion.code"]]

def test(name, crosstab, unit):
    chi2, p, dof, exp = chi2_contingency(crosstab)
    n = crosstab.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
    small = (exp < 5).mean()  # Cochran's rule: at most 20% of cells with expected count < 5
    # Monte Carlo permutation p-value, robust to small expected counts
    obs = crosstab.reset_index().melt(id_vars=crosstab.index.name, var_name="miss", value_name="k")
    lab = np.repeat(obs[crosstab.index.name].to_numpy(), obs["k"]); miss = np.repeat(obs["miss"].to_numpy(), obs["k"])
    rng = np.random.default_rng(0); hits = 0; B = 2000
    for _ in range(B):
        hits += chi2_contingency(pd.crosstab(lab, rng.permutation(miss)))[0] >= chi2
    p_perm = (hits + 1) / (B + 1)
    rate = (crosstab[True] / crosstab.sum(axis=1) * 100).round(1)
    print(f"\n== {name}  (n = {n:,} {unit})")
    print(f"chi2 = {chi2:.1f}, df = {dof}, p = {p:.2e}, Cramér's V = {v:.2f}, min expected = {exp.min():.1f}, cells < 5: {small:.0%}, permutation p = {p_perm:.4f}")
    print("missing % by opinion:", rate.to_dict())
    return dict(table=name, n=n, chi2=round(chi2, 1), df=dof, p=p, cramers_v=round(v, 2), cells_below_5=f"{small:.0%}", p_perm=round(p_perm, 4))

rows = []
panel = a[a["financial_year_end.year"].between(2018, 2024)]
for cube in ["incexp_v2", "financial_position_v2"]:
    d = pd.read_parquet(R / f"{cube}.parquet")
    have = d[d["amount_type.code"] == "AUDA"][K].drop_duplicates().assign(has=True)
    p = panel.merge(have, on=K, how="left")
    rows.append(test(f"{cube}: no audited statements, 2018-2024", pd.crosstab(p["opinion.code"], p["has"].isna()), "municipality-years"))

d = pd.read_parquet(R / "aged_debtor_v2.parquet")
have = d[d["amount_type.code"] == "AUDA"][K].drop_duplicates().assign(has=True)
p = a[a["financial_year_end.year"].between(2020, 2024)].merge(have, on=K, how="left")
rows.append(test("aged_debtor_v2: no audited debtors, 2020-2024", pd.crosstab(p["opinion.code"], p["has"].isna()), "municipality-years"))

u = pd.read_parquet(R / "uifwexp.parquet")
u = u[u["financial_year_end.year"] <= 2017].merge(a, on=K)
rows.append(test("uifwexp: blank amount, 2012-2017", pd.crosstab(u["opinion.code"], u["amount"].isna()), "amounts"))

print("\n", pd.DataFrame(rows).to_string(index=False))
