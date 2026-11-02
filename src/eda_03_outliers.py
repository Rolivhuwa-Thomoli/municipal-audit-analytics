"""EDA step 2: outliers and anomalies.

Checks: robust z-score (median/MAD, |z| > 3.5) on log total expenditure within category;
year-on-year jumps > 5x; own-history check of the largest irregular expenditure values;
line items vs reported total; Benford first-digit MAD (Nigrini thresholds).
Run: python src/eda_03_outliers.py [data/raw]
"""
import sys, pathlib, numpy as np, pandas as pd

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "data" / "raw")
K = ["demarcation.code", "financial_year_end.year"]
muni = pd.read_parquet(R / "municipalities.parquet")[["municipality.demarcation_code", "municipality.category"]] \
    .rename(columns={"municipality.demarcation_code": "demarcation.code", "municipality.category": "category"})

# total expenditure per municipality-year = sum of audited expense line items 3100-4300
inc = pd.read_parquet(R / "incexp_v2.parquet")
auda = inc[inc["amount_type.code"] == "AUDA"]
wide = auda.pivot_table(index=K, columns="item.code", values="amount.sum", aggfunc="sum")
wide["tot_exp"] = wide[[c for c in wide.columns if "3100" <= c <= "4300"]].sum(axis=1, min_count=1)
rep = wide.dropna(subset=["4400"])
print("line items vs reported total, mismatch > 1%:",
      int(((rep.tot_exp - rep["4400"]).abs() / rep["4400"].abs() > 0.01).sum()), "of", len(rep))

te = wide["tot_exp"].reset_index().merge(muni, on="demarcation.code").sort_values(K)
te["ratio"] = te.tot_exp / te.groupby("demarcation.code").tot_exp.shift()
te["jump"] = (te.ratio > 5) | (te.ratio < 0.2)
print("YoY jumps > 5x by year:", te.groupby("financial_year_end.year").jump.sum().to_dict())
print("total expenditure < R20m by year:", te[te.tot_exp < 2e7].groupby("financial_year_end.year").size().to_dict())

pos = te[te.tot_exp > 0].copy()
pos["log"] = np.log10(pos.tot_exp)
med = pos.groupby("category").log.transform("median")
mad = (pos.log - med).abs().groupby(pos.category).transform("median")
pos["robust_z"] = 0.6745 * (pos.log - med) / mad
print("\nrobust |z| > 3.5:\n", pos[pos.robust_z.abs() > 3.5][K + ["category", "tot_exp", "robust_z"]].sort_values("robust_z").to_string(index=False))

u = pd.read_parquet(R / "uifwexp.parquet")
print("\nlargest irregular/fruitless/unauthorised values:\n",
      u.nlargest(6, "amount")[["demarcation.label", "financial_year_end.year", "item.code", "amount"]].to_string(index=False))
print("negative values:\n", u[u.amount < 0][["demarcation.label", "financial_year_end.year", "item.code", "amount"]].to_string(index=False))

BEN = np.log10(1 + 1 / np.arange(1, 10))
def benford(x, name):
    x = x[x >= 10]
    d = (x / 10 ** np.floor(np.log10(x))).astype(int)
    obs = np.bincount(d, minlength=10)[1:] / len(d)
    m = np.abs(obs - BEN).mean()
    verdict = "close" if m < .006 else "acceptable" if m < .012 else "marginal" if m < .015 else "nonconformity"
    print(f"Benford {name}: n={len(x):,}, MAD={m:.4f} ({verdict})")
print()
benford(auda["amount.sum"], "income & expenditure AUDA")
fp = pd.read_parquet(R / "financial_position_v2.parquet")
benford(fp[fp["amount_type.code"] == "AUDA"]["amount.sum"].dropna(), "balance sheet AUDA")
for item in ["irregular", "fruitless", "unauthorised"]:
    benford(u[u["item.code"] == item].amount.dropna(), item)
