"""EDA step 2: outliers and anomalies.

Checks:
1. Do the expense line items add up to the reported total?
2. Year-on-year jumps: total expenditure growing or shrinking more than 5 times in one year.
3. Robust z-score (median and MAD) on log total expenditure, within each municipality type; flag |z| > 3.5.
4. The largest and any negative irregular/fruitless/unauthorised expenditure values.
5. Benford's law: do leading digits follow the pattern expected of natural financial data?
Run: python src/eda_03_outliers.py [data/raw]
"""
import sys  # lets us read the folder path typed after the script name
import pathlib  # makes working with folder and file paths easy
import numpy as np  # fast maths on lists of numbers
import pandas as pd  # the table library we use for the analysis

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "data" / "raw")  # the data folder: the one typed on the command line, or data/raw next to this project
K = ["demarcation.code", "financial_year_end.year"]  # the two columns that identify one municipality in one year
muni = pd.read_parquet(R / "municipalities.parquet")[["municipality.demarcation_code", "municipality.category"]]  # load municipality codes and types (A metro, B local, C district)
muni = muni.rename(columns={"municipality.demarcation_code": "demarcation.code", "municipality.category": "category"})  # rename columns so they match the other tables

# ---------- Check 1: total expenditure, and whether the line items add up ----------
inc = pd.read_parquet(R / "incexp_v2.parquet")  # load the income and expenditure table
auda = inc[inc["amount_type.code"] == "AUDA"]  # keep only audited actual figures
wide = auda.pivot_table(index=K, columns="item.code", values="amount.sum", aggfunc="sum")  # one row per municipality-year, one column per line item
expense_cols = [c for c in wide.columns if "3100" <= c <= "4300"]  # the expense line items (codes 3100 to 4300)
wide["tot_exp"] = wide[expense_cols].sum(axis=1, min_count=1)  # add the expense items up to get total expenditure (stays empty if there are none)
rep = wide.dropna(subset=["4400"])  # rows where the municipality also reported its own total (item 4400)
mismatch = (rep.tot_exp - rep["4400"]).abs() / rep["4400"].abs() > 0.01  # True where our sum differs from their total by more than 1%
print("line items vs reported total, mismatch > 1%:", int(mismatch.sum()), "of", len(rep))  # how many don't add up

# ---------- Check 2: big year-on-year jumps and implausibly small totals ----------
te = wide["tot_exp"].reset_index().merge(muni, on="demarcation.code").sort_values(K)  # total expenditure per municipality-year, with its type, in year order
te["ratio"] = te.tot_exp / te.groupby("demarcation.code").tot_exp.shift()  # this year's total divided by last year's, for the same municipality
te["jump"] = (te.ratio > 5) | (te.ratio < 0.2)  # True if it grew more than 5 times or shrank to under a fifth
print("YoY jumps > 5x by year:", te.groupby("financial_year_end.year").jump.sum().to_dict())  # count of jumps in each year
print("total expenditure < R20m by year:", te[te.tot_exp < 2e7].groupby("financial_year_end.year").size().to_dict())  # totals under R20 million, too small for any municipality

# ---------- Check 3: robust z-score ----------
pos = te[te.tot_exp > 0].copy()  # keep positive totals only (a log can't be taken of zero or negatives)
pos["log"] = np.log10(pos.tot_exp)  # log scale, so a R40bn metro and a R100m town can be compared
med = pos.groupby("category").log.transform("median")  # the median (middle value) for each municipality type
mad = (pos.log - med).abs().groupby(pos.category).transform("median")  # the MAD: median distance from that median, a spread measure that extremes can't inflate
pos["robust_z"] = 0.6745 * (pos.log - med) / mad  # the robust z-score: how many "typical distances" from the middle (0.6745 makes it comparable to a normal z-score)
flagged = pos[pos.robust_z.abs() > 3.5][K + ["category", "tot_exp", "robust_z"]].sort_values("robust_z")  # rows beyond 3.5, the standard cut-off (Iglewicz and Hoaglin 1993)
print("\nrobust |z| > 3.5:\n", flagged.to_string(index=False))  # show them

# ---------- Check 4: extreme and negative irregular expenditure ----------
u = pd.read_parquet(R / "uifwexp.parquet")  # load irregular, fruitless, wasteful and unauthorised expenditure
cols = ["demarcation.label", "financial_year_end.year", "item.code", "amount"]  # the columns worth showing
print("\nlargest irregular/fruitless/unauthorised values:\n", u.nlargest(6, "amount")[cols].to_string(index=False))  # the 6 biggest amounts
print("negative values:\n", u[u.amount < 0][cols].to_string(index=False))  # any amounts below zero

# ---------- Check 5: Benford's law ----------
BEN = np.log10(1 + 1 / np.arange(1, 10))  # the share of numbers expected to start with 1, 2, ... 9 (1 is most common at about 30%)


def benford(x, name):
    """Compare the leading digits of some amounts with Benford's law and print the verdict."""
    x = x[x >= 10]  # use amounts of at least 10, so every number has a clear leading digit
    d = (x / 10 ** np.floor(np.log10(x))).astype(int)  # get each number's first digit (e.g. 4,812 becomes 4.812, then 4)
    obs = np.bincount(d, minlength=10)[1:] / len(d)  # the share of numbers starting with each digit 1-9
    m = np.abs(obs - BEN).mean()  # MAD: the average gap between what we see and what Benford expects
    verdict = "close" if m < .006 else "acceptable" if m < .012 else "marginal" if m < .015 else "nonconformity"  # Nigrini's thresholds for first digits
    print(f"Benford {name}: n={len(x):,}, MAD={m:.4f} ({verdict})")  # print the result


print()  # a blank line before the Benford results
benford(auda["amount.sum"], "income & expenditure AUDA")  # test the audited income and expenditure amounts
fp = pd.read_parquet(R / "financial_position_v2.parquet")  # load the balance sheet table
benford(fp[fp["amount_type.code"] == "AUDA"]["amount.sum"].dropna(), "balance sheet AUDA")  # test the audited balance sheet amounts
for item in ["irregular", "fruitless", "unauthorised"]:  # go through the three expenditure types
    benford(u[u["item.code"] == item].amount.dropna(), item)  # test each one separately
