"""EDA step 1b: chi-square tests of independence between missingness and audit opinion.

The question: is "this record is missing" linked to the municipality's audit opinion?
H0 (the starting assumption we test): missingness and audit opinion are unrelated.
If data were missing completely at random (MCAR), H0 would be true.
Effect size: Cramer's V (0 = no link, 1 = perfect link).
Run: python src/eda_02_chisquare_missingness.py [data/raw]
"""
import sys  # lets us read the folder path typed after the script name
import pathlib  # makes working with folder and file paths easy
import numpy as np  # fast maths on lists of numbers
import pandas as pd  # the table library we use for the analysis
from scipy.stats import chi2_contingency  # the chi-square test of independence

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "data" / "raw")  # the data folder: the one typed on the command line, or data/raw next to this project
K = ["demarcation.code", "financial_year_end.year"]  # the two columns that identify one municipality in one year
a = pd.read_parquet(R / "audit_opinions.parquet")[K + ["opinion.code"]]  # load audit opinions, keeping only municipality, year and opinion


def test(name, crosstab, unit):
    """Run the chi-square test on a table of counts (opinion x missing yes/no) and print the results."""
    chi2, p, dof, exp = chi2_contingency(crosstab)  # the test: chi-square statistic, p-value, degrees of freedom, and the counts we'd expect if H0 were true
    n = crosstab.to_numpy().sum()  # total number of observations in the table
    v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))  # Cramer's V: how strong the link is, on a 0-1 scale
    small = (exp < 5).mean()  # share of cells where the expected count is under 5 (Cochran's rule: should be at most 20%)

    # Permutation test: a second check that doesn't rely on Cochran's rule.
    # We shuffle the "missing" labels at random many times and see how often chance alone gives a chi-square as big as ours.
    obs = crosstab.reset_index().melt(id_vars=crosstab.index.name, var_name="miss", value_name="k")  # turn the count table into a long list: opinion, missing yes/no, count
    lab = np.repeat(obs[crosstab.index.name].to_numpy(), obs["k"])  # rebuild one opinion label per observation (repeat each label by its count)
    miss = np.repeat(obs["miss"].to_numpy(), obs["k"])  # and one missing yes/no per observation, lined up with the labels
    rng = np.random.default_rng(0)  # a random number generator with a fixed seed, so results are the same every run
    hits = 0  # how many shuffles beat our real chi-square
    B = 2000  # number of shuffles
    for _ in range(B):  # repeat 2,000 times
        shuffled = rng.permutation(miss)  # randomly reorder the missing yes/no labels, breaking any real link with opinion
        hits += chi2_contingency(pd.crosstab(lab, shuffled))[0] >= chi2  # count it if this random version scores at least as high as the real data
    p_perm = (hits + 1) / (B + 1)  # the permutation p-value (adding 1 avoids ever reporting exactly 0)

    rate = (crosstab[True] / crosstab.sum(axis=1) * 100).round(1)  # percentage missing for each audit opinion
    print(f"\n== {name}  (n = {n:,} {unit})")  # heading for this test
    print(f"chi2 = {chi2:.1f}, df = {dof}, p = {p:.2e}, Cramér's V = {v:.2f}, min expected = {exp.min():.1f}, cells < 5: {small:.0%}, permutation p = {p_perm:.4f}")  # all the test results
    print("missing % by opinion:", rate.to_dict())  # the missing rate for each opinion
    return dict(table=name, n=n, chi2=round(chi2, 1), df=dof, p=p, cramers_v=round(v, 2),  # hand back the results as one row
                cells_below_5=f"{small:.0%}", p_perm=round(p_perm, 4))  # (continued) for the summary table at the end


rows = []  # collects one results row per test
panel = a[a["financial_year_end.year"].between(2018, 2024)]  # audit opinions for 2018-2024, the years the new financial tables cover

for cube in ["incexp_v2", "financial_position_v2"]:  # tests 1 and 2: income statement and balance sheet
    d = pd.read_parquet(R / f"{cube}.parquet")  # load the table
    have = d[d["amount_type.code"] == "AUDA"][K].drop_duplicates().assign(has=True)  # municipality-years that have audited figures, marked True
    p = panel.merge(have, on=K, how="left")  # attach that mark to each audit opinion; no audited figures leaves it empty
    table = pd.crosstab(p["opinion.code"], p["has"].isna())  # count table: audit opinion x (audited figures missing: True/False)
    rows.append(test(f"{cube}: no audited statements, 2018-2024", table, "municipality-years"))  # run the test and keep the results

d = pd.read_parquet(R / "aged_debtor_v2.parquet")  # test 3: load the debtors table
have = d[d["amount_type.code"] == "AUDA"][K].drop_duplicates().assign(has=True)  # municipality-years with audited debtor figures
p = a[a["financial_year_end.year"].between(2020, 2024)].merge(have, on=K, how="left")  # audited debtors only exist from 2020, so test 2020-2024
table = pd.crosstab(p["opinion.code"], p["has"].isna())  # count table: audit opinion x debtors missing
rows.append(test("aged_debtor_v2: no audited debtors, 2020-2024", table, "municipality-years"))  # run the test

u = pd.read_parquet(R / "uifwexp.parquet")  # test 4: load irregular, fruitless, wasteful and unauthorised expenditure
u = u[u["financial_year_end.year"] <= 2017].merge(a, on=K)  # blanks only exist up to 2017; attach the audit opinion to each amount
table = pd.crosstab(u["opinion.code"], u["amount"].isna())  # count table: audit opinion x amount blank
rows.append(test("uifwexp: blank amount, 2012-2017", table, "amounts"))  # run the test

print("\n", pd.DataFrame(rows).to_string(index=False))  # print all four results together as one summary table
