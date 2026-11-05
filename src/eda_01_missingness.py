"""EDA step 1: missingness profile.

Counts what is missing in each table, and checks whether missing values cluster
by year, item, municipality type, province or audit opinion.
Run: python src/eda_01_missingness.py [data/raw]
"""
import sys  # lets us read the folder path typed after the script name
import pathlib  # makes working with folder and file paths easy
import pandas as pd  # the table library we use for all the analysis

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "data" / "raw")  # the data folder: the one typed on the command line, or data/raw next to this project
K = ["demarcation.code", "financial_year_end.year"]  # the two columns that identify one municipality in one year

m = pd.read_parquet(R / "municipalities.parquet")  # load the municipalities table
a = pd.read_parquet(R / "audit_opinions.parquet")  # load the audit opinions table
u = pd.read_parquet(R / "uifwexp.parquet")  # load the irregular, fruitless, wasteful and unauthorised expenditure table

# ---------- Part 1: basic counts of missing values in the three small tables ----------
for n, d in [("muni", m), ("audit", a), ("uifw", u)]:  # go through each table with a short name for printing
    print("==", n, d.shape)  # print the table's name and its size (rows, columns)
    print(d.isna().sum().to_string())  # for each column, count the truly empty cells
    blank_like = d.astype(str).apply(lambda s: s.str.strip().isin(["", "None", "nan"]))  # also mark cells that are empty text or say "None"/"nan"
    print(blank_like.sum().to_string())  # count those per column too
print(m.columns.tolist())  # list the municipality table's column names
print(m["municipality.category"].value_counts())  # how many metros (A), local (B) and district (C) municipalities
print(a["financial_year_end.year"].value_counts().sort_index().to_string())  # how many audit opinions per year
print(a["opinion.code"].value_counts())  # how many of each kind of audit opinion
print(u["financial_year_end.year"].value_counts().sort_index().to_string())  # how many expenditure rows per year
print(u["item.code"].value_counts())  # how many rows of each expenditure type
dup_audit = a.duplicated(K).sum()  # count municipality-years that appear more than once in the audit table
dup_uifw = u.duplicated(K + ["item.code"]).sum()  # count municipality-year-items that appear more than once in the expenditure table
print(dup_audit, dup_uifw)  # both should be 0
print(u.amount.describe())  # summary of expenditure amounts: count, mean, min, max and so on
print((u.amount == 0).sum(), (u.amount < 0).sum())  # how many amounts are exactly zero, and how many are negative

# ---------- Part 2: where do the blank expenditure amounts cluster? ----------
u["miss"] = u.amount.isna()  # add a True/False column: is this amount blank?
print(u.groupby("financial_year_end.year").miss.mean().round(3).to_string())  # share of blanks in each year
print(u.groupby("item.code").miss.mean().round(3))  # share of blanks for each expenditure type
w = u.pivot_table(index=K, columns="item.code", values="miss", aggfunc="first")  # one row per municipality-year, one column per expenditure type, True where blank
blanks_per_row = w.sum(axis=1)  # how many of the three types are blank for each municipality-year
print("muni-years", len(w), " all3 missing", (blanks_per_row == 3).sum(), " some", ((blanks_per_row > 0) & (blanks_per_row < 3)).sum())  # all three blank vs only some blank
muni_cols = m[["municipality.demarcation_code", "municipality.category", "municipality.province_code"]]  # the municipality details we want to attach
x = u.merge(muni_cols, left_on="demarcation.code", right_on="municipality.demarcation_code", how="left")  # attach category and province to every expenditure row
unmatched = x["municipality.category"].isna()  # rows whose municipality wasn't found in the municipalities table
print("unmatched munis", unmatched.sum(), x.loc[unmatched, "demarcation.code"].unique()[:20])  # should be 0
print(x.groupby("municipality.category").miss.mean().round(3))  # share of blanks by municipality type
print(x.groupby("municipality.province_code").miss.mean().round(3).to_string())  # share of blanks by province
y = x.merge(a[K + ["opinion.code"]], on=K, how="left")  # attach the audit opinion to every expenditure row
print(y.groupby("opinion.code", dropna=False).miss.agg(["mean", "size"]).round(3))  # share of blanks (and row count) for each audit opinion

# ---------- Part 3: gaps in the audit opinions themselves ----------
print("audit munis", a["demarcation.code"].nunique(), "uifw munis", u["demarcation.code"].nunique(), "muni table", len(m))  # number of municipalities in each table
cur = set(a.loc[a["financial_year_end.year"] == 2024, "demarcation.code"])  # the municipalities that still exist (have a 2024 opinion)
years_covered = a[a["demarcation.code"].isin(cur)].groupby("demarcation.code").size()  # how many years of opinions each current municipality has
print("current munis years covered", years_covered.value_counts().sort_index().to_string())  # e.g. 242 have all 14 years
in_both = a.merge(u[K].drop_duplicates(), on=K, how="left", indicator=True)  # match audit rows to expenditure rows, marking which were found
print("audit keys missing in uifw", len(in_both.query('_merge=="left_only"')))  # audit municipality-years with no expenditure row
print(a[a["opinion.code"] == "outstanding"].groupby("financial_year_end.year").size().to_string())  # audits not finished, by year
print(a[a["opinion.report_url"].isna()].groupby("financial_year_end.year").size().to_string())  # missing report links, by year
print(u[u.amount < 0])  # show any negative expenditure amounts

# ---------- Part 4: which municipality-years have no audited financial statements? ----------
panel = a[a["financial_year_end.year"].between(2018, 2024)][K + ["opinion.code"]]  # audit opinions for 2018-2024, the years the new financial tables cover
print("panel muni-years 2018-24:", len(panel))  # how many municipality-years that is
for c in ["incexp_v2", "financial_position_v2", "aged_debtor_v2"]:  # go through the three large financial tables
    d = pd.read_parquet(R / f"{c}.parquet")  # load the table
    print("==", c, d.shape, d.columns.tolist())  # print its name, size and columns
    nulls = d.isna().sum()  # count empty cells per column
    print(" nulls:", nulls[nulls > 0].to_dict())  # show only the columns that have any
    print(" amount types:", d["amount_type.code"].value_counts().to_dict())  # count rows per amount type (AUDA = audited actual, ORGB = original budget...)
    au = d[d["amount_type.code"] == "AUDA"]  # keep only audited actual figures
    print(" AUDA years:", au["financial_year_end.year"].value_counts().sort_index().to_dict(), "items", au["item.code"].nunique())  # audited rows per year, and how many line items exist
    have = au[K].drop_duplicates()  # the municipality-years that have at least one audited figure
    have["has"] = 1  # mark them with 1
    p = panel.merge(have, on=K, how="left")  # attach that mark to every audit opinion
    p["has"] = p.has.fillna(0)  # municipality-years with no audited figures get 0
    print(" panel muni-years with NO audited figures:", int((p.has == 0).sum()), f"({(p.has == 0).mean():.1%})")  # count and share with nothing audited
    print("  by year:", p.groupby("financial_year_end.year").has.apply(lambda s: int((s == 0).sum())).to_dict())  # that count for each year
    print("  missing rate by opinion:", p.groupby("opinion.code").has.apply(lambda s: round((s == 0).mean(), 3)).to_dict())  # that share for each audit opinion
    cnt = au.groupby(K)["item.code"].nunique()  # how many line items each reporting municipality-year has
    print("  items per muni-year: median", cnt.median(), "min", cnt.min(), "max", cnt.max())  # typical, smallest and largest
    v = [col for col in d.columns if col.endswith(".sum")]  # the number columns in this table
    print("  zeros share:", {col: round((au[col] == 0).mean(), 3) for col in v[:3]})  # share of exact zeros in the first three number columns
