"""EDA step 3: charts for the exploratory data analysis findings.

Draws one chart for each main EDA finding and saves it as a PNG file:
1. Missing audited statements, by audit opinion (missingness is not random).
2. Blank and zero irregular-expenditure amounts, by year (the 2018 recording change).
3. Robust z-scores of total expenditure, by municipality type (outliers beyond 3.5).
4. Benford's law: observed leading digits against the expected pattern.
5. Serial correlation: share non-reliable next year, by this year's opinion.
Run: python src/eda_04_visuals.py [data/raw] [docs/eda]
"""
import sys  # lets us read the folder paths typed after the script name
import pathlib  # makes working with folder and file paths easy
import numpy as np  # fast maths on lists of numbers
import pandas as pd  # the table library we use for the analysis
import matplotlib  # the charting library
matplotlib.use("Agg")  # draw charts straight to files, without opening a window
import matplotlib.pyplot as plt  # the part of matplotlib that draws charts

ROOT = pathlib.Path(__file__).resolve().parents[1]  # the project folder (one level above src)
R = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw"  # the data folder: typed on the command line, or data/raw
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "eda"  # where to save the charts: typed, or docs/eda
OUT.mkdir(parents=True, exist_ok=True)  # create the output folder if it does not exist yet
K = ["demarcation.code", "financial_year_end.year"]  # the two columns that identify one municipality in one year

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b8b7b0", "#0b0b0b", "#52514e"  # chart colours: two series colours, a neutral grey, and two text colours
plt.rcParams.update({  # general chart style, applied to every chart below
    "font.size": 10,  # normal text size
    "axes.spines.top": False,  # remove the top border line of each chart
    "axes.spines.right": False,  # remove the right border line of each chart
    "axes.edgecolor": GREY,  # light grey for the remaining axis lines
    "axes.labelcolor": MUTED,  # grey for axis titles
    "xtick.color": MUTED,  # grey for the x-axis numbers
    "ytick.color": MUTED,  # grey for the y-axis numbers
    "axes.titleweight": "bold",  # bold chart titles
    "axes.titlelocation": "left",  # titles start on the left
    "figure.dpi": 150,  # sharp image resolution
})
ORDER = ["unqualified", "unqualified_emphasis_of_matter", "qualified", "adverse", "disclaimer", "outstanding"]  # audit opinions from best to worst
NAMES = {"unqualified": "Unqualified (clean)", "unqualified_emphasis_of_matter": "Unqualified with emphasis",  # short readable names for each opinion
         "qualified": "Qualified", "adverse": "Adverse", "disclaimer": "Disclaimer", "outstanding": "Outstanding"}  # (continued)
NON_RELIABLE = {"adverse", "disclaimer", "outstanding"}  # the opinions grouped as "non-reliable" (decision 12)


def save(fig, name):
    """Tidy the layout, save the chart as a PNG and close it."""
    fig.tight_layout()  # stop labels from overlapping or being cut off
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")  # save the file with a white background
    plt.close(fig)  # free the memory used by the chart
    print("saved", OUT / name)  # report where it was saved


def opinion_bars(ax, values, colours, fmt):
    """Draw horizontal bars, one per audit opinion (best at top), with the value written at the end of each bar."""
    labels = [NAMES[o] for o in values.index]  # readable name for each bar
    y = np.arange(len(values))[::-1]  # bar positions, reversed so the best opinion is at the top
    ax.barh(y, values.values, color=colours, height=0.6)  # draw the bars
    ax.set_yticks(y, labels)  # put the opinion names next to the bars
    for yi, v in zip(y, values.values):  # go through each bar
        ax.text(v, yi, " " + fmt.format(v), va="center", color=INK, fontsize=9)  # write its value just after the end of the bar
    ax.xaxis.set_visible(False)  # hide the x-axis, because each bar is labelled
    ax.spines["bottom"].set_visible(False)  # and hide its line


a = pd.read_parquet(R / "audit_opinions.parquet")[K + ["opinion.code"]]  # load audit opinions: municipality, year and opinion

# ---------- Chart 1: missing audited statements by audit opinion ----------
inc = pd.read_parquet(R / "incexp_v2.parquet", columns=K + ["amount_type.code"])  # load only the columns we need from the income and expenditure table
have = inc[inc["amount_type.code"] == "AUDA"][K].drop_duplicates().assign(has=True)  # municipality-years that have audited figures
panel = a[a["financial_year_end.year"].between(2018, 2024)].merge(have, on=K, how="left")  # attach that mark to each opinion, 2018-2024
rate = panel["has"].isna().groupby(panel["opinion.code"]).mean().reindex(ORDER) * 100  # percentage with no audited statements, per opinion
colours = [ORANGE if o in NON_RELIABLE else BLUE for o in rate.index]  # orange for non-reliable opinions, blue for reliable ones
fig, ax = plt.subplots(figsize=(7, 3.4))  # create an empty chart
opinion_bars(ax, rate, colours, "{:.1f}%")  # draw the bars with percentage labels
ax.set_title("Municipalities with no audited financial statements, by audit opinion (2018-2024)")  # chart title
fig.text(0.01, -0.02, "Blue = reliable opinion, orange = non-reliable. Chi-square = 163.7, p < 0.001, Cramér's V = 0.30.", color=MUTED, fontsize=8)  # a note under the chart
save(fig, "01_missing_statements_by_opinion.png")  # save it

# ---------- Chart 2: blank and zero irregular-expenditure amounts by year ----------
u = pd.read_parquet(R / "uifwexp.parquet")  # load irregular, fruitless, wasteful and unauthorised expenditure
by_year = u.groupby("financial_year_end.year").agg(blank=("amount", lambda s: s.isna().mean() * 100),  # percentage of amounts left blank each year
                                                   zero=("amount", lambda s: (s == 0).mean() * 100))  # percentage of amounts recorded as exactly zero each year
fig, ax = plt.subplots(figsize=(7, 3.4))  # create an empty chart
ax.plot(by_year.index, by_year.blank, color=ORANGE, lw=2, marker="o", ms=5, label="Blank")  # line for blank amounts
ax.plot(by_year.index, by_year.zero, color=BLUE, lw=2, marker="o", ms=5, label="Zero")  # line for zero amounts
ax.axvline(2017.5, color=GREY, ls="--", lw=1)  # dashed line between 2017 and 2018, where the recording rule changed
ax.text(2017.6, ax.get_ylim()[1] * 0.92, "Recording change\n(2018)", color=MUTED, fontsize=8, va="top")  # label the dashed line
ax.set_ylabel("% of amounts")  # y-axis title
ax.legend(frameon=False, loc="upper right")  # legend without a box, in the empty top-right corner
ax.grid(axis="y", color="#e6e5e0", lw=0.8)  # light horizontal grid lines
ax.set_axisbelow(True)  # keep the grid behind the lines
ax.set_title("Irregular-expenditure amounts recorded as blank or zero, by year")  # chart title
fig.text(0.01, -0.02, "Blanks stop and zeros start in 2018, so the earlier blanks are most likely zeros (decision 7).", color=MUTED, fontsize=8)  # a note under the chart
save(fig, "02_uifw_blank_vs_zero_by_year.png")  # save it

# ---------- Chart 3: robust z-scores of total expenditure ----------
full = pd.read_parquet(R / "incexp_v2.parquet")  # load the whole income and expenditure table
auda = full[full["amount_type.code"] == "AUDA"]  # keep only audited actual figures
wide = auda.pivot_table(index=K, columns="item.code", values="amount.sum", aggfunc="sum")  # one row per municipality-year, one column per line item
exp_cols = [c for c in wide.columns if "3100" <= c <= "4300"]  # the expense line items
te = wide[exp_cols].sum(axis=1, min_count=1).rename("tot_exp").reset_index()  # total expenditure per municipality-year
muni = pd.read_parquet(R / "municipalities.parquet")[["municipality.demarcation_code", "municipality.category"]]  # municipality codes and types
te = te.merge(muni, left_on="demarcation.code", right_on="municipality.demarcation_code")  # attach each municipality's type
te = te[(te.tot_exp > 0) & (te["financial_year_end.year"] >= 2019)].copy()  # positive totals from 2019 (2018 is excluded, decision 10)
te["log"] = np.log10(te.tot_exp)  # log scale, so metros and small towns can be compared
med = te.groupby("municipality.category").log.transform("median")  # median per municipality type
mad = (te.log - med).abs().groupby(te["municipality.category"]).transform("median")  # MAD (median absolute deviation) per type
te["z"] = 0.6745 * (te.log - med) / mad  # robust z-score (Iglewicz and Hoaglin 1993)
cats = [("A", "Metro (A)"), ("B", "Local (B)"), ("C", "District (C)")]  # municipality types and their readable names
fig, ax = plt.subplots(figsize=(7, 3.6))  # create an empty chart
rng = np.random.default_rng(0)  # random numbers with a fixed seed, so the chart is the same every run
for i, (code, label) in enumerate(cats):  # go through each municipality type
    z = te.loc[te["municipality.category"] == code, "z"]  # robust z-scores for this type
    jitter = rng.uniform(-0.25, 0.25, len(z))  # small random sideways shift so points do not sit on top of each other
    out = z.abs() > 3.5  # True for points beyond the threshold
    ax.scatter(z[~out], i + jitter[~out], s=10, color=BLUE, alpha=0.35, linewidths=0)  # normal points in light blue
    ax.scatter(z[out], i + jitter[out], s=30, color=ORANGE, edgecolors="white", linewidths=1)  # flagged points in orange
ax.set_yticks(range(len(cats)), [c[1] for c in cats])  # name each row of points
for t in (-3.5, 3.5):  # the two threshold lines
    ax.axvline(t, color=GREY, ls="--", lw=1)  # draw a dashed line at -3.5 and +3.5
ax.set_xlabel("Robust z-score of log total expenditure")  # x-axis title
ax.set_title("Outlier screen: robust z-scores, 2019-2024 (orange = |z| > 3.5)")  # chart title
n_out = int((te.z.abs() > 3.5).sum())  # number of flagged points
fig.text(0.01, -0.02, f"{n_out} municipality-years beyond 3.5 (plus 3 negative totals, which cannot be logged). Each was checked by hand.", color=MUTED, fontsize=8)  # a note under the chart
save(fig, "03_robust_z_outliers.png")  # save it

# ---------- Chart 4: Benford's law ----------
x = auda["amount.sum"]  # audited income and expenditure amounts
x = x[x >= 10]  # use amounts of at least 10, so each has a clear first digit
d = (x / 10 ** np.floor(np.log10(x))).astype(int)  # first digit of each amount
obs = np.bincount(d, minlength=10)[1:] / len(d) * 100  # observed percentage starting with 1 to 9
exp = np.log10(1 + 1 / np.arange(1, 10)) * 100  # Benford's expected percentage for 1 to 9
mad_b = np.abs(obs - exp).mean() / 100  # Nigrini's MAD statistic (on proportions)
fig, ax = plt.subplots(figsize=(7, 3.4))  # create an empty chart
ax.bar(range(1, 10), obs, color=BLUE, width=0.6, label="Observed")  # bars for what we see in the data
ax.plot(range(1, 10), exp, color=ORANGE, lw=2, marker="o", ms=5, label="Benford expected")  # line for what Benford's law expects
ax.set_xticks(range(1, 10))  # one tick per digit
ax.set_xlabel("First digit")  # x-axis title
ax.set_ylabel("% of amounts")  # y-axis title
ax.legend(frameon=False)  # legend without a box
ax.set_title("Benford's law: first digits of audited income and expenditure amounts")  # chart title
fig.text(0.01, -0.02, f"n = {len(d):,}; MAD = {mad_b:.4f} (Nigrini: below 0.006 is close conformity).", color=MUTED, fontsize=8)  # a note under the chart
save(fig, "04_benford_first_digit.png")  # save it

# ---------- Chart 5: serial correlation of audit outcomes ----------
s = a.sort_values(K).copy()  # opinions in year order for each municipality
s["next"] = s.groupby("demarcation.code")["opinion.code"].shift(-1)  # the same municipality's opinion in the following row
s["next_year"] = s.groupby("demarcation.code")["financial_year_end.year"].shift(-1)  # and the year of that row
s = s[(s.next_year == s["financial_year_end.year"] + 1) & s["financial_year_end.year"].between(2019, 2023)]  # keep only true next-year pairs, 2019-2023
nxt = s.next.isin(NON_RELIABLE).groupby(s["opinion.code"]).mean().reindex(ORDER) * 100  # percentage non-reliable next year, per opinion this year
colours = [ORANGE if o in NON_RELIABLE else BLUE for o in nxt.index]  # orange for non-reliable opinions this year, blue for reliable
fig, ax = plt.subplots(figsize=(7, 3.4))  # create an empty chart
opinion_bars(ax, nxt, colours, "{:.0f}%")  # draw the bars with percentage labels
ax.set_title("Share with a non-reliable opinion next year, by this year's opinion (2019-2023)")  # chart title
fig.text(0.01, -0.02, "Strong serial correlation (phi = 0.64): poor outcomes tend to repeat, so audit history is the strongest predictor.", color=MUTED, fontsize=8)  # a note under the chart
save(fig, "05_serial_correlation_next_year.png")  # save it
