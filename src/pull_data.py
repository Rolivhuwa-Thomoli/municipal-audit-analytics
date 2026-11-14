"""Pull Municipal Money (National Treasury) data into raw Parquet files.

Small tables are downloaded row by row ("facts"). Large tables have millions of
monthly rows, so for those we ask the website to add the numbers up for us
("aggregate"): one row per municipality, per year, per line item, per amount type.

Run: python src/pull_data.py data/raw            (pulls every table)
     python src/pull_data.py data/raw uifwexp    (pulls only the tables you name)
"""
import sys  # lets us read the words typed after the script name on the command line
import time  # lets us pause between retries
import pathlib  # makes working with folder and file paths easy
import requests  # downloads data from websites
import pandas as pd  # the table library we use to hold and save the data

B = "https://municipaldata.treasury.gov.za/api/cubes"  # the web address all the data lives under
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")  # the folder to save into: the first word typed after the script name, or data/raw if none was given
OUT.mkdir(parents=True, exist_ok=True)  # create that folder (and any parent folders) if it doesn't exist yet
S = requests.Session()  # one reusable connection to the website, faster than reconnecting every time


def get(url, **params):
    """Download one page of data from the website, retrying if the connection drops."""
    for i in range(8):  # try up to 8 times
        try:  # attempt the download; if anything goes wrong, jump to "except"
            r = S.get(url, params=params, timeout=120)  # ask the website for the data, giving up after 120 seconds
            r.raise_for_status()  # turn an error reply from the website (like "page not found") into a Python error
            return r.json()  # success: convert the reply from JSON text into Python lists and dictionaries and hand it back
        except Exception as e:  # something went wrong (the connection resets now and then)
            print(f"  retry {i+1}: {e}", flush=True)  # say which attempt failed and why
            time.sleep(min(2 ** i, 60))  # wait 1, 2, 4, 8... seconds (at most 60) before trying again
    raise RuntimeError(url)  # all 8 attempts failed: stop the script and show which address failed


def facts(cube, pagesize=10000):
    """Download every row of a small table, one page of 10,000 rows at a time."""
    rows, page = [], 1  # start with an empty list of rows, on page 1
    while True:  # keep going until we break out below
        d = get(f"{B}/{cube}/facts", pagesize=pagesize, page=page)  # download one page of rows
        rows += d["data"]  # add this page's rows to our list
        if len(d["data"]) < pagesize:  # a page that isn't full means it was the last page
            break  # so stop looping
        page += 1  # otherwise move on to the next page
    return pd.DataFrame(rows)  # turn the list of rows into a table


def aggregate(cube, drill, measures, cut, pagesize=20000):
    """Ask the website to add up a large table for us, then download the totals page by page."""
    rows, page = [], 1  # start with an empty list of rows, on page 1
    while True:  # keep going until we break out below
        d = get(f"{B}/{cube}/aggregate",  # the "add it up for me" address for this table
                drilldown="|".join(drill),  # the columns to group by, joined with | as the website expects
                cut=cut,  # a filter, e.g. only annual figures
                aggregates="|".join(measures),  # the numbers to add up, joined with |
                pagesize=pagesize,  # how many rows per page
                page=page,  # which page we want
                order="demarcation.code:asc")  # sort by municipality code so pages don't overlap
        rows += d["cells"]  # add this page's rows to our list
        print(f"  {cube} page {page}: {len(rows)} rows of {d.get('total_cell_count')}", flush=True)  # show progress
        if len(d["cells"]) < pagesize:  # a page that isn't full means it was the last page
            break  # so stop looping
        page += 1  # otherwise move on to the next page
    return pd.DataFrame(rows)  # turn the list of rows into a table


ANNUAL = 'period_length.length:"year"'  # filter: keep only full-year figures, not monthly or quarterly ones
ITEM = ["demarcation.code", "financial_year_end.year", "item.code", "item.label", "amount_type.code"]  # group by municipality, year, line item and amount type

DEBTOR_MEASURES = ["total_amount", "l30_amount", "l60_amount", "l90_amount", "l120_amount",  # debtor amounts: total, and by age (under 30 days, 31-60 days...)
                   "l150_amount", "l180_amount", "g1_amount", "bad_amount", "badi_amount"]  # ...up to over a year, plus bad debts

jobs = {  # the list of tables to pull, and how to pull each one
    "municipalities": lambda: facts("municipalities"),  # small table: names, provinces and categories of municipalities
    "audit_opinions": lambda: facts("audit_opinions"),  # small table: the Auditor-General's opinion for each municipality and year
    "uifwexp": lambda: facts("uifwexp"),  # small table: unauthorised, irregular, fruitless and wasteful expenditure
    "incexp_v2": lambda: aggregate("incexp_v2", ITEM, ["amount.sum"], ANNUAL),  # large table: income and expenditure, added up per year
    "financial_position_v2": lambda: aggregate("financial_position_v2", ITEM, ["amount.sum"], ANNUAL),  # large table: balance sheet, added up per year
    "aged_debtor_v2": lambda: aggregate("aged_debtor_v2", ITEM, [m + ".sum" for m in DEBTOR_MEASURES], ANNUAL),  # large table: money owed to municipalities, by age
}

for name in (sys.argv[2:] or jobs):  # go through the tables named on the command line, or all of them if none were named
    if (OUT / f"{name}.parquet").exists():  # if we already saved this table earlier...
        print("skip", name)  # ...say so...
        continue  # ...and move on without downloading it again
    print("pulling", name, flush=True)  # say which table we're downloading
    df = jobs[name]()  # run the download recipe for this table
    df.to_parquet(OUT / f"{name}.parquet", index=False)  # save it as a Parquet file (a compact, fast table format)
    print(f"saved {name}: {df.shape}", flush=True)  # report how many rows and columns were saved
