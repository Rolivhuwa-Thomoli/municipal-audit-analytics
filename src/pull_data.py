"""Pull Municipal Money (National Treasury) cubes into raw Parquet files.

Small cubes are pulled fact by fact. Large cubes (millions of monthly/quarterly
rows) are pulled through the aggregate endpoint, filtered to annual figures,
one row per municipality x year x line item x amount type.
"""
import sys, time, pathlib, requests, pandas as pd

B = "https://municipaldata.treasury.gov.za/api/cubes"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
OUT.mkdir(parents=True, exist_ok=True)
S = requests.Session()

def get(url, **params):
    for i in range(8):  # the connection resets intermittently, so retry with backoff
        try:
            r = S.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry {i+1}: {e}", flush=True)
            time.sleep(min(2 ** i, 60))
    raise RuntimeError(url)

def facts(cube, pagesize=10000):
    rows, page = [], 1
    while True:
        d = get(f"{B}/{cube}/facts", pagesize=pagesize, page=page)
        rows += d["data"]
        if len(d["data"]) < pagesize: break
        page += 1
    return pd.DataFrame(rows)

def aggregate(cube, drill, measures, cut, pagesize=20000):
    rows, page = [], 1
    while True:
        d = get(f"{B}/{cube}/aggregate", drilldown="|".join(drill), cut=cut,
                aggregates="|".join(measures), pagesize=pagesize, page=page,
                order="demarcation.code:asc")
        rows += d["cells"]
        print(f"  {cube} page {page}: {len(rows)} rows of {d.get('total_cell_count')}", flush=True)
        if len(d["cells"]) < pagesize: break
        page += 1
    return pd.DataFrame(rows)

ANNUAL = 'period_length.length:"year"'
ITEM = ["demarcation.code", "financial_year_end.year", "item.code", "item.label", "amount_type.code"]

jobs = {
    "municipalities": lambda: facts("municipalities"),
    "audit_opinions": lambda: facts("audit_opinions"),
    "uifwexp": lambda: facts("uifwexp"),
    "incexp_v2": lambda: aggregate("incexp_v2", ITEM, ["amount.sum"], ANNUAL),
    "financial_position_v2": lambda: aggregate("financial_position_v2", ITEM, ["amount.sum"], ANNUAL),
    "aged_debtor_v2": lambda: aggregate("aged_debtor_v2", ITEM, [m + ".sum" for m in
        ["total_amount", "l30_amount", "l60_amount", "l90_amount", "l120_amount", "l150_amount",
         "l180_amount", "g1_amount", "bad_amount", "badi_amount"]], ANNUAL),
}
for name in (sys.argv[2:] or jobs):
    if (OUT / f"{name}.parquet").exists(): print("skip", name); continue
    print("pulling", name, flush=True)
    df = jobs[name]()
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"saved {name}: {df.shape}", flush=True)
