# Municipal audit analytics

Audit-risk analytics on South African municipal finance data, built as a hands-on learning project in internal audit data analytics. The data comes from National Treasury's [Municipal Money](https://municipaldata.treasury.gov.za) portal: Auditor-General opinions, irregular and wasteful expenditure, and audited financial statements for all 257 municipalities.

**Question:** which financial and reporting signals flag a municipality at risk of a poor audit outcome?

## Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python | Simple to read and review, widely used in audit analytics |
| Storage and SQL | Microsoft Fabric Lakehouse | Learn a modern lakehouse: Delta files plus a SQL endpoint |
| Dashboard | Power BI | Standard reporting tool for audit teams, connects natively to Fabric |
| Version control | GitHub | Every change is traceable and reversible, an audit trail for the analysis itself |

## Data

| Table | Rows | Content |
|---|---|---|
| audit_opinions | 3,688 | Auditor-General opinion per municipality-year, 2011–2024 |
| uifwexp | 10,314 | Unauthorised, irregular, fruitless and wasteful expenditure, 2012–2024 |
| incexp_v2 | 338,484 | Income and expenditure, annual totals per line item (mSCOA, 2018+) |
| financial_position_v2 | 321,481 | Balance sheet, annual totals per line item (mSCOA, 2018+) |
| aged_debtor_v2 | 45,374 | Debtors by age bucket (audited from 2020) |
| municipalities | 292 | Name, province, category |

Reproduce with `pip install -r requirements.txt && python src/pull_data.py data/raw`. The API resets connections now and then, so the script retries with backoff.

## Progress

1. **Data pull** (`src/pull_data.py`): done.
2. **Fabric load** (`fabric/01_load_raw_to_lakehouse.ipynb`): Parquet files in `Files/raw/` become `raw_*` Delta tables.
3. **EDA step 1: missingness** (`src/eda_01_missingness.py`): done. Key finding: missing audited statements are strongly linked to poor audit opinions (38% of outstanding audits against 1.8% of clean audits, chi-square p < 0.001). The gaps are not random, and they are themselves a risk signal.
4. Next: cleaning decisions, feature engineering, risk model, Power BI dashboard.

## Fabric screenshots

| Workspace on trial capacity | Load notebook |
|---|---|
| ![workspace](docs/screenshots/02_fabric_workspace_trial_capacity.png) | ![notebook](docs/screenshots/03_fabric_load_notebook.png) |
