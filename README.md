# Municipal audit analytics

Audit-risk analytics on South African municipal finance data, built as a hands-on learning project in internal audit data analytics. The data comes from National Treasury's [Municipal Money](https://municipaldata.treasury.gov.za) portal: Auditor-General opinions, irregular and wasteful expenditure, and audited financial statements for all 257 municipalities.

**Question:** which financial and reporting signals flag a municipality at risk of a poor audit outcome?

## Key finding so far

> **The gaps in this data are not random.** Municipalities with worse audit outcomes are far more likely to be missing their audited financial statements. In audit terms, the missingness is itself a risk signal.

| Audit opinion (2018–2024) | Municipality-years with no audited financial statements |
|---|---|
| Outstanding (audit not finalised) | 38.0% |
| Disclaimer | 9.9% |
| Adverse | 9.3% |
| Qualified | 6.6% |
| Unqualified with findings | 2.7% |
| Unqualified (clean) | 1.8% |

Chi-square test of missingness against audit opinion: 163.7, p < 0.001, across 1,799 municipality-years. That rules out deleting incomplete records (complete-case analysis is only unbiased when data is missing completely at random; Rubin 1976), because doing so would drop the very municipalities an auditor most needs to see.

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
2. **Fabric load** (`fabric/01_load_raw_to_lakehouse.ipynb`): done. Parquet files in `Files/raw/` become `raw_*` Delta tables, row counts reconciled to source.
3. **EDA step 1: missingness** (`src/eda_01_missingness.py`): done. Key finding: missing audited statements are strongly linked to poor audit opinions (38% of outstanding audits against 1.8% of clean audits, chi-square p < 0.001). The gaps are not random, and they are themselves a risk signal.
4. Next: cleaning decisions, feature engineering, risk model, Power BI dashboard.

## Fabric screenshots

| Workspace on trial capacity | Load notebook |
|---|---|
| ![workspace](docs/screenshots/02_fabric_workspace_trial_capacity.png) | ![notebook](docs/screenshots/03_fabric_load_notebook.png) |

Spark SQL check after the load: 81 year-by-opinion rows, matching the source exactly.

![SQL check](docs/screenshots/04_fabric_sql_check.png)
