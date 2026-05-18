# Head Of Data 101 Baseline Repo

This repository is the **official teaching baseline** for Head Of Data 101.
It is a simple reference implementation of the end-to-end course pipeline: intentionally readable, notebook-first, and complete enough for students to revisit the full flow after class.

It is **not production-grade** and it is not meant to hide complexity behind packages, orchestration, or helper layers. The goal is to keep the course data product coherent and easy to inspect.

## Business Case

The course roleplay is a data unit inside a retail / consumer bank evaluating used-vehicle acquisition opportunities for resale and financing portfolios.

The baseline supports a committee-style decision process:

- actual listing price comes from the scraped marketplace data
- regression estimates `expected_price`
- classification estimates the external `top_price` signal/probability
- BI combines actual price, expected-price gap, top-price probability, and business assumptions
- Streamlit provides the final decision-support demo

The model does not decide the strategy. It helps the committee compare strategies with data.

## What This Repo Covers

The mandatory baseline pipeline covers:

1. Data acquisition through scraping
2. Data preprocessing and quality checks
3. BigQuery warehouse tables
4. SQL analytical views for ML and BI
5. Regression expected-price model
6. Classification top-price model
7. BI-ready decision-support view

The Streamlit dashboard is optional for execution, but recommended as the final live demo once the warehouse and model output tables exist.

## Mandatory Run Path

Use [docs/RUN_ORDER.md](docs/RUN_ORDER.md) as the operational sequence.

Main pipeline notebooks:

1. `notebooks/01_scraping_audi_a3_germany.ipynb`
2. `notebooks/02_preprocessing_audi_a3_germany.ipynb`
3. `notebooks/03_sqlqueries_audi_a3_germany.ipynb`
4. `notebooks/04_regression_audi_a3_germany.ipynb`
5. `notebooks/05_classification_audi_a3_germany.ipynb`

These notebooks are the single final notebook set for the baseline. There is no separate class/full notebook split.

## Optional Classroom Labs

Optional notebooks are clearly separated from the mandatory pipeline:

- `notebooks/01b_raw_data_eda_before_preprocessing_audi_a3_germany.ipynb`:
  Session 03 support notebook for raw scrape inspection before preprocessing. It does not save cleaned outputs or replace Notebook 02.

- `notebooks/04b_regression_challenge_lab_audi_a3_germany.ipynb`:
  Session 06 regression lab that runs from processed CSV files without BigQuery. It does not replace Notebook 04.

- `notebooks/05b_classification_challenge_lab_audi_a3_germany.ipynb`:
  Session 07 classification lab that runs from processed CSV files without BigQuery writes. It does not replace Notebook 05.

## SQL Assets

SQL is ordered explicitly under `sql/` for classroom execution:

1. `00_create_dataset.sql`
2. `01_create_staging.sql`
3. `02_build_dimensions.sql`
4. `03_build_fact.sql`
5. `04_vw_regression_dataset.sql`
6. `05_vw_classification_dataset.sql`
7. `06_vw_bi_dashboard.sql`

The SQL folder defines the BigQuery dataset, staging table, dimensions, fact table, ML-facing views, prediction table shells, and BI dashboard view. Preserve these object names because notebooks and BI depend on them.

## Docs Folder

The `docs/` folder contains student-facing reference material:

- [RUN_ORDER.md](docs/RUN_ORDER.md): mandatory and optional execution sequence
- [DATA_CONTRACT.md](docs/DATA_CONTRACT.md): warehouse, ML, prediction, and BI contracts
- [PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md): business framing for the course case
- [genai_text_to_sql_bonus.md](docs/genai_text_to_sql_bonus.md): optional Gemini text-to-SQL bonus
- images used by course documentation

## BI / Streamlit

The Streamlit app in `bi/` is the final decision-support layer. It reads the governed BigQuery view `vw_bi_dashboard`, combines model signals with editable business assumptions, and helps compare portfolio strategies.

It is not an approval engine and it does not replace committee judgment.

See [bi/README.md](bi/README.md) for dashboard-specific setup and demo notes.

## Optional Gemini Functionality

Gemini features are optional. The Streamlit dashboard can use Gemini for committee memo generation and the GenAI SQL assistant when:

- `google-genai` is installed
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set in the process environment

Configure a key locally in PowerShell with one of:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

or:

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

`GEMINI_API_KEY` takes priority over `GOOGLE_API_KEY`. If no key is set, the dashboard keeps working with deterministic/default behavior. Gemini credentials are not read from `.streamlit/secrets.toml`, Streamlit Cloud secrets, `.env`, or committed credential files.

## Installation

Create and activate a Python environment, then install the minimal course dependencies:

```bash
pip install -r requirements_min.in
```

For BigQuery-backed notebooks and Streamlit, authenticate with Google Cloud in your local environment:

```bash
gcloud auth application-default login
```

Project defaults live in:

- `config/project_config.yaml`

## Run The Baseline

1. Review `config/project_config.yaml`.
2. Run the five mandatory notebooks in order.
3. Execute the SQL files in the order shown in [docs/RUN_ORDER.md](docs/RUN_ORDER.md), including loading the processed CSV into `stg_listings_clean`.
4. Confirm Notebook 04 writes `fact_expected_price_predictions`.
5. Confirm Notebook 05 writes `fact_top_price_predictions`.
6. Optionally run the Streamlit decision-support dashboard:

```bash
streamlit run bi/streamlit_app.py
```

## What To Expect

Expect a readable teaching baseline that preserves the full course flow and the BI-ready data contract.

Do not expect production orchestration, CI/CD, Docker, package structure, advanced MLOps, or a hardened cloud deployment. Those are valid extensions, but they are intentionally outside this baseline repository.
