# Run Order

This is the recommended execution sequence for the official teaching baseline.

## 0. Setup

1. Create a Python environment.
2. Install dependencies from `requirements_min.in`.
3. Review defaults in `config/project_config.yaml`.
4. Validate Google Cloud credentials and BigQuery access for the mandatory warehouse-backed path.

BigQuery access is required for the mandatory pipeline from Notebook 03 onward. It is not required for the optional Session 06 and Session 07 challenge labs, which run from processed CSV files.

## 1. Mandatory Notebook Path

Run the single final notebook set in this order:

1. `notebooks/01_scraping_audi_a3_germany.ipynb`
2. `notebooks/02_preprocessing_audi_a3_germany.ipynb`
3. `notebooks/03_sqlqueries_audi_a3_germany.ipynb`
4. `notebooks/04_regression_audi_a3_germany.ipynb`
5. `notebooks/05_classification_audi_a3_germany.ipynb`

Dependency notes:

- Notebook 01 produces the raw scraped CSV.
- Notebook 02 consumes the raw CSV and produces the processed CSV.
- Notebook 03 validates the BigQuery warehouse tables, views, and SQL assets.
- Notebook 04 consumes `vw_regression_dataset` and writes `fact_expected_price_predictions`.
- Notebook 05 consumes `vw_classification_dataset` and writes `fact_top_price_predictions`.

## 2. Mandatory SQL Execution Order

Create staging first, load the processed CSV from `data/processed` into `stg_listings_clean`, then build the downstream objects.

1. `sql/00_create_dataset.sql`
2. `sql/01_create_staging.sql`
3. Load the processed CSV into `stg_listings_clean`
4. `sql/02_build_dimensions.sql`
5. `sql/03_build_fact.sql`
6. `sql/04_vw_regression_dataset.sql`
7. `sql/05_vw_classification_dataset.sql`
8. `sql/06_vw_bi_dashboard.sql`

`sql/06_vw_bi_dashboard.sql` also creates empty prediction tables if they do not already exist, so the BI view can be defined before model notebooks replace the prediction data.

## 3. Optional Classroom Labs

These notebooks support classroom discussion but are outside the mandatory end-to-end run path:

- `notebooks/01b_raw_data_eda_before_preprocessing_audi_a3_germany.ipynb`:
  Session 03 raw-data EDA support. It consumes a raw scrape and produces exploratory displays only.

- `notebooks/04b_regression_challenge_lab_audi_a3_germany.ipynb`:
  Session 06 regression challenge lab. It consumes processed CSV files and does not require BigQuery.

- `notebooks/05b_classification_challenge_lab_audi_a3_germany.ipynb`:
  Session 07 classification challenge lab. It consumes processed CSV files and does not write warehouse tables.

These labs do not replace the production baseline notebooks.

## 4. Optional Streamlit Dashboard

The Streamlit dashboard is optional but recommended as the final decision-support demo after the warehouse and prediction tables are populated.

Run it with:

```bash
streamlit run bi/streamlit_app.py
```

The default dashboard path requires BigQuery access to `vw_bi_dashboard`. It combines actual price, expected-price gap, top-price probability, and editable business assumptions for portfolio comparison. It is not an approval engine.

## 5. Optional GenAI / Gemini Functionality

Gemini is optional. Without a Gemini API key, the Streamlit dashboard still runs with deterministic/default behavior.

With `GEMINI_API_KEY` or `GOOGLE_API_KEY` set in the process environment, Gemini can support:

- committee memo generation
- the governed text-to-SQL assistant over `vw_bi_dashboard`

The GenAI assistant is bonus material and is not part of the mandatory pipeline.

## 6. Narrative Contract To Preserve

- Regression predicts `expected_price`.
- Classification predicts the external `top_price` label/probability.
- BI combines actual price, expected-price gap, and top-price outputs.
- Streamlit is the final decision-support layer.

Do not derive the classification target from regression outputs.
