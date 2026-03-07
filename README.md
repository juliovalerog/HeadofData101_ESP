# Head Of Data 101 Baseline Repo

This repository is the **day-0 baseline** for a professional end-to-end data product workflow.
It is intentionally incomplete and designed for students to extend.

The course roleplay is a data unit inside a retail / consumer bank evaluating used-vehicle acquisition opportunities for resale and financing portfolios.

## What This Repo Covers

The baseline supports the full delivery chain:

1. Data acquisition (web scraping)
2. Data preprocessing and quality checks
3. Analytical warehouse modeling in BigQuery
4. SQL dataset definitions for ML and BI consumption
5. Regression to estimate expected price
6. Classification to estimate top-price probability
7. BI-ready outputs for decision support

## What Students Must Improve

Students are expected to evolve this baseline into a stronger product by improving:

- model performance and validation rigor
- BI layer and dashboard storytelling
- business insights and recommendation logic
- technical and decision documentation

This repository is **not** the finished product.

## Repository Contract

- Keep the repo simple and readable for teaching.
- Keep notebooks as the main learning surface.
- Use one single final notebook set (no class/full split in this repo).
- Preserve the narrative:
  - regression predicts `expected_price`
  - classification predicts `top_price`
  - BI combines actual price, expected-price gap, and top-price outputs

## Recommended Run Path

Use [docs/RUN_ORDER.md](docs/RUN_ORDER.md) as the operational sequence.

Current final notebook target path:

- `notebooks/01_scraping_audi_a3_germany.ipynb`
- `notebooks/02_preprocessing_audi_a3_germany.ipynb`
- `notebooks/03_sqlqueries_audi_a3_germany.ipynb`
- `notebooks/04_regression_audi_a3_germany.ipynb`
- `notebooks/05_classification_audi_a3_germany.ipynb`

## SQL Assets

SQL is organized in ordered files under `sql/`:

- `00_create_dataset.sql`
- `01_create_staging.sql`
- `02_build_dimensions.sql`
- `03_build_fact.sql`
- `04_vw_regression_dataset.sql`
- `05_vw_classification_dataset.sql`
- `06_vw_bi_dashboard.sql`

## Environment

Install minimum requirements from:

- `requirements_min.in`

Project-level defaults live in:

- `config/project_config.yaml`
