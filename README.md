# HeadOfData101 (Audi A3 Germany)

Repository for the Head of Data 101 course (Albert School). This trimmed repo keeps only the Audi A3 Germany pipeline: scrape AutoScout24, clean data, build a BigQuery star schema, and run regression/classification models.

## Project flow

1. Scraping: collect raw Audi A3 Germany listings from AutoScout24.
2. Preprocessing: clean and normalize the raw data.
3. BigQuery tables: create the schema and load data.
4. Regression: fit models, label bargains, and publish results.
5. Classification: compare classifiers and report model performance.

## Repository structure

```
.
|- data/
|  |- raw/                # Raw scraping output (timestamped CSV)
|  |- processed/          # Cleaned listings (timestamped CSV)
|  `- samplefiles/
|     `- sampledata/      # Sample CSVs for class exercises
|- notebooks/
|  |- full_notebooks/     # Full versions (deeper explanations and extras)
|  |  |- 01 scrapping_audi_a3_germany.ipynb
|  |  |- 02 preprocessing_audi_a3_germany.ipynb
|  |  |- 03 sqlqueries_audi_a3_germany.ipynb
|  |  |- 04 regression_audi_a3_germany.ipynb
|  |  `- 05 classification_audi_a3_germany.ipynb
|  `- class_notebooks/    # Class versions (beginner-friendly, simplified)
|     |- 01 scrapping_audi_a3_germany_class.ipynb
|     |- 02 preprocessing_audi_a3_germany_class.ipynb
|     |- 03 sqlqueries_audi_a3_germany_class.ipynb
|     |- 04 regression_audi_a3_germany_class.ipynb
|     `- 05 classification_audi_a3_germany_class.ipynb
|- docs/
|  `- star_schema.png     # Star schema diagram
|- sql/                   # BigQuery DDL/DML scripts
|  |- 00_create_all_tables.sql
|  |- 01_create_tables.sql
|  |- 02_build_dimensions.sql
|  `- 03_build_fact.sql
`- README.md
```

## Requirements

- Python 3.x
- Jupyter (or VS Code with notebook support)
- Core libraries: `requests`, `beautifulsoup4`, `pandas`, `numpy`, `matplotlib`
- Modeling libraries: `scikit-learn`, `statsmodels`
- BigQuery libraries: `google-cloud-bigquery`, `db-dtypes`

## How to run

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install requests beautifulsoup4 pandas numpy matplotlib scikit-learn statsmodels
.\.venv\Scripts\python.exe -m pip install "google-cloud-bigquery[pandas]" db-dtypes pyarrow
```

3. Open the notebooks (in order):

Class notebooks for in-class use (recommended):

- `notebooks/class_notebooks/01 scrapping_audi_a3_germany_class.ipynb`
- `notebooks/class_notebooks/02 preprocessing_audi_a3_germany_class.ipynb`
- `notebooks/class_notebooks/03 sqlqueries_audi_a3_germany_class.ipynb`
- `notebooks/class_notebooks/04 regression_audi_a3_germany_class.ipynb`
- `notebooks/class_notebooks/05 classification_audi_a3_germany_class.ipynb`

Full notebooks for deeper self-study:

- `notebooks/full_notebooks/01 scrapping_audi_a3_germany.ipynb`
- `notebooks/full_notebooks/02 preprocessing_audi_a3_germany.ipynb`
- `notebooks/full_notebooks/03 sqlqueries_audi_a3_germany.ipynb`
- `notebooks/full_notebooks/04 regression_audi_a3_germany.ipynb`
- `notebooks/full_notebooks/05 classification_audi_a3_germany.ipynb`

## Data

- **Raw**: `data/raw/autoscout24_listings_audi_a3_germany_*.csv`
- **Processed**: `data/processed/autoscout24_listings_processed_audi_a3_germany_*.csv`
- **Sample**: `data/samplefiles/sampledata/*.csv`

Expected raw columns (from the preprocessing notebook):
`make`, `model`, `mileage`, `price`, `registration`, `fuel`, `country`, `brand`, `page`.

## Scraping good practices

- Limit pages and brands during tests.
- Respect delays between requests.
- Review the website terms of use and local legal requirements.

## BigQuery analytical database

Project: albertheadofdata101
Dataset: autoscout

This project uses Google BigQuery as an analytical data warehouse following a star schema design.
Tables include dimensions (`dim_model`, `dim_fuel`, `dim_country`), fact listings (`fact_listings`), and model output (`fact_bargains`).

### Build order

1. Upload the cleaned CSV as `stg_listings_clean`
2. Execute `sql/00_create_all_tables.sql` (consolidated DDL for all tables)
3. Execute `sql/02_build_dimensions.sql`
4. Execute `sql/03_build_fact.sql`
5. Run the regression notebook to create `fact_bargains`

### Design principles

- Star schema with one fact table and three dimensions
- Primary and foreign keys are enforced by design, not by engine constraints
- Tables are rebuilt using `CREATE OR REPLACE`
- SQL logic is versioned in GitHub for reproducibility and auditability
