# Data Contract

## Warehouse Scope

Default warehouse target:

- project: `albertheadofdata101`
- dataset: `autoscout_audi_a3_germany`

These defaults can be replaced using `config/project_config.yaml` values.

## Main Datasets And Grain

### `stg_listings_clean`

- Grain: one row per scraped listing record
- Role: cleaned staging input for dimensional/fact modeling
- Key columns (minimum expected):
  - `make`, `model`
  - `fuel_type`
  - `listing_country`
  - `price_label`
  - `price_eur`
  - `mileage_km`
  - `registration_year`
  - `registration_month`
  - `age_years`
  - `power_hp`

### Dimensions

- `dim_model`
  - Grain: one row per (`make`, `model`)
  - Key: `model_id`
- `dim_fuel`
  - Grain: one row per `fuel_type`
  - Key: `fuel_id`
- `dim_country`
  - Grain: one row per `listing_country`
  - Key: `country_id`
- `dim_price_label`
  - Grain: one row per `price_label`
  - Key: `price_label_id`

### `fact_listings`

- Grain: one row per listing
- Key: `listing_id`
- Foreign keys:
  - `model_id` -> `dim_model`
  - `fuel_id` -> `dim_fuel`
  - `country_id` -> `dim_country`
  - `price_label_id` -> `dim_price_label`
- Measures/attributes:
  - `price_eur`, `mileage_km`, `power_hp`
  - `registration_year`, `registration_month`, `age_years`

## ML-Facing Views

### `vw_regression_dataset`

- Grain: one row per listing
- Purpose: regression input to estimate expected price
- Core fields:
  - `listing_id`, `make`, `model`, `listing_country`
  - `actual_price_eur`, `mileage_km`, `age_years`, `power_hp`

### `vw_classification_dataset`

- Grain: one row per listing
- Purpose: classification input to estimate `top_price`
- Target field:
  - `top_price` (1 when `price_label = 'top-price'`, else 0)

## Prediction Tables (Notebook Outputs)

### `fact_expected_price_predictions`

- Grain: one row per listing
- Key columns:
  - `listing_id`, `expected_price_eur`

### `fact_top_price_predictions`

- Grain: multiple rows per listing (one row per model)
- Key columns:
  - `listing_id`, `model_name`, `predicted_proba`, `predicted_label`, `threshold_used`

## BI-Facing Output

### `vw_bi_dashboard`

- Grain: one row per listing
- Intended joins:
  - listing facts and dimensions
  - `fact_expected_price_predictions`
  - `fact_top_price_predictions` (single selected model row per listing)
- Required BI fields:
  - actual price
  - expected price
  - expected-price gap (`actual - expected`)
  - top-price probability and label
  - decision-support flag

## Narrative Rules

This contract enforces the target storyline:

- regression -> expected price
- classification -> external top_price label
- BI -> decision support from both outputs

Do not derive the classification target from regression outputs.
