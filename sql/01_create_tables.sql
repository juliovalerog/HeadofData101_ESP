-- =========================
-- DIMENSION: MODEL
-- =========================
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout.dim_model` (
  model_id INT64 NOT NULL,
  make STRING NOT NULL,
  model STRING NOT NULL
);

-- =========================
-- DIMENSION: FUEL
-- =========================
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout.dim_fuel` (
  fuel_id INT64 NOT NULL,
  fuel_type STRING NOT NULL
);

-- =========================
-- DIMENSION: COUNTRY
-- =========================
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout.dim_country` (
  country_id INT64 NOT NULL,
  listing_country STRING NOT NULL
);

-- =========================
-- FACT TABLE: LISTINGS
-- =========================
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout.fact_listings` (
  listing_id INT64 NOT NULL,
  model_id INT64 NOT NULL,
  fuel_id INT64 NOT NULL,
  country_id INT64 NOT NULL,
  price_eur FLOAT64,
  mileage_km INT64,
  registration_year INT64,
  registration_month INT64,
  age_years FLOAT64
);
