-- =========================
-- DIMENSION: MODEL
-- =========================
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout.dim_model` AS
SELECT
  ROW_NUMBER() OVER (ORDER BY make, model) AS model_id,
  make,
  model
FROM (
  SELECT DISTINCT
    TRIM(make)  AS make,
    TRIM(model) AS model
  FROM `albertheadofdata101.autoscout.stg_listings_clean`
  WHERE make IS NOT NULL
    AND model IS NOT NULL
);

-- =========================
-- DIMENSION: FUEL
-- =========================
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout.dim_fuel` AS
SELECT
  ROW_NUMBER() OVER (ORDER BY fuel_type) AS fuel_id,
  fuel_type
FROM (
  SELECT DISTINCT
    TRIM(fuel_type) AS fuel_type
  FROM `albertheadofdata101.autoscout.stg_listings_clean`
  WHERE fuel_type IS NOT NULL
);

-- =========================
-- DIMENSION: COUNTRY
-- =========================
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout.dim_country` AS
SELECT
  ROW_NUMBER() OVER (ORDER BY listing_country) AS country_id,
  listing_country
FROM (
  SELECT DISTINCT
    TRIM(listing_country) AS listing_country
  FROM `albertheadofdata101.autoscout.stg_listings_clean`
  WHERE listing_country IS NOT NULL
);
