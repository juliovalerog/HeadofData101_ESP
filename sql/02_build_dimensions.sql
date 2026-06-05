-- Construya dimensiones a partir de la puesta en escena.

CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.dim_model` AS
WITH normalized AS (
  SELECT
    NULLIF(TRIM(brand), '') AS brand,
    NULLIF(TRIM(make), '') AS make,
    NULLIF(TRIM(model), '') AS model
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
),
dedup AS (
  SELECT DISTINCT
    COALESCE(brand, make) AS brand,
    make,
    model
  FROM normalized
  WHERE make IS NOT NULL
    AND model IS NOT NULL
)
SELECT
  ROW_NUMBER() OVER (ORDER BY brand, make, model) AS model_id,
  brand,
  make,
  model
FROM dedup;

CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.dim_fuel` AS
WITH dedup AS (
  SELECT DISTINCT
    NULLIF(TRIM(fuel_type), '') AS fuel_type
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
  WHERE fuel_type IS NOT NULL
)
SELECT
  ROW_NUMBER() OVER (ORDER BY fuel_type) AS fuel_id,
  fuel_type
FROM dedup
WHERE fuel_type IS NOT NULL;

CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.dim_country` AS
WITH dedup AS (
  SELECT DISTINCT
    NULLIF(TRIM(listing_country), '') AS listing_country
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
  WHERE listing_country IS NOT NULL
)
SELECT
  ROW_NUMBER() OVER (ORDER BY listing_country) AS country_id,
  listing_country
FROM dedup
WHERE listing_country IS NOT NULL;

CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.dim_price_label` AS
WITH dedup AS (
  SELECT DISTINCT
    NULLIF(TRIM(price_label), '') AS price_label
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
  WHERE price_label IS NOT NULL
)
SELECT
  ROW_NUMBER() OVER (ORDER BY price_label) AS price_label_id,
  price_label
FROM dedup
WHERE price_label IS NOT NULL;
