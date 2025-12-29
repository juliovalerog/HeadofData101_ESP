-- dim_model (poblada)
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.dim_model` AS
WITH dedup AS (
  SELECT DISTINCT
    TRIM(make)  AS make,
    TRIM(model) AS model
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
  WHERE make IS NOT NULL AND model IS NOT NULL
)
SELECT
  ROW_NUMBER() OVER (ORDER BY make, model) AS model_id,
  make,
  model
FROM dedup;

-- dim_fuel (poblada)
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.dim_fuel` AS
WITH dedup AS (
  SELECT DISTINCT
    TRIM(fuel_type) AS fuel_type
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
  WHERE fuel_type IS NOT NULL
)
SELECT
  ROW_NUMBER() OVER (ORDER BY fuel_type) AS fuel_id,
  fuel_type
FROM dedup;

-- dim_country (poblada)
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.dim_country` AS
WITH dedup AS (
  SELECT DISTINCT
    TRIM(listing_country) AS listing_country
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
  WHERE listing_country IS NOT NULL
)
SELECT
  ROW_NUMBER() OVER (ORDER BY listing_country) AS country_id,
  listing_country
FROM dedup;

-- dim_price_label (poblada)
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.dim_price_label` AS
WITH dedup AS (
  SELECT DISTINCT
    TRIM(price_label) AS price_label
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
  WHERE price_label IS NOT NULL
)
SELECT
  ROW_NUMBER() OVER (ORDER BY price_label) AS price_label_id,
  price_label
FROM dedup;