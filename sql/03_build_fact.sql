-- Construya una tabla de hechos del listado.

CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.fact_listings` AS
WITH stg AS (
  SELECT
    COALESCE(NULLIF(TRIM(brand), ''), NULLIF(TRIM(make), '')) AS brand,
    NULLIF(TRIM(make), '') AS make,
    NULLIF(TRIM(model), '') AS model,
    NULLIF(TRIM(fuel_type), '') AS fuel_type,
    NULLIF(TRIM(listing_country), '') AS listing_country,
    NULLIF(TRIM(price_label), '') AS price_label,
    SAFE_CAST(price_eur AS INT64) AS price_eur,
    SAFE_CAST(ROUND(mileage_km) AS INT64) AS mileage_km,
    SAFE_CAST(power_hp AS FLOAT64) AS power_hp,
    registration_date,
    SAFE_CAST(ROUND(registration_year) AS INT64) AS registration_year,
    SAFE_CAST(ROUND(registration_month) AS INT64) AS registration_month,
    SAFE_CAST(age_years AS FLOAT64) AS age_years,
    SAFE_CAST(page AS INT64) AS page,
    COALESCE(price_outlier_iqr, FALSE) AS price_outlier_iqr,
    COALESCE(mileage_outlier_iqr, FALSE) AS mileage_outlier_iqr,
    COALESCE(power_outlier_iqr, FALSE) AS power_outlier_iqr,
    COALESCE(logical_issue, FALSE) AS logical_issue
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
),
filtered AS (
  SELECT *
  FROM stg
  WHERE make IS NOT NULL
    AND model IS NOT NULL
    AND fuel_type IS NOT NULL
    AND listing_country IS NOT NULL
    AND price_label IS NOT NULL
),
joined AS (
  SELECT
    dm.model_id,
    df.fuel_id,
    dc.country_id,
    dpl.price_label_id,
    filtered.price_eur,
    filtered.mileage_km,
    filtered.power_hp,
    filtered.registration_date,
    filtered.registration_year,
    filtered.registration_month,
    filtered.age_years,
    filtered.page,
    filtered.price_outlier_iqr,
    filtered.mileage_outlier_iqr,
    filtered.power_outlier_iqr,
    filtered.logical_issue
  FROM filtered
  JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_model` dm
    ON dm.brand = filtered.brand
   AND dm.make = filtered.make
   AND dm.model = filtered.model
  JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_fuel` df
    ON df.fuel_type = filtered.fuel_type
  JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_country` dc
    ON dc.listing_country = filtered.listing_country
  JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_price_label` dpl
    ON dpl.price_label = filtered.price_label
)
SELECT
  ROW_NUMBER() OVER (
    ORDER BY model_id, fuel_id, country_id, price_label_id,
             price_eur, mileage_km, power_hp, registration_date,
             registration_year, registration_month, age_years, page
  ) AS listing_id,
  model_id,
  fuel_id,
  country_id,
  price_label_id,
  price_eur,
  mileage_km,
  power_hp,
  registration_date,
  registration_year,
  registration_month,
  age_years,
  page,
  price_outlier_iqr,
  mileage_outlier_iqr,
  power_outlier_iqr,
  logical_issue
FROM joined;
