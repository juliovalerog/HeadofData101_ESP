CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.fact_listings` AS
WITH stg AS (
  SELECT
    TRIM(make) AS make,
    TRIM(model) AS model,
    TRIM(fuel_type) AS fuel_type,
    TRIM(listing_country) AS listing_country,
    TRIM(price_label) AS price_label,
    price_eur,
    mileage_km,
    registration_year,
    registration_month,
    age_years,
    power_hp
  FROM `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean`
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
    stg.price_eur,
    stg.mileage_km,
    stg.registration_year,
    stg.registration_month,
    stg.age_years,
    stg.power_hp
  FROM stg
  JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_model` dm
    ON dm.make = stg.make AND dm.model = stg.model
  JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_fuel` df
    ON df.fuel_type = stg.fuel_type
  JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_country` dc
    ON dc.listing_country = stg.listing_country
  JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_price_label` dpl
    ON dpl.price_label = stg.price_label
)
SELECT
  ROW_NUMBER() OVER (
    ORDER BY model_id, fuel_id, country_id, price_label_id,
             price_eur, mileage_km, power_hp,
             registration_year, registration_month, age_years
  ) AS listing_id,
  model_id,
  fuel_id,
  country_id,
  price_label_id,
  price_eur,
  mileage_km,
  power_hp,
  registration_year,
  registration_month,
  age_years
FROM joined;
