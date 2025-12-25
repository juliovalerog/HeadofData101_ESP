CREATE OR REPLACE TABLE `albertheadofdata101.autoscout.fact_listings` AS
SELECT
  ROW_NUMBER() OVER (
    ORDER BY model_id,
             fuel_id,
             country_id,
             price_eur,
             mileage_km,
             registration_year,
             registration_month,
             age_years
  ) AS listing_id,
  dm.model_id,
  df.fuel_id,
  dc.country_id,
  stg.price_eur,
  stg.mileage_km,
  stg.registration_year,
  stg.registration_month,
  stg.age_years
FROM `albertheadofdata101.autoscout.stg_listings_clean` stg
JOIN `albertheadofdata101.autoscout.dim_model` dm
  ON dm.make  = TRIM(stg.make)
 AND dm.model = TRIM(stg.model)
JOIN `albertheadofdata101.autoscout.dim_fuel` df
  ON df.fuel_type = TRIM(stg.fuel_type)
JOIN `albertheadofdata101.autoscout.dim_country` dc
  ON dc.listing_country = TRIM(stg.listing_country);
