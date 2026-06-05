-- Vista del conjunto de datos de regresión.
-- La variable objetivo de la regresión es el precio real (`actual_price_eur`).

CREATE OR REPLACE VIEW `albertheadofdata101.autoscout_audi_a3_germany.vw_regression_dataset` AS
SELECT
  fl.listing_id,
  dm.brand,
  dm.make,
  dm.model,
  df.fuel_type,
  dc.listing_country,
  fl.mileage_km,
  fl.power_hp,
  fl.registration_date,
  fl.registration_year,
  fl.registration_month,
  fl.age_years,
  fl.price_eur AS actual_price_eur
FROM `albertheadofdata101.autoscout_audi_a3_germany.fact_listings` fl
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_model` dm
  ON dm.model_id = fl.model_id
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_fuel` df
  ON df.fuel_id = fl.fuel_id
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_country` dc
  ON dc.country_id = fl.country_id;
