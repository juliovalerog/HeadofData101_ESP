-- Classification dataset view.
-- Target variable is `top_price` derived from external price label semantics.

CREATE OR REPLACE VIEW `albertheadofdata101.autoscout_audi_a3_germany.vw_classification_dataset` AS
SELECT
  fl.listing_id,
  dm.make,
  dm.model,
  df.fuel_type,
  dc.listing_country,
  fl.price_eur AS actual_price_eur,
  fl.mileage_km,
  fl.power_hp,
  fl.registration_year,
  fl.registration_month,
  fl.age_years,
  dpl.price_label,
  CASE WHEN dpl.price_label = 'top-price' THEN 1 ELSE 0 END AS top_price
FROM `albertheadofdata101.autoscout_audi_a3_germany.fact_listings` fl
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_model` dm
  ON dm.model_id = fl.model_id
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_fuel` df
  ON df.fuel_id = fl.fuel_id
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_country` dc
  ON dc.country_id = fl.country_id
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_price_label` dpl
  ON dpl.price_label_id = fl.price_label_id;
