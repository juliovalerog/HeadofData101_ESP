-- Vista de soporte de decisiones lista para BI.
-- Combina el precio real, las predicciones expected-price y las probabilidades top-price.

CREATE TABLE IF NOT EXISTS `albertheadofdata101.autoscout_audi_a3_germany.fact_expected_price_predictions` (
  listing_id INT64,
  expected_price_eur FLOAT64
);

CREATE TABLE IF NOT EXISTS `albertheadofdata101.autoscout_audi_a3_germany.fact_top_price_predictions` (
  listing_id INT64,
  model_name STRING,
  predicted_proba FLOAT64,
  predicted_label BOOL,
  threshold_used FLOAT64
);

CREATE OR REPLACE VIEW `albertheadofdata101.autoscout_audi_a3_germany.vw_bi_dashboard` AS
WITH top_price_selected AS (
  SELECT
    listing_id,
    predicted_proba,
    CAST(predicted_label AS INT64) AS predicted_label_int
  FROM (
    SELECT
      listing_id,
      model_name,
      predicted_proba,
      predicted_label,
      ROW_NUMBER() OVER (
        PARTITION BY listing_id
        ORDER BY
          CASE WHEN model_name = 'random_forest' THEN 0 ELSE 1 END,
          predicted_proba DESC
      ) AS rn
    FROM `albertheadofdata101.autoscout_audi_a3_germany.fact_top_price_predictions`
  )
  WHERE rn = 1
)
SELECT
  fl.listing_id,
  dm.brand,
  dm.make,
  dm.model,
  df.fuel_type,
  dc.listing_country,
  dpl.price_label,
  fl.price_eur AS actual_price_eur,
  fl.mileage_km,
  fl.power_hp,
  fl.registration_date,
  fl.registration_year,
  fl.registration_month,
  fl.age_years,
  fl.page,
  fl.price_outlier_iqr,
  fl.mileage_outlier_iqr,
  fl.power_outlier_iqr,
  fl.logical_issue,
  epp.expected_price_eur,
  SAFE_SUBTRACT(fl.price_eur, epp.expected_price_eur) AS expected_price_gap_eur,
  tps.predicted_proba AS top_price_probability,
  tps.predicted_label_int AS predicted_top_price,
  CASE
    WHEN epp.expected_price_eur IS NULL OR tps.predicted_proba IS NULL THEN 'review_missing_model_outputs'
    WHEN fl.price_eur < epp.expected_price_eur AND tps.predicted_proba >= 0.5 THEN 'high_priority_review'
    WHEN fl.price_eur < epp.expected_price_eur THEN 'price_opportunity'
    WHEN tps.predicted_proba >= 0.5 THEN 'top_price_signal'
    ELSE 'standard_review'
  END AS decision_flag
FROM `albertheadofdata101.autoscout_audi_a3_germany.fact_listings` fl
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_model` dm
  ON dm.model_id = fl.model_id
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_fuel` df
  ON df.fuel_id = fl.fuel_id
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_country` dc
  ON dc.country_id = fl.country_id
JOIN `albertheadofdata101.autoscout_audi_a3_germany.dim_price_label` dpl
  ON dpl.price_label_id = fl.price_label_id
LEFT JOIN `albertheadofdata101.autoscout_audi_a3_germany.fact_expected_price_predictions` epp
  ON epp.listing_id = fl.listing_id
LEFT JOIN top_price_selected tps
  ON tps.listing_id = fl.listing_id;
