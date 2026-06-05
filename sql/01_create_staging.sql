-- Contrato de preparación para el CSV procesado producido por Notebook 02.
-- Ejecute esto antes de cargar el CSV procesado, luego cargue los datos antes de que se generen dimensiones/hechos.
CREATE OR REPLACE TABLE `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean` (
  make STRING,
  model STRING,
  brand STRING,
  price_eur INT64,
  price_label STRING,
  mileage_km FLOAT64,
  power_hp FLOAT64,
  registration_date DATE,
  registration_year FLOAT64,
  registration_month FLOAT64,
  fuel_type STRING,
  listing_country STRING,
  page INT64,
  price_outlier_iqr BOOL,
  mileage_outlier_iqr BOOL,
  power_outlier_iqr BOOL,
  logical_issue BOOL,
  age_years FLOAT64
);
