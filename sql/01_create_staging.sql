-- Staging contract for cleaned listings.
-- Load cleaned CSV data into this table before running dimension/fact builds.
CREATE TABLE IF NOT EXISTS `albertheadofdata101.autoscout_audi_a3_germany.stg_listings_clean` (
  make STRING,
  model STRING,
  fuel_type STRING,
  listing_country STRING,
  price_label STRING,
  price_eur FLOAT64,
  mileage_km INT64,
  registration_year INT64,
  registration_month INT64,
  age_years FLOAT64,
  power_hp FLOAT64
);
