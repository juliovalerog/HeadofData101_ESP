# BI Integration Guide

## Purpose

The BI layer consumes curated warehouse outputs to support listing-level investment decisions.
It is intentionally thin in this baseline and should be extended by students.

## Required Inputs

Primary source view:

- `vw_bi_dashboard`

Expected upstream dependencies:

- `fact_listings`
- `fact_expected_price_predictions`
- `fact_top_price_predictions`
- dimensions (`dim_model`, `dim_fuel`, `dim_country`, `dim_price_label`)

## Expected Semantic Layer

Expose business-friendly fields grouped as:

- listing profile: brand, make, model, fuel, country, age, mileage, power
- observed economics: actual listing price
- expected economics: expected price, expected-price gap
- model signals: top-price probability, predicted top-price class
- decision helper: shortlist flags / priority buckets

## Dashboard Questions To Answer

1. Which listings are priced below expected value?
2. Which listings show high top-price probability?
3. Where do expected-price gap and top-price probability agree or disagree?
4. Which vehicle segments (model-year, mileage, fuel) concentrate opportunities?
5. Which candidates should be prioritized for acquisition review?

## Delivery Note

This repo does not include the final dashboard build in this baseline.
The contract above is the handoff target for BI implementation in later phases.
