# Project Brief

## Context

Head Of Data 101 simulates a professional delivery in which the team acts as the data product unit of a retail / consumer bank.
The business question is whether specific used vehicles are attractive acquisition opportunities for a resale and financing portfolio.

## Why This Project Exists

The goal is not syntax practice.
The goal is to practice complete product thinking across the end-to-end flow:

- acquisition of operational data
- preprocessing and data quality
- analytical warehouse design
- SQL-based dataset definition
- ML modeling and interpretation
- BI-ready communication for decisions

## Business Case

The bank wants a repeatable process to evaluate listings, prioritize opportunities, and support pricing decisions.
The baseline repo provides a working starting point that students must upgrade into a stronger analytical product.

## Scope In This Baseline

In-scope baseline capabilities:

- scrape Audi A3 Germany listings
- clean and standardize listing features
- load warehouse entities in BigQuery
- create regression and classification datasets in SQL
- estimate expected price via regression
- estimate top-price probability via classification
- publish BI-facing outputs for decision support

Out-of-scope in this baseline phase:

- final executive BI dashboard build
- production-grade orchestration
- advanced MLOps packaging

## Expected Analytical Outcome

At the end of the workflow, the team should be able to explain for each listing:

- observed market price
- expected price from the regression model
- price gap versus expectation
- probability of being top-price
- action-oriented interpretation for portfolio screening

## Professional Framing

This repo is a baseline contract, not a final answer.
Students are expected to improve model quality, BI communication, and written decisions while preserving a clear and teachable structure.
