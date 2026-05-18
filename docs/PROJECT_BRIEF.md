# Project Brief

## Context

Head Of Data 101 simulates a professional delivery in which the team acts as the data product unit of a retail / consumer bank.
The business question is whether specific used vehicles are attractive acquisition opportunities for a resale and financing portfolio.

Students already have the course context. This brief is a reference for remembering the case, the baseline scope, and the expected decision-support storyline.

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

The bank wants a repeatable process to evaluate listings, prioritize opportunities, and support pricing discussions.

The baseline repo provides a simple but coherent reference implementation of that data product. It is designed for learning, review, and extension after the course, not as a production-grade platform.

## Scope In This Baseline

In-scope baseline capabilities:

- scrape Audi A3 Germany listings
- clean and standardize listing features
- load warehouse entities in BigQuery
- create regression and classification datasets in SQL
- estimate expected price via regression
- estimate top-price probability via classification
- publish BI-facing outputs for decision support
- optionally demo the Streamlit committee dashboard

Out-of-scope for this teaching baseline:

- production-grade orchestration
- CI/CD, Docker, or deployment automation
- advanced MLOps packaging
- hidden helper packages or complex abstractions
- automated approval, credit, compliance, or risk decisions

## Expected Analytical Outcome

At the end of the workflow, the team should be able to explain for each listing:

- observed market price
- expected price from the regression model
- price gap versus expectation
- probability of being top-price
- action-oriented interpretation for portfolio screening

## Professional Framing

This repo is the official baseline contract for the taught course flow.
It is intentionally simple, notebook-first, and readable so students can inspect how the full data product fits together.

Students can improve model quality, BI communication, and written decisions while preserving the baseline structure and the `expected_price` + `top_price` narrative.
