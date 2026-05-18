# GenAI Text-to-SQL Bonus

This document describes optional bonus material. It is not part of the mandatory scraping, preprocessing, BigQuery, SQL, regression, classification, or BI contract run path.

## Purpose

This bonus adds a professional text-to-SQL MVP to the Streamlit BI dashboard. It lets students ask natural-language business questions and see how a governed BI data product can be queried through a GenAI layer without making the LLM the system of record.

The classroom framing is:

> The LLM writes candidate SQL. The application owns validation. BigQuery owns the data. The user owns the decision.

## Architecture

Streamlit UI -> Gemini SQL generator -> SQL validator -> BigQuery dry-run -> BigQuery execution -> result summarizer

The user enters a business question in Streamlit. Gemini receives a semantic prompt describing the governed BI view, allowed fields, field meanings, and strict SQL rules. Gemini returns structured JSON containing SQL, business intent, chart preference, confidence, and limitations.

Gemini is optional. The app uses the `google-genai` package with model `gemini-2.5-flash` when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set in the process environment. It does not read Gemini credentials from `.streamlit/secrets.toml`, Streamlit Cloud secrets, `.env`, or committed credential files.

The app then validates the SQL deterministically before BigQuery sees it. If the SQL is safe, the app runs a BigQuery dry-run to estimate scanned bytes. Only queries below the classroom scan threshold are executed. Results are shown as a dataframe, optional Plotly chart, and concise executive interpretation.

## Why This Is Text-to-SQL

This is not a simple predefined query router. With a Gemini API key configured through an environment variable, the assistant translates open-ended natural-language questions into new BigQuery Standard SQL constrained by the semantic layer.

Demo Mode does include predefined questions and SQL so the classroom demo still works without Gemini credentials, but that path is clearly labeled. The production-style path is natural language -> generated SQL -> validation -> dry-run -> execution.

## Why `vw_bi_dashboard` Is The Query Surface

The assistant is constrained to:

`{project_id}.{dataset_id}.vw_bi_dashboard`

That view is the governed BI decision-support product. It already combines the expected-price regression output, top-price classification output, listing attributes, and quality flags into a BI-ready contract.

Constraining the assistant to this view keeps the demo explainable and prevents the LLM from bypassing the curated business layer or inventing joins against raw tables.

## Why Validation And Dry-Run Are Mandatory

LLM output is treated as a candidate, not as trusted code.

The app blocks unsafe SQL before execution:

- non-SELECT statements
- multiple statements
- comments
- `INFORMATION_SCHEMA`
- `SELECT *`
- forbidden write, DDL, permission, and execution keywords
- tables outside the governed BI view
- unbounded listing-level queries

BigQuery dry-run then estimates scan size before execution. The classroom MVP uses a conservative 50 MB maximum bytes billed threshold so students can see cost governance as part of the architecture. For local demos where BigQuery reports a higher minimum billed amount for the governed view, set `BQ_MAX_BYTES_BILLED_MB` before launching Streamlit.

## What This MVP Can Answer

The assistant can answer governed BI questions such as:

- Which listings should the committee review first?
- Which fuel types concentrate the strongest price opportunities?
- Where do expected-price gap and top-price probability agree?
- Which opportunities have quality or logical risk flags?
- How many listings fall into each decision flag?
- How do actual and expected price compare by registration year?
- Which records are missing model outputs?

It understands the intended storyline:

- regression predicts `expected_price`
- classification predicts `top_price`
- BI combines actual price, expected-price gap, top-price outputs, decision flags, and quality risks

## What It Cannot Answer

The assistant cannot query raw scraping tables, training datasets, model internals, or local Streamlit simulation fields.

It should not generate SQL for local dashboard-only fields such as:

- `expected_total_profit`
- `expected_roi`
- `vehicle_margin`
- `finance_margin`
- `cross_sell_income`
- `investment_score`

Those are computed inside Streamlit for the portfolio simulator, not stored in BigQuery.

The assistant also cannot replace acquisition, credit, compliance, pricing, or risk approval. It is a decision-support layer.

## Production Evolution

A production version would move the orchestration behind a backend API and add:

- user authentication and authorization
- row-level or policy-based permissions
- query logs and audit trails
- cost monitoring and budgets
- owned semantic layer definitions
- prompt versioning and regression tests
- model output monitoring
- data quality checks before answer generation
- human review workflows for high-impact decisions
- approval gates for new metrics and governed views

The core pattern remains the same: generated SQL is only a proposal until the application validates it and the governed warehouse executes it.
