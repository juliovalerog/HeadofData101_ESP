# Streamlit BI Dashboard

## Purpose

This dashboard is the final business decision layer for the course. It turns upstream model outputs into an executive investment simulator for a consumer finance business evaluating used-vehicle acquisition opportunities.

The dashboard is optional for running the baseline, but recommended as the final decision-support demo after the warehouse and model output tables are populated.

The dashboard is designed for an investment committee, portfolio committee, or vehicle financing unit. It is not a technical notebook, not a production approval engine, and not an automatic acquisition decision tool.

## Business Question

With a fixed investment budget, which vehicle portfolio should the business acquire and resell through financed operations, considering resale margin, financing margin, cross-sell income, loyalty value, risk, and commercial campaign coherence?

## Data Source

Primary source:

- BigQuery view: `vw_bi_dashboard`

The default app path requires BigQuery access. It does not use a bundled mock dataset.

The app reads `gcp_project_id` and `bq_dataset` from:

- `config/project_config.yaml`

The upstream pipeline provides:

- Regression output: `expected_price_eur`
- Classification output: commercial attractiveness score from `top_price_probability`
- BI view: listing profile, actual price, expected-price gap, commercial attractiveness signal, and decision flag

Dashboard-specific investment, financing, cross-sell, risk, and portfolio-selection metrics are computed inside Streamlit.

## How To Run

Install the minimal dependencies:

```bash
pip install -r requirements_min.in
```

Run the dashboard:

```bash
streamlit run bi/streamlit_app.py
```

## BigQuery Authentication

The default path loads data from BigQuery. Authenticate before running the app:

```bash
gcloud auth application-default login
```

Alternatively, configure service account credentials in your execution environment using Google Cloud standard authentication variables.

If credentials are missing or invalid, the app shows a business-friendly error instead of a raw traceback.

The GenAI SQL assistant dry-runs every query and applies a classroom bytes-billed cap. The default is 50 MB. If BigQuery reports that a governed-view query requires a higher minimum billed amount, launch Streamlit with a larger local cap:

```powershell
$env:BQ_MAX_BYTES_BILLED_MB="100"
streamlit run bi/streamlit_app.py
```

## Gemini API Key

Gemini is optional. The committee memo and GenAI SQL assistant use the `google-genai` package when it is installed and an API key is available.

This repo reads Gemini credentials from environment variables only. It does not read Gemini keys from `.streamlit/secrets.toml`, Streamlit Cloud secrets, `.env`, or any committed credential file.

In PowerShell, configure one of:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

or:

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

`GEMINI_API_KEY` takes priority over `GOOGLE_API_KEY`. If neither variable is set, the dashboard keeps working with deterministic/default behavior. The app does not create, print, log, hardcode, or commit API keys.

## Dashboard Structure

The app has three tabs:

1. `Committee Dashboard`
2. `Strategy & Portfolio Builder`
3. `Ask the Data - GenAI SQL Assistant`

The first tab is the executive decision screen. The second tab is the interactive working screen for strategy tuning and portfolio construction. The third tab is a governed text-to-SQL assistant over the BI dashboard view.

## Vehicle Strategy Presets

Vehicle strategy presets define the investment mandate. They do not change model outputs.

- `Broad Market`: wide opportunity discovery with minimal filtering.
- `Young Low-Mileage Core`: cleaner, easier-to-sell portfolio with stricter age and mileage filters.
- `Mainstream Retail Campaign`: coherent campaign around mainstream, easy-to-explain vehicles.
- `Higher-Ticket Margin`: higher unit profit with stronger capital concentration risk.
- `Conservative Risk`: defensibility over volume, with stronger quality and model-signal thresholds.

After a preset is selected, every assumption remains editable in the sidebar.

## Pricing And Cross-Sell Presets

Pricing and cross-sell presets define the monetization strategy.

- `Base Case`: balanced default assumptions.
- `Aggressive Cross-Sell`: stronger bundle economics and higher attach rates.
- `Finance Margin Focus`: higher direct financing margin with lower cross-sell emphasis.
- `Customer Loyalty Focus`: stronger long-term relationship value and more generous APR discounts.
- `Stress Case`: adverse resale, funding, risk, and commercial assumptions.

Manual overrides are detected and shown in the sidebar and strategy overview.

## Main Business Calculations

For each eligible vehicle, the app estimates:

- expected discount versus model-predicted market price
- conservative resale price
- vehicle resale margin
- capital deployed
- finance margin after expected weighted APR discounts
- insurance, fuel card, payroll transfer, and loyalty value
- inventory funding cost
- expected total profit
- expected ROI
- resale speed proxy
- portfolio fit weight
- investment score

The recommended portfolio under the current strategy is selected by a transparent ranking rule. Vehicles are ranked by investment score and selected until the budget, cash buffer, or maximum vehicle count is reached. This is not a full mathematical optimization model.

## Gemini Committee Memo

The optional Gemini memo receives only aggregated strategy context, assumptions, warnings, and the top selected candidates. It does not send unnecessary raw data.

The memo includes:

1. Selected Strategy and Investment Mandate
2. Executive Recommendation
3. Portfolio Rationale
4. Key Assumptions
5. Expected Financial Result
6. Cross-Sell and Loyalty Logic
7. Main Risks and Model Limitations
8. Suggested Next Validation Steps

## Suggested 10-Minute Live Demo

1. Start with `Base Case` and `Broad Market`.
2. Explain the committee decision box.
3. Switch to `Aggressive Cross-Sell` and show how value drivers change.
4. Switch to `Conservative Risk` and show how the eligible universe shrinks.
5. Adjust investment budget and show selected portfolio changes.
6. Modify age and mileage filters and explain the investment mandate.
7. Generate the Gemini committee memo.
8. Conclude: "The model does not decide the strategy; it allows the committee to compare strategies with data."

Do not explain every formula during the demo. Focus on how changing the business strategy changes the eligible universe, the selected portfolio, the value drivers, and the committee recommendation.

## Teaching Message

- The dashboard is the final decision layer of the pipeline.
- Regression creates expected market price.
- Classification creates commercial attractiveness.
- Business assumptions create the investment strategy.
- BI converts all of this into a committee-ready decision product.

The model does not decide the strategy; it allows the committee to compare strategies with data.

## Decision-Support Disclaimer

This is a simulated decision-support tool. It is not a final acquisition, credit, compliance, or risk approval.

Regression and classification outputs come from the upstream course pipeline. Streamlit combines those model signals with editable business assumptions so the committee can discuss strategy, economics, and portfolio coherence.
