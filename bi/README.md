# Streamlit BI Dashboard

## Purpose

This dashboard is the final business decision layer for the course. It turns upstream model outputs into an executive investment simulator for a consumer finance business evaluating used-vehicle acquisition opportunities.

The dashboard is designed for an investment committee, portfolio committee, or vehicle financing unit. It is not a technical notebook and it is not an automatic approval tool.

## Business Question

With a fixed investment budget, which vehicle portfolio should the business acquire and resell through financed operations, considering resale margin, financing margin, cross-sell income, loyalty value, risk, and commercial campaign coherence?

## Data Source

Primary source:

- BigQuery view: `vw_bi_dashboard`

The app reads `gcp_project_id` and `bq_dataset` from:

- `config/project_config.yaml`

The upstream pipeline provides:

- Regression output: `expected_price_eur`
- Classification output: `top_price_probability`
- BI view: listing profile, actual price, expected-price gap, top-price signal, and decision flag

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

## Gemini API Key

The committee memo is optional. To enable it, configure a Gemini API key in one of these places:

1. Streamlit secrets: `.streamlit/secrets.toml`
2. Environment variable: `GEMINI_API_KEY`

Use the example file:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

The repository includes only:

- `.streamlit/secrets.toml.example`

Never commit a real `.streamlit/secrets.toml` file.

## Dashboard Structure

The app has exactly two tabs:

1. `Committee Dashboard`
2. `Strategy & Portfolio Builder`

The first tab is the executive decision screen. The second tab is the interactive working screen for strategy tuning and portfolio construction.

## Vehicle Strategy Presets

Vehicle strategy presets define the investment mandate. They do not change model outputs.

- `Broad Market`: wide opportunity discovery with minimal filtering.
- `Young Low-Mileage Core`: cleaner, easier-to-sell portfolio with stricter age and mileage filters.
- `Efficient Engine Campaign`: coherent campaign around practical, efficient, mass-market vehicles.
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
- finance margin
- insurance, fuel card, payroll transfer, and loyalty value
- inventory funding cost
- expected total profit
- expected ROI
- resale speed proxy
- portfolio fit weight
- investment score

The portfolio is selected greedily by investment score until the budget, cash buffer, or maximum vehicle count is reached.

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

## Decision-Support Disclaimer

This is a simulated decision-support tool. It is not a final acquisition, credit, compliance, or risk approval.

Regression and classification outputs come from the upstream course pipeline. Streamlit combines those model signals with editable business assumptions so the committee can discuss strategy, economics, and portfolio coherence.
