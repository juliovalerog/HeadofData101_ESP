from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google.api_core.exceptions import BadRequest, Forbidden, GoogleAPIError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

try:
    from gemini_helper import generate_gemini_content, gemini_available, gemini_unavailable_message
except ImportError:
    from bi.gemini_helper import generate_gemini_content, gemini_available, gemini_unavailable_message


REQUIRED_COLUMNS = [
    "listing_id",
    "brand",
    "make",
    "model",
    "fuel_type",
    "listing_country",
    "price_label",
    "actual_price_eur",
    "mileage_km",
    "power_hp",
    "registration_date",
    "registration_year",
    "registration_month",
    "age_years",
    "page",
    "price_outlier_iqr",
    "mileage_outlier_iqr",
    "power_outlier_iqr",
    "logical_issue",
    "expected_price_eur",
    "expected_price_gap_eur",
    "top_price_probability",
    "predicted_top_price",
    "decision_flag",
]

VEHICLE_PRESETS = [
    "Broad Market",
    "Young Low-Mileage Core",
    "Mainstream Retail Campaign",
    "Higher-Ticket Margin",
    "Conservative Risk",
]

PRICING_PRESETS = [
    "Base Case",
    "Aggressive Cross-Sell",
    "Finance Margin Focus",
    "Customer Loyalty Focus",
    "Stress Case",
]

PERCENT_SLIDER_KEYS = {
    "min_top_price_probability",
    "min_expected_discount_pct",
    "resale_haircut",
    "inventory_funding_cost_rate",
    "financed_amount_pct",
    "base_customer_apr",
    "funding_cost_rate",
    "credit_risk_cost_rate",
    "financing_take_up_rate",
    "min_net_finance_margin_buffer",
    "insurance_attach_rate",
    "insurance_apr_discount",
    "fuel_card_attach_rate",
    "fuel_card_apr_discount",
    "payroll_attach_rate",
    "payroll_apr_discount",
}

def env_megabytes(name: str, default: int) -> int:
    try:
        value = os.getenv(name)
        if value:
            return int(float(value))
    except ValueError:
        pass
    return default


MAX_BYTES_BILLED = env_megabytes("BQ_MAX_BYTES_BILLED_MB", 50) * 1024 * 1024
ALLOWED_BI_VIEW_TEMPLATE = "`{project_id}.{dataset_id}.vw_bi_dashboard`"

AI_EXAMPLE_PROMPTS = [
    "Which listings should the committee review first?",
    "Which fuel types concentrate the strongest price opportunities?",
    "Where do expected-price gap and top-price probability agree?",
    "Which opportunities are risky because of quality flags?",
    "Summarize the decision_flag distribution.",
    "Compare actual vs expected price by registration year.",
    "Show cases where the model outputs are missing.",
]

DEMO_QUESTIONS = [
    "Which listings should the committee review first?",
    "Which fuel types concentrate the strongest price opportunities?",
    "Where do regression and classification signals agree?",
    "Which opportunities have quality or logical risk flags?",
    "How many listings fall into each decision flag?",
]

DECISION_FLAG_MEANINGS = {
    "high_priority_review": "below expected price and strong top-price signal",
    "price_opportunity": "below expected price but the top-price signal is weaker or missing",
    "top_price_signal": "positive classification signal, price gap not necessarily attractive",
    "review_missing_model_outputs": "model outputs are missing and need review",
    "standard_review": "no strong priority signal",
}

BI_FIELD_MEANINGS = {
    "actual_price_eur": "observed marketplace listing price",
    "expected_price_eur": "regression model estimate of normal market price",
    "expected_price_gap_eur": "actual price minus expected price; negative means below model-expected value",
    "top_price_probability": "classification model probability associated with the external top-price signal",
    "predicted_top_price": "binary model output for the top-price classification task",
    "decision_flag": "business priority flag produced in the BI view",
    "price_outlier_iqr": "IQR-based price quality and risk flag",
    "mileage_outlier_iqr": "IQR-based mileage quality and risk flag",
    "power_outlier_iqr": "IQR-based power quality and risk flag",
    "logical_issue": "logical data quality or business rule issue flag",
}


st.set_page_config(
    page_title="Vehicle Portfolio Investment Simulator",
    layout="wide",
)


def find_repo_root(start: Path) -> Path:
    for path in [start] + list(start.parents):
        if (path / ".git").exists() or (path / "config" / "project_config.yaml").exists():
            return path
    return start


def load_project_config() -> dict[str, Any]:
    config_path = find_repo_root(Path(__file__).resolve()) / "config" / "project_config.yaml"
    config: dict[str, Any] = {}
    if not config_path.exists():
        return config

    for raw_line in config_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        config[key.strip()] = value
    return config


def query_text(project_id: str, dataset_id: str) -> str:
    table_name = f"`{project_id}.{dataset_id}.vw_bi_dashboard`"
    return f"""
    SELECT
      {", ".join(REQUIRED_COLUMNS)}
    FROM {table_name}
    """


@st.cache_data(ttl=900, show_spinner="Loading BI dashboard data from BigQuery...")
def load_dashboard_data(project_id: str, dataset_id: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        client = bigquery.Client(project=project_id)
        df = client.query(query_text(project_id, dataset_id)).to_dataframe()
        return df, None
    except DefaultCredentialsError:
        return None, (
            "BigQuery credentials are not configured. Authenticate with Google Cloud before running "
            "the dashboard, for example with `gcloud auth application-default login`, or configure "
            "the service account credentials used by your environment."
        )
    except GoogleAPIError:
        return None, (
            "BigQuery could not be reached or the configured project and dataset are unavailable. "
            "Check the Google Cloud project, dataset, permissions, and upstream pipeline status."
        )
    except Exception:
        return None, (
            "The dashboard could not load the BI view. Check credentials, network access, and the "
            "configured BigQuery project and dataset."
        )


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"])


def clean_dashboard_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_columns = [
        "actual_price_eur",
        "mileage_km",
        "power_hp",
        "registration_year",
        "registration_month",
        "age_years",
        "expected_price_eur",
        "expected_price_gap_eur",
        "top_price_probability",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in ["price_outlier_iqr", "mileage_outlier_iqr", "power_outlier_iqr", "logical_issue"]:
        cleaned[column] = normalize_bool(cleaned[column])

    cleaned["fuel_type"] = cleaned["fuel_type"].fillna("Unknown").astype(str)
    cleaned["quality_issue"] = cleaned[
        ["price_outlier_iqr", "mileage_outlier_iqr", "power_outlier_iqr", "logical_issue"]
    ].any(axis=1)
    return cleaned


def clipped(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def numeric_bounds(df: pd.DataFrame, column: str) -> tuple[float, float]:
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return 0.0, 1.0
    return float(values.min()), float(values.max())


def quantile(df: pd.DataFrame, column: str, q: float) -> float:
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.quantile(q))


def mainstream_fuel_defaults(df: pd.DataFrame, fuel_types: list[str]) -> list[str]:
    frequent_fuels = df["fuel_type"].dropna().astype(str).value_counts().head(3).index.tolist()
    return frequent_fuels if frequent_fuels else fuel_types[:3]


def build_vehicle_preset_defaults(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    age_min, age_max = numeric_bounds(df, "age_years")
    mileage_min, mileage_max = numeric_bounds(df, "mileage_km")
    power_min, power_max = numeric_bounds(df, "power_hp")
    price_min, price_max = numeric_bounds(df, "actual_price_eur")
    fuel_types = sorted(df["fuel_type"].dropna().astype(str).unique().tolist())

    q = lambda column, value: quantile(df, column, value)

    return {
        "Broad Market": {
            "age_range": (age_min, age_max),
            "mileage_range": (mileage_min, mileage_max),
            "power_range": (power_min, power_max),
            "price_range": (price_min, price_max),
            "fuel_types": fuel_types,
            "min_top_price_probability": 0.25,
            "min_expected_discount_pct": 0.00,
            "exclude_price_outlier": False,
            "exclude_mileage_outlier": False,
            "exclude_power_outlier": False,
            "exclude_logical_issue": False,
            "expected_days_to_resale": 90,
        },
        "Young Low-Mileage Core": {
            "age_range": (age_min, min(age_max, q("age_years", 0.55))),
            "mileage_range": (mileage_min, min(mileage_max, q("mileage_km", 0.55))),
            "power_range": (power_min, power_max),
            "price_range": (q("actual_price_eur", 0.10), q("actual_price_eur", 0.85)),
            "fuel_types": fuel_types,
            "min_top_price_probability": 0.45,
            "min_expected_discount_pct": 0.03,
            "exclude_price_outlier": False,
            "exclude_mileage_outlier": True,
            "exclude_power_outlier": False,
            "exclude_logical_issue": True,
            "expected_days_to_resale": 75,
        },
        "Mainstream Retail Campaign": {
            "age_range": (age_min, min(age_max, q("age_years", 0.75))),
            "mileage_range": (mileage_min, min(mileage_max, q("mileage_km", 0.70))),
            "power_range": (power_min, power_max),
            "price_range": (q("actual_price_eur", 0.10), q("actual_price_eur", 0.80)),
            "fuel_types": mainstream_fuel_defaults(df, fuel_types),
            "min_top_price_probability": 0.40,
            "min_expected_discount_pct": 0.02,
            "exclude_price_outlier": False,
            "exclude_mileage_outlier": True,
            "exclude_power_outlier": False,
            "exclude_logical_issue": True,
            "expected_days_to_resale": 70,
        },
        "Higher-Ticket Margin": {
            "age_range": (age_min, age_max),
            "mileage_range": (mileage_min, mileage_max),
            "power_range": (power_min, power_max),
            "price_range": (q("actual_price_eur", 0.55), price_max),
            "fuel_types": fuel_types,
            "min_top_price_probability": 0.45,
            "min_expected_discount_pct": 0.04,
            "exclude_price_outlier": False,
            "exclude_mileage_outlier": False,
            "exclude_power_outlier": False,
            "exclude_logical_issue": True,
            "expected_days_to_resale": 110,
        },
        "Conservative Risk": {
            "age_range": (age_min, min(age_max, q("age_years", 0.50))),
            "mileage_range": (mileage_min, min(mileage_max, q("mileage_km", 0.50))),
            "power_range": (power_min, power_max),
            "price_range": (q("actual_price_eur", 0.10), q("actual_price_eur", 0.75)),
            "fuel_types": fuel_types,
            "min_top_price_probability": 0.60,
            "min_expected_discount_pct": 0.06,
            "exclude_price_outlier": True,
            "exclude_mileage_outlier": True,
            "exclude_power_outlier": True,
            "exclude_logical_issue": True,
            "expected_days_to_resale": 65,
        },
    }


def pricing_preset_defaults() -> dict[str, dict[str, Any]]:
    return {
        "Base Case": {
            "resale_haircut": 0.06,
            "reconditioning_cost": 850,
            "transaction_cost": 450,
            "inventory_funding_cost_rate": 0.055,
            "financed_amount_pct": 0.80,
            "base_customer_apr": 0.095,
            "funding_cost_rate": 0.045,
            "credit_risk_cost_rate": 0.018,
            "loan_term_months": 48,
            "financing_take_up_rate": 0.55,
            "min_net_finance_margin_buffer": 0.010,
            "insurance_attach_rate": 0.35,
            "insurance_commission": 380,
            "insurance_apr_discount": 0.004,
            "fuel_card_attach_rate": 0.22,
            "fuel_card_annual_margin": 90,
            "fuel_card_years": 2.0,
            "fuel_card_apr_discount": 0.002,
            "payroll_attach_rate": 0.16,
            "payroll_lifetime_value": 420,
            "payroll_apr_discount": 0.003,
            "attach_rate_elasticity": 4.0,
        },
        "Aggressive Cross-Sell": {
            "resale_haircut": 0.06,
            "reconditioning_cost": 850,
            "transaction_cost": 450,
            "inventory_funding_cost_rate": 0.055,
            "financed_amount_pct": 0.82,
            "base_customer_apr": 0.090,
            "funding_cost_rate": 0.045,
            "credit_risk_cost_rate": 0.018,
            "loan_term_months": 48,
            "financing_take_up_rate": 0.62,
            "min_net_finance_margin_buffer": 0.008,
            "insurance_attach_rate": 0.52,
            "insurance_commission": 430,
            "insurance_apr_discount": 0.008,
            "fuel_card_attach_rate": 0.42,
            "fuel_card_annual_margin": 110,
            "fuel_card_years": 2.5,
            "fuel_card_apr_discount": 0.005,
            "payroll_attach_rate": 0.36,
            "payroll_lifetime_value": 620,
            "payroll_apr_discount": 0.006,
            "attach_rate_elasticity": 5.5,
        },
        "Finance Margin Focus": {
            "resale_haircut": 0.055,
            "reconditioning_cost": 800,
            "transaction_cost": 450,
            "inventory_funding_cost_rate": 0.055,
            "financed_amount_pct": 0.82,
            "base_customer_apr": 0.112,
            "funding_cost_rate": 0.045,
            "credit_risk_cost_rate": 0.020,
            "loan_term_months": 54,
            "financing_take_up_rate": 0.58,
            "min_net_finance_margin_buffer": 0.014,
            "insurance_attach_rate": 0.28,
            "insurance_commission": 330,
            "insurance_apr_discount": 0.002,
            "fuel_card_attach_rate": 0.16,
            "fuel_card_annual_margin": 75,
            "fuel_card_years": 1.5,
            "fuel_card_apr_discount": 0.001,
            "payroll_attach_rate": 0.12,
            "payroll_lifetime_value": 330,
            "payroll_apr_discount": 0.001,
            "attach_rate_elasticity": 2.0,
        },
        "Customer Loyalty Focus": {
            "resale_haircut": 0.065,
            "reconditioning_cost": 900,
            "transaction_cost": 450,
            "inventory_funding_cost_rate": 0.055,
            "financed_amount_pct": 0.78,
            "base_customer_apr": 0.088,
            "funding_cost_rate": 0.045,
            "credit_risk_cost_rate": 0.018,
            "loan_term_months": 48,
            "financing_take_up_rate": 0.60,
            "min_net_finance_margin_buffer": 0.007,
            "insurance_attach_rate": 0.50,
            "insurance_commission": 410,
            "insurance_apr_discount": 0.008,
            "fuel_card_attach_rate": 0.30,
            "fuel_card_annual_margin": 95,
            "fuel_card_years": 2.5,
            "fuel_card_apr_discount": 0.004,
            "payroll_attach_rate": 0.44,
            "payroll_lifetime_value": 800,
            "payroll_apr_discount": 0.007,
            "attach_rate_elasticity": 5.0,
        },
        "Stress Case": {
            "resale_haircut": 0.12,
            "reconditioning_cost": 1000,
            "transaction_cost": 500,
            "inventory_funding_cost_rate": 0.075,
            "financed_amount_pct": 0.72,
            "base_customer_apr": 0.100,
            "funding_cost_rate": 0.065,
            "credit_risk_cost_rate": 0.035,
            "loan_term_months": 42,
            "financing_take_up_rate": 0.42,
            "min_net_finance_margin_buffer": 0.012,
            "insurance_attach_rate": 0.24,
            "insurance_commission": 320,
            "insurance_apr_discount": 0.003,
            "fuel_card_attach_rate": 0.12,
            "fuel_card_annual_margin": 70,
            "fuel_card_years": 1.5,
            "fuel_card_apr_discount": 0.001,
            "payroll_attach_rate": 0.10,
            "payroll_lifetime_value": 300,
            "payroll_apr_discount": 0.002,
            "attach_rate_elasticity": 1.5,
        },
    }


def set_prefixed_defaults(prefix: str, defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        st.session_state[f"{prefix}_{key}"] = value * 100 if key in PERCENT_SLIDER_KEYS else value


def reset_all_assumptions(vehicle_preset: str, pricing_preset: str, vehicle_defaults: dict, pricing_defaults: dict) -> None:
    set_prefixed_defaults("vehicle", vehicle_defaults[vehicle_preset])
    set_prefixed_defaults("pricing", pricing_defaults[pricing_preset])
    st.session_state["budget_eur"] = 3_000_000
    st.session_state["max_vehicles"] = 100
    st.session_state["cash_buffer_pct"] = 0.0
    st.session_state["allow_missing_model_outputs"] = False
    st.session_state["_active_vehicle_preset"] = vehicle_preset
    st.session_state["_active_pricing_preset"] = pricing_preset


def populate_defaults_if_needed(vehicle_preset: str, pricing_preset: str, vehicle_defaults: dict, pricing_defaults: dict) -> None:
    if "budget_eur" not in st.session_state:
        st.session_state["budget_eur"] = 3_000_000
    if "max_vehicles" not in st.session_state:
        st.session_state["max_vehicles"] = 100
    if "cash_buffer_pct" not in st.session_state:
        st.session_state["cash_buffer_pct"] = 0.0
    if "allow_missing_model_outputs" not in st.session_state:
        st.session_state["allow_missing_model_outputs"] = False

    if st.session_state.get("_active_vehicle_preset") != vehicle_preset:
        set_prefixed_defaults("vehicle", vehicle_defaults[vehicle_preset])
        st.session_state["_active_vehicle_preset"] = vehicle_preset
    if st.session_state.get("_active_pricing_preset") != pricing_preset:
        set_prefixed_defaults("pricing", pricing_defaults[pricing_preset])
        st.session_state["_active_pricing_preset"] = pricing_preset


def percent_slider(label: str, key: str, min_value: float, max_value: float, step: float = 0.5) -> float:
    value = st.sidebar.slider(label, min_value, max_value, step=step, key=key)
    return value / 100


def render_sidebar(df: pd.DataFrame) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    st.sidebar.title("Strategy Configuration")
    vehicle_preset = st.sidebar.selectbox("Vehicle Filter Strategy", VEHICLE_PRESETS)
    pricing_preset = st.sidebar.selectbox("Pricing and Cross-Sell Strategy", PRICING_PRESETS)

    vehicle_defaults = build_vehicle_preset_defaults(df)
    pricing_defaults = pricing_preset_defaults()
    populate_defaults_if_needed(vehicle_preset, pricing_preset, vehicle_defaults, pricing_defaults)

    if st.sidebar.button("Reset all assumptions"):
        reset_all_assumptions(vehicle_preset, pricing_preset, vehicle_defaults, pricing_defaults)
        st.rerun()

    st.sidebar.info(
        "Strategy presets are business assumptions. They define the investment mandate and commercial campaign "
        "logic. They do not change the underlying model outputs."
    )

    with st.sidebar.expander("Investment controls", expanded=True):
        budget = st.number_input("Investment budget in EUR", min_value=0, step=50_000, key="budget_eur")
        max_vehicles = st.number_input("Maximum number of vehicles", min_value=1, step=1, key="max_vehicles")
        cash_buffer_pct = percent_slider("Minimum cash buffer %", "cash_buffer_pct", 0.0, 50.0, 1.0)

    age_min, age_max = numeric_bounds(df, "age_years")
    mileage_min, mileage_max = numeric_bounds(df, "mileage_km")
    power_min, power_max = numeric_bounds(df, "power_hp")
    price_min, price_max = numeric_bounds(df, "actual_price_eur")
    fuel_options = sorted(df["fuel_type"].dropna().astype(str).unique().tolist())

    with st.sidebar.expander("Vehicle filters", expanded=True):
        age_range = st.slider("Age range", age_min, age_max, key="vehicle_age_range")
        mileage_range = st.slider("Mileage range", mileage_min, mileage_max, key="vehicle_mileage_range")
        power_range = st.slider("Power range", power_min, power_max, key="vehicle_power_range")
        price_range = st.slider("Purchase price range", price_min, price_max, key="vehicle_price_range")
        fuel_types = st.multiselect("Fuel types", fuel_options, key="vehicle_fuel_types")
        min_top_price_probability = percent_slider(
            "Minimum commercial attractiveness score",
            "vehicle_min_top_price_probability",
            0.0,
            100.0,
            1.0,
        )
        st.caption("This score comes from the upstream classification model field `top_price_probability`.")
        min_expected_discount_pct = percent_slider(
            "Minimum expected discount %",
            "vehicle_min_expected_discount_pct",
            -50.0,
            50.0,
            1.0,
        )
        exclude_price_outlier = st.checkbox("Exclude price outlier flag", key="vehicle_exclude_price_outlier")
        exclude_mileage_outlier = st.checkbox("Exclude mileage outlier flag", key="vehicle_exclude_mileage_outlier")
        exclude_power_outlier = st.checkbox("Exclude power outlier flag", key="vehicle_exclude_power_outlier")
        exclude_logical_issue = st.checkbox("Exclude logical issue flag", key="vehicle_exclude_logical_issue")
        allow_missing_model_outputs = st.checkbox(
            "Allow missing model outputs",
            key="allow_missing_model_outputs",
            help="Disabled by default because the portfolio should use both regression and classification signals.",
        )

    with st.sidebar.expander("Resale economics", expanded=False):
        resale_haircut = percent_slider("Conservative resale haircut over expected price", "pricing_resale_haircut", 0.0, 40.0)
        reconditioning_cost = st.number_input("Reconditioning cost per vehicle", min_value=0, step=50, key="pricing_reconditioning_cost")
        transaction_cost = st.number_input("Transaction cost per vehicle", min_value=0, step=50, key="pricing_transaction_cost")
        inventory_funding_cost_rate = percent_slider("Annual inventory funding cost rate", "pricing_inventory_funding_cost_rate", 0.0, 30.0)
        expected_days_to_resale = st.number_input("Expected days to resale", min_value=1, max_value=365, step=5, key="vehicle_expected_days_to_resale")

    with st.sidebar.expander("Financing assumptions", expanded=False):
        financed_amount_pct = percent_slider("Expected financed amount as % of resale price", "pricing_financed_amount_pct", 0.0, 100.0)
        base_customer_apr = percent_slider("Base customer APR", "pricing_base_customer_apr", 0.0, 30.0)
        funding_cost_rate = percent_slider("Funding cost rate", "pricing_funding_cost_rate", 0.0, 20.0)
        credit_risk_cost_rate = percent_slider("Expected credit risk cost rate", "pricing_credit_risk_cost_rate", 0.0, 20.0)
        loan_term_months = st.number_input("Average loan term in months", min_value=1, max_value=120, step=1, key="pricing_loan_term_months")
        financing_take_up_rate = percent_slider("Financing take-up rate", "pricing_financing_take_up_rate", 0.0, 100.0)
        min_net_finance_margin_buffer = percent_slider("Minimum net finance margin buffer", "pricing_min_net_finance_margin_buffer", 0.0, 10.0, 0.25)

    with st.sidebar.expander("Cross-sell and loyalty", expanded=False):
        insurance_attach_rate = percent_slider("Insurance attach rate base", "pricing_insurance_attach_rate", 0.0, 100.0)
        insurance_commission = st.number_input("Insurance commission per policy", min_value=0, step=25, key="pricing_insurance_commission")
        insurance_apr_discount = percent_slider("Interest rate discount if insurance is contracted", "pricing_insurance_apr_discount", 0.0, 10.0, 0.25)
        fuel_card_attach_rate = percent_slider("Fuel card attach rate base", "pricing_fuel_card_attach_rate", 0.0, 100.0)
        fuel_card_annual_margin = st.number_input("Fuel card annual margin", min_value=0, step=10, key="pricing_fuel_card_annual_margin")
        fuel_card_years = st.number_input("Fuel card expected duration in years", min_value=0.0, max_value=10.0, step=0.25, key="pricing_fuel_card_years")
        fuel_card_apr_discount = percent_slider("Interest rate discount if fuel card is contracted", "pricing_fuel_card_apr_discount", 0.0, 10.0, 0.25)
        payroll_attach_rate = percent_slider("Payroll transfer attach rate base", "pricing_payroll_attach_rate", 0.0, 100.0)
        payroll_lifetime_value = st.number_input("Payroll customer lifetime value", min_value=0, step=25, key="pricing_payroll_lifetime_value")
        payroll_apr_discount = percent_slider("Interest rate discount if payroll is transferred", "pricing_payroll_apr_discount", 0.0, 10.0, 0.25)
        attach_rate_elasticity = st.number_input(
            "Elasticity parameter",
            min_value=0.0,
            max_value=20.0,
            step=0.25,
            key="pricing_attach_rate_elasticity",
            help="Each 1 percentage point APR discount increases attach rate by this many percentage points.",
        )

    vehicle_params = {
        "age_range": age_range,
        "mileage_range": mileage_range,
        "power_range": power_range,
        "price_range": price_range,
        "fuel_types": fuel_types,
        "min_top_price_probability": min_top_price_probability,
        "min_expected_discount_pct": min_expected_discount_pct,
        "exclude_price_outlier": exclude_price_outlier,
        "exclude_mileage_outlier": exclude_mileage_outlier,
        "exclude_power_outlier": exclude_power_outlier,
        "exclude_logical_issue": exclude_logical_issue,
        "expected_days_to_resale": expected_days_to_resale,
        "allow_missing_model_outputs": allow_missing_model_outputs,
    }
    pricing_params = {
        "resale_haircut": resale_haircut,
        "reconditioning_cost": reconditioning_cost,
        "transaction_cost": transaction_cost,
        "inventory_funding_cost_rate": inventory_funding_cost_rate,
        "financed_amount_pct": financed_amount_pct,
        "base_customer_apr": base_customer_apr,
        "funding_cost_rate": funding_cost_rate,
        "credit_risk_cost_rate": credit_risk_cost_rate,
        "loan_term_months": loan_term_months,
        "financing_take_up_rate": clipped(financing_take_up_rate),
        "min_net_finance_margin_buffer": min_net_finance_margin_buffer,
        "insurance_attach_rate": clipped(insurance_attach_rate),
        "insurance_commission": insurance_commission,
        "insurance_apr_discount": insurance_apr_discount,
        "fuel_card_attach_rate": clipped(fuel_card_attach_rate),
        "fuel_card_annual_margin": fuel_card_annual_margin,
        "fuel_card_years": fuel_card_years,
        "fuel_card_apr_discount": fuel_card_apr_discount,
        "payroll_attach_rate": clipped(payroll_attach_rate),
        "payroll_lifetime_value": payroll_lifetime_value,
        "payroll_apr_discount": payroll_apr_discount,
        "attach_rate_elasticity": attach_rate_elasticity,
    }
    budget_params = {
        "budget_eur": budget,
        "max_vehicles": int(max_vehicles),
        "cash_buffer_pct": cash_buffer_pct,
    }

    customized = assumptions_customized(
        vehicle_preset,
        pricing_preset,
        vehicle_params,
        pricing_params,
        budget_params,
        vehicle_defaults,
        pricing_defaults,
    )
    st.sidebar.caption(f"Manual overrides active: {'Yes' if customized else 'No'}")
    return vehicle_preset, pricing_preset, vehicle_params, pricing_params, budget_params, customized


def comparable(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(round(float(item), 4) for item in value)
    if isinstance(value, list):
        return sorted(value)
    if isinstance(value, float):
        return round(value, 4)
    return value


def assumptions_customized(
    vehicle_preset: str,
    pricing_preset: str,
    vehicle_params: dict[str, Any],
    pricing_params: dict[str, Any],
    budget_params: dict[str, Any],
    vehicle_defaults: dict[str, dict[str, Any]],
    pricing_defaults: dict[str, dict[str, Any]],
) -> bool:
    vehicle_default = vehicle_defaults[vehicle_preset]
    pricing_default = pricing_defaults[pricing_preset]
    for key, value in vehicle_params.items():
        if key == "allow_missing_model_outputs":
            if value:
                return True
            continue
        if comparable(value) != comparable(vehicle_default[key]):
            return True
    for key, value in pricing_params.items():
        if comparable(value) != comparable(pricing_default[key]):
            return True
    return (
        budget_params["budget_eur"] != 3_000_000
        or budget_params["max_vehicles"] != 100
        or round(budget_params["cash_buffer_pct"], 4) != 0.0
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator.replace(0, np.nan).notna(), numerator / denominator.replace(0, np.nan), np.nan)


def normalized_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    low = values.min()
    high = values.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(0.5, index=series.index)
    return ((values - low) / (high - low)).clip(0, 1)


def normalized_inverse(series: pd.Series) -> pd.Series:
    return 1 - normalized_positive(series)


def compute_economics(
    df: pd.DataFrame,
    vehicle_params: dict[str, Any],
    pricing_params: dict[str, Any],
    vehicle_preset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    result["expected_discount_eur"] = result["expected_price_eur"] - result["actual_price_eur"]
    result["expected_discount_pct"] = safe_divide(result["expected_discount_eur"], result["expected_price_eur"])
    result["conservative_resale_price"] = result["expected_price_eur"] * (1 - pricing_params["resale_haircut"])
    result["gross_resale_spread"] = result["conservative_resale_price"] - result["actual_price_eur"]
    result["vehicle_margin"] = (
        result["gross_resale_spread"]
        - pricing_params["reconditioning_cost"]
        - pricing_params["transaction_cost"]
    )
    result["capital_deployed"] = (
        result["actual_price_eur"] + pricing_params["reconditioning_cost"] + pricing_params["transaction_cost"]
    )
    result["reconditioning_cost"] = pricing_params["reconditioning_cost"]
    result["transaction_cost"] = pricing_params["transaction_cost"]
    result["financed_amount"] = result["conservative_resale_price"] * pricing_params["financed_amount_pct"]

    # APR discounts improve the product bundle. The elasticity translates discount size into attach-rate uplift.
    adjusted_insurance_attach_rate = clipped(
        pricing_params["insurance_attach_rate"]
        + pricing_params["attach_rate_elasticity"] * pricing_params["insurance_apr_discount"]
    )
    adjusted_fuel_card_attach_rate = clipped(
        pricing_params["fuel_card_attach_rate"]
        + pricing_params["attach_rate_elasticity"] * pricing_params["fuel_card_apr_discount"]
    )
    adjusted_payroll_attach_rate = clipped(
        pricing_params["payroll_attach_rate"]
        + pricing_params["attach_rate_elasticity"] * pricing_params["payroll_apr_discount"]
    )
    expected_insurance_discount = adjusted_insurance_attach_rate * pricing_params["insurance_apr_discount"]
    expected_fuel_card_discount = adjusted_fuel_card_attach_rate * pricing_params["fuel_card_apr_discount"]
    expected_payroll_discount = adjusted_payroll_attach_rate * pricing_params["payroll_apr_discount"]
    expected_total_apr_discount = (
        expected_insurance_discount + expected_fuel_card_discount + expected_payroll_discount
    )
    raw_effective_apr = pricing_params["base_customer_apr"] - expected_total_apr_discount
    apr_floor = (
        pricing_params["funding_cost_rate"]
        + pricing_params["credit_risk_cost_rate"]
        + pricing_params["min_net_finance_margin_buffer"]
    )
    effective_customer_apr = max(raw_effective_apr, apr_floor)
    net_finance_spread = effective_customer_apr - pricing_params["funding_cost_rate"] - pricing_params["credit_risk_cost_rate"]

    result["effective_customer_apr"] = effective_customer_apr
    result["net_finance_spread"] = net_finance_spread
    result["finance_margin"] = (
        result["financed_amount"]
        * net_finance_spread
        * (pricing_params["loan_term_months"] / 12)
        * 0.5
        * pricing_params["financing_take_up_rate"]
    )

    result["adjusted_insurance_attach_rate"] = adjusted_insurance_attach_rate
    result["adjusted_fuel_card_attach_rate"] = adjusted_fuel_card_attach_rate
    result["adjusted_payroll_attach_rate"] = adjusted_payroll_attach_rate

    result["insurance_income"] = result["adjusted_insurance_attach_rate"] * pricing_params["insurance_commission"]
    result["fuel_card_income"] = (
        result["adjusted_fuel_card_attach_rate"]
        * pricing_params["fuel_card_annual_margin"]
        * pricing_params["fuel_card_years"]
    )
    result["payroll_income"] = result["adjusted_payroll_attach_rate"] * pricing_params["payroll_lifetime_value"]
    result["cross_sell_income"] = result["insurance_income"] + result["fuel_card_income"] + result["payroll_income"]
    result["inventory_funding_cost"] = (
        result["capital_deployed"]
        * pricing_params["inventory_funding_cost_rate"]
        * vehicle_params["expected_days_to_resale"]
        / 365
    )
    result["expected_total_profit"] = (
        result["vehicle_margin"]
        + result["finance_margin"]
        + result["cross_sell_income"]
        - result["inventory_funding_cost"]
    )
    result["expected_roi"] = safe_divide(result["expected_total_profit"], result["capital_deployed"])
    result["quality_weight"] = np.where(result["quality_issue"], 0.7, 1.0)

    # Resale speed is a transparent proxy, not a trained model: attractive score, lower age, and lower mileage.
    result["resale_speed_score"] = (
        0.50 * result["top_price_probability"].fillna(0)
        + 0.25 * normalized_inverse(result["age_years"])
        + 0.25 * normalized_inverse(result["mileage_km"])
    ).clip(0, 1)
    result["portfolio_fit_weight"] = portfolio_fit_weight(result, vehicle_preset, vehicle_params)
    result["investment_score"] = (
        result["expected_roi"].fillna(-1)
        * result["top_price_probability"].fillna(0)
        * result["resale_speed_score"].fillna(0)
        * result["quality_weight"]
        * result["portfolio_fit_weight"].fillna(1)
    )

    metadata = {
        "raw_effective_customer_apr": raw_effective_apr,
        "effective_customer_apr": effective_customer_apr,
        "expected_total_apr_discount": expected_total_apr_discount,
        "expected_insurance_discount": expected_insurance_discount,
        "expected_fuel_card_discount": expected_fuel_card_discount,
        "expected_payroll_discount": expected_payroll_discount,
        "apr_floor": apr_floor,
        "apr_floor_applied": effective_customer_apr > raw_effective_apr,
        "net_finance_spread": net_finance_spread,
        "adjusted_insurance_attach_rate": adjusted_insurance_attach_rate,
        "adjusted_fuel_card_attach_rate": adjusted_fuel_card_attach_rate,
        "adjusted_payroll_attach_rate": adjusted_payroll_attach_rate,
    }
    return result.replace([np.inf, -np.inf], np.nan), metadata


def portfolio_fit_weight(df: pd.DataFrame, vehicle_preset: str, vehicle_params: dict[str, Any]) -> pd.Series:
    if vehicle_preset == "Broad Market":
        return pd.Series(1.0, index=df.index)
    if vehicle_preset == "Young Low-Mileage Core":
        return (0.80 + 0.20 * normalized_inverse(df["age_years"]) + 0.20 * normalized_inverse(df["mileage_km"])).clip(0.8, 1.2)
    if vehicle_preset == "Mainstream Retail Campaign":
        selected_fuels = set(vehicle_params["fuel_types"])
        fuel_reward = df["fuel_type"].isin(selected_fuels).astype(float)
        moderate_price = 1 - (normalized_positive(df["actual_price_eur"]) - 0.5).abs() * 2
        return (0.85 + 0.20 * fuel_reward + 0.15 * moderate_price).clip(0.75, 1.2)
    if vehicle_preset == "Higher-Ticket Margin":
        return (0.80 + 0.20 * normalized_positive(df["vehicle_margin"]) + 0.20 * normalized_positive(df["conservative_resale_price"])).clip(0.8, 1.25)
    if vehicle_preset == "Conservative Risk":
        no_flags = (~df["quality_issue"]).astype(float)
        return (
            0.70
            + 0.20 * no_flags
            + 0.20 * df["top_price_probability"].fillna(0)
            + 0.10 * normalized_inverse(df["age_years"])
            + 0.10 * normalized_inverse(df["mileage_km"])
        ).clip(0.7, 1.25)
    return pd.Series(1.0, index=df.index)


def apply_filters(df: pd.DataFrame, vehicle_params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    total_universe = len(df)
    filtered = df.copy()
    missing_model_mask = filtered["expected_price_eur"].isna() | filtered["top_price_probability"].isna()

    if not vehicle_params["allow_missing_model_outputs"]:
        filtered = filtered.loc[~missing_model_mask].copy()
    after_missing_output_filter = len(filtered)

    filtered = filtered.loc[
        filtered["age_years"].between(*vehicle_params["age_range"], inclusive="both")
        & filtered["mileage_km"].between(*vehicle_params["mileage_range"], inclusive="both")
        & filtered["power_hp"].between(*vehicle_params["power_range"], inclusive="both")
        & filtered["actual_price_eur"].between(*vehicle_params["price_range"], inclusive="both")
        & filtered["fuel_type"].isin(vehicle_params["fuel_types"])
        & (filtered["top_price_probability"].fillna(0) >= vehicle_params["min_top_price_probability"])
        & (filtered["expected_discount_pct"].fillna(-999) >= vehicle_params["min_expected_discount_pct"])
    ].copy()
    after_vehicle_filters = len(filtered)

    quality_exclusion_mask = pd.Series(False, index=filtered.index)
    if vehicle_params["exclude_price_outlier"]:
        quality_exclusion_mask |= filtered["price_outlier_iqr"]
    if vehicle_params["exclude_mileage_outlier"]:
        quality_exclusion_mask |= filtered["mileage_outlier_iqr"]
    if vehicle_params["exclude_power_outlier"]:
        quality_exclusion_mask |= filtered["power_outlier_iqr"]
    if vehicle_params["exclude_logical_issue"]:
        quality_exclusion_mask |= filtered["logical_issue"]
    excluded_quality = int(quality_exclusion_mask.sum())
    filtered = filtered.loc[~quality_exclusion_mask].copy()

    counts = {
        "total_universe": total_universe,
        "eligible_after_filters": len(filtered),
        "excluded_by_filters": after_missing_output_filter - after_vehicle_filters,
        "excluded_by_quality_flags": excluded_quality,
        "excluded_by_missing_model_outputs": int(missing_model_mask.sum()) if not vehicle_params["allow_missing_model_outputs"] else 0,
    }
    return filtered, counts


def select_portfolio(df: pd.DataFrame, budget_params: dict[str, Any]) -> pd.DataFrame:
    budget_limit = budget_params["budget_eur"] * (1 - budget_params["cash_buffer_pct"])
    candidates = df.loc[(df["expected_total_profit"] > 0) & (df["investment_score"] > 0)].copy()
    candidates = candidates.sort_values("investment_score", ascending=False)

    selected_rows = []
    deployed = 0.0
    for _, row in candidates.iterrows():
        if len(selected_rows) >= budget_params["max_vehicles"]:
            break
        capital = float(row["capital_deployed"])
        if capital <= 0 or deployed + capital > budget_limit:
            continue
        selected_rows.append(row)
        deployed += capital

    if not selected_rows:
        selected = candidates.iloc[0:0].copy()
    else:
        selected = pd.DataFrame(selected_rows)
    selected["selected"] = True
    return selected


def assign_recommended_actions(full_df: pd.DataFrame, selected_ids: set[Any]) -> pd.DataFrame:
    result = full_df.copy()
    result["selected"] = result["listing_id"].isin(selected_ids)
    result["recommended_action"] = "Do not prioritize"
    result.loc[
        result["selected"] & (result["expected_total_profit"] > 0) & (~result["quality_issue"]),
        "recommended_action",
    ] = "Buy candidate"
    result.loc[result["selected"] & result["quality_issue"], "recommended_action"] = "Manual review"
    return result


def summarize_portfolio(selected: pd.DataFrame, budget_params: dict[str, Any]) -> dict[str, float]:
    if selected.empty:
        return {
            "capital_deployed": 0.0,
            "remaining_budget": float(budget_params["budget_eur"]),
            "selected_count": 0,
            "expected_total_profit": 0.0,
            "expected_roi": 0.0,
            "vehicle_margin": 0.0,
            "gross_resale_spread": 0.0,
            "finance_margin": 0.0,
            "cross_sell_income": 0.0,
            "insurance_income": 0.0,
            "fuel_card_income": 0.0,
            "payroll_income": 0.0,
            "inventory_funding_cost": 0.0,
            "reconditioning_cost": 0.0,
            "transaction_cost": 0.0,
            "avg_top_price_probability": 0.0,
            "avg_expected_discount_pct": 0.0,
            "avg_expected_profit_per_vehicle": 0.0,
        }

    capital = selected["capital_deployed"].sum()
    total_profit = selected["expected_total_profit"].sum()
    return {
        "capital_deployed": float(capital),
        "remaining_budget": float(max(budget_params["budget_eur"] - capital, 0)),
        "selected_count": int(len(selected)),
        "expected_total_profit": float(total_profit),
        "expected_roi": float(total_profit / capital) if capital else 0.0,
        "vehicle_margin": float(selected["vehicle_margin"].sum()),
        "gross_resale_spread": float(selected["gross_resale_spread"].sum()),
        "finance_margin": float(selected["finance_margin"].sum()),
        "cross_sell_income": float(selected["cross_sell_income"].sum()),
        "insurance_income": float(selected["insurance_income"].sum()),
        "fuel_card_income": float(selected["fuel_card_income"].sum()),
        "payroll_income": float(selected["payroll_income"].sum()),
        "inventory_funding_cost": float(selected["inventory_funding_cost"].sum()),
        "reconditioning_cost": float(selected["reconditioning_cost"].sum()),
        "transaction_cost": float(selected["transaction_cost"].sum()),
        "avg_top_price_probability": float(selected["top_price_probability"].mean()),
        "avg_expected_discount_pct": float(selected["expected_discount_pct"].mean()),
        "avg_expected_profit_per_vehicle": float(selected["expected_total_profit"].mean()),
    }


def portfolio_warnings(
    df: pd.DataFrame,
    eligible: pd.DataFrame,
    selected: pd.DataFrame,
    summary: dict[str, float],
    budget_params: dict[str, Any],
) -> list[str]:
    warnings = []
    if eligible.empty:
        warnings.append("No vehicles meet the current strategy criteria.")
    elif len(eligible) < max(10, len(df) * 0.03):
        warnings.append("The eligible universe is small. The mandate may be too narrow for a robust committee decision.")
    if selected.empty:
        warnings.append("No vehicles were selected because no positive business candidates fit the budget and assumptions.")
    budget_used = summary["capital_deployed"] / budget_params["budget_eur"] if budget_params["budget_eur"] else 0
    if selected.shape[0] > 0 and budget_used < 0.50:
        warnings.append("The selected portfolio uses less than half of the available budget.")
    if selected.shape[0] > 0 and selected["fuel_type"].value_counts(normalize=True).iloc[0] > 0.70:
        warnings.append("The selected portfolio is concentrated in one fuel type.")
    if selected.shape[0] > 0:
        price_bands = pd.cut(selected["actual_price_eur"], bins=4, duplicates="drop")
        if price_bands.value_counts(normalize=True).iloc[0] > 0.70:
            warnings.append("The selected portfolio is concentrated in one purchase price band.")
    if summary["expected_total_profit"] > 0 and summary["cross_sell_income"] / summary["expected_total_profit"] > 0.50:
        warnings.append("Expected ROI is strongly influenced by cross-sell and loyalty assumptions.")
    return warnings


def highest_price_band_concentration(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.0
    price_bands = pd.cut(selected["actual_price_eur"], bins=4, duplicates="drop")
    return float(price_bands.value_counts(normalize=True).iloc[0]) if len(price_bands) else 0.0


def risk_confidence_metrics(
    full_df: pd.DataFrame,
    eligible: pd.DataFrame,
    selected: pd.DataFrame,
    counts: dict[str, int],
    summary: dict[str, float],
    budget_params: dict[str, Any],
) -> dict[str, float]:
    model_output_coverage = (
        full_df["expected_price_eur"].notna() & full_df["top_price_probability"].notna()
    ).mean()
    eligible_quality_share = float(eligible["quality_issue"].mean()) if not eligible.empty else 0.0
    selected_quality_share = float(selected["quality_issue"].mean()) if not selected.empty else 0.0
    fuel_concentration = (
        float(selected["fuel_type"].value_counts(normalize=True).iloc[0]) if not selected.empty else 0.0
    )
    price_concentration = highest_price_band_concentration(selected)
    cross_sell_dependency = (
        summary["cross_sell_income"] / summary["expected_total_profit"]
        if summary["expected_total_profit"] > 0
        else 0.0
    )
    budget_used = summary["capital_deployed"] / budget_params["budget_eur"] if budget_params["budget_eur"] else 0.0
    return {
        "model_output_coverage": float(model_output_coverage),
        "missing_model_outputs_excluded": counts["excluded_by_missing_model_outputs"],
        "eligible_quality_share": eligible_quality_share,
        "selected_quality_share": selected_quality_share,
        "fuel_concentration": fuel_concentration,
        "price_band_concentration": price_concentration,
        "cross_sell_dependency": float(cross_sell_dependency),
        "budget_used": float(budget_used),
    }


def risk_confidence_warnings(metrics: dict[str, float]) -> list[str]:
    warnings = []
    if metrics["model_output_coverage"] < 0.90:
        warnings.append("Model output coverage is below 90%.")
    if metrics["selected_quality_share"] > 0:
        warnings.append("The selected portfolio includes vehicles with quality flags.")
    if metrics["fuel_concentration"] > 0.70:
        warnings.append("Fuel type concentration is above 70%.")
    if metrics["price_band_concentration"] > 0.70:
        warnings.append("Price band concentration is above 70%.")
    if metrics["cross_sell_dependency"] > 0.50:
        warnings.append("Cross-sell dependency is above 50% of expected profit.")
    if metrics["budget_used"] < 0.50:
        warnings.append("The selected portfolio uses less than 50% of the available budget.")
    return warnings


def committee_decision(summary: dict[str, float], risk_warnings: list[str]) -> tuple[str, list[str], list[str]]:
    if summary["selected_count"] == 0 or summary["expected_roi"] < 0.05:
        status = "Do not proceed"
    elif summary["expected_roi"] >= 0.12 and not risk_warnings:
        status = "Recommend"
    else:
        status = "Review"

    reasons = [
        f"Selected {summary['selected_count']:,} vehicles under the current mandate.",
        f"Expected portfolio ROI is {fmt_pct(summary['expected_roi'])}.",
        f"Expected total profit is {fmt_eur(summary['expected_total_profit'])}.",
    ]
    risks = (risk_warnings + [
        "Validate resale timing, reconditioning capacity, and final purchase conditions.",
        "Confirm that commercial attach-rate assumptions are realistic for this campaign.",
        "Review individual vehicle quality flags before any acquisition decision.",
    ])[:3]
    return status, reasons[:3], risks


def render_committee_decision(summary: dict[str, float], risk_warnings: list[str]) -> None:
    status, reasons, risks = committee_decision(summary, risk_warnings)
    if status == "Recommend":
        st.success(f"**Decision status: {status}**")
    elif status == "Review":
        st.warning(f"**Decision status: {status}**")
    else:
        st.error(f"**Decision status: {status}**")

    reason_col, risk_col = st.columns(2)
    with reason_col:
        st.markdown("**Reasons**")
        for reason in reasons:
            st.markdown(f"- {reason}")
    with risk_col:
        st.markdown("**Risks or validation points**")
        for risk in risks:
            st.markdown(f"- {risk}")

    st.caption(
        "The decision status is a simulation output for committee discussion. It is not a final acquisition, "
        "credit, compliance, or risk approval."
    )


def main_risk_driver(metrics: dict[str, float]) -> tuple[str, str]:
    if metrics["model_output_coverage"] < 0.90:
        return "Low model output coverage", "Model signals are missing for a material share of the vehicle universe."
    if metrics["selected_quality_share"] > 0:
        return "Selected vehicle quality flags", "Some selected vehicles require manual quality review before action."
    if metrics["fuel_concentration"] > 0.70:
        return "Fuel type concentration", "The selected portfolio is concentrated in one fuel type."
    if metrics["price_band_concentration"] > 0.70:
        return "Price band concentration", "The selected portfolio is concentrated in one purchase price band."
    if metrics["cross_sell_dependency"] > 0.50:
        return (
            "Cross-sell dependency",
            "Cross-sell dependency means expected profit relies materially on commercial attach-rate assumptions.",
        )
    if metrics["budget_used"] < 0.50:
        return (
            "Low budget deployment",
            "Low budget deployment means the current mandate may be too restrictive to absorb available capital.",
        )
    return "No major risk driver detected", "The current risk indicators do not show a dominant single concern."


def render_risk_confidence(metrics: dict[str, float], warnings: list[str]) -> None:
    row1 = st.columns(3)
    row1[0].metric("Model output coverage", fmt_pct(metrics["model_output_coverage"]))
    row1[1].metric("Missing model outputs excluded", f"{metrics['missing_model_outputs_excluded']:,}")
    row1[2].metric("Eligible vehicles with quality flags", fmt_pct(metrics["eligible_quality_share"]))

    row2 = st.columns(4)
    row2[0].metric("Selected vehicles with quality flags", fmt_pct(metrics["selected_quality_share"]))
    row2[1].metric("Highest fuel concentration", fmt_pct(metrics["fuel_concentration"]))
    row2[2].metric("Highest price band concentration", fmt_pct(metrics["price_band_concentration"]))
    row2[3].metric("Cross-sell dependency ratio", fmt_pct(metrics["cross_sell_dependency"]))

    driver, explanation = main_risk_driver(metrics)
    st.markdown(f"**Main risk driver:** {driver}. {explanation}")

    for warning in warnings:
        st.warning(warning)


def strategy_interpretation(vehicle_preset: str, pricing_preset: str) -> str:
    if vehicle_preset == "Broad Market":
        priority = "volume and opportunity discovery"
    elif vehicle_preset == "Young Low-Mileage Core":
        priority = "portfolio cleanliness and resale simplicity"
    elif vehicle_preset == "Mainstream Retail Campaign":
        priority = "campaign coherence and easy-to-explain retail selection"
    elif vehicle_preset == "Higher-Ticket Margin":
        priority = "unit margin with capital concentration"
    else:
        priority = "risk control and defensibility"

    if pricing_preset == "Customer Loyalty Focus":
        priority += " with customer loyalty economics"
    elif pricing_preset == "Aggressive Cross-Sell":
        priority += " with commercial bundling"
    elif pricing_preset == "Stress Case":
        priority += " under adverse assumptions"
    return f"This strategy prioritizes {priority}."


def deterministic_recommendation(summary: dict[str, float], warnings: list[str]) -> str:
    roi = summary["expected_roi"]
    if summary["selected_count"] == 0:
        status = "unattractive"
        reason = "no vehicles meet the current investment and profitability criteria"
    elif roi >= 0.12:
        status = "attractive"
        reason = "expected portfolio ROI is above a strong committee threshold"
    elif roi >= 0.05:
        status = "borderline"
        reason = "expected ROI is positive but requires careful validation"
    else:
        status = "unattractive"
        reason = "expected ROI is too low for the current assumptions"

    drivers = {
        "vehicle resale margin": summary["vehicle_margin"],
        "financing margin": summary["finance_margin"],
        "cross-sell and loyalty income": summary["cross_sell_income"],
    }
    driver = max(drivers, key=drivers.get)
    caution = warnings[0] if warnings else "Validate pricing, resale timing, and operational capacity before approval."
    return (
        f"The current strategy is **{status}** because {reason}. "
        f"The main value driver is **{driver}**. Main caution: {caution}"
    )


def fmt_eur(value: float) -> str:
    return f"EUR {value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def render_kpis(summary: dict[str, float], budget_params: dict[str, Any]) -> None:
    row1 = st.columns(4)
    row1[0].metric("Investment budget", fmt_eur(budget_params["budget_eur"]))
    row1[1].metric("Capital deployed", fmt_eur(summary["capital_deployed"]))
    row1[2].metric("Remaining budget", fmt_eur(summary["remaining_budget"]))
    row1[3].metric("Selected vehicles", f"{summary['selected_count']:,}")

    row2 = st.columns(4)
    row2[0].metric("Expected total profit", fmt_eur(summary["expected_total_profit"]))
    row2[1].metric("Expected portfolio ROI", fmt_pct(summary["expected_roi"]))
    row2[2].metric("Vehicle resale margin", fmt_eur(summary["vehicle_margin"]))
    row2[3].metric("Financing margin", fmt_eur(summary["finance_margin"]))

    row3 = st.columns(4)
    row3[0].metric("Cross-sell and loyalty income", fmt_eur(summary["cross_sell_income"]))
    row3[1].metric(
        "Average commercial attractiveness score",
        fmt_pct(summary["avg_top_price_probability"]),
        help="This score comes from the upstream classification model field `top_price_probability`.",
    )
    row3[2].metric("Average expected discount %", fmt_pct(summary["avg_expected_discount_pct"]))
    row3[3].metric("Average profit per vehicle", fmt_eur(summary["avg_expected_profit_per_vehicle"]))


def strategy_overview_markdown(
    vehicle_preset: str,
    pricing_preset: str,
    vehicle_params: dict[str, Any],
    pricing_params: dict[str, Any],
    metadata: dict[str, Any],
    customized: bool,
) -> str:
    return f"""
**Selected vehicle strategy:** {vehicle_preset}  
**Selected pricing and cross-sell strategy:** {pricing_preset}  
**Manual customization:** {"Yes" if customized else "No"}

**Eligible vehicle mandate**
- Age range: {vehicle_params["age_range"][0]:,.1f} to {vehicle_params["age_range"][1]:,.1f} years
- Mileage range: {vehicle_params["mileage_range"][0]:,.0f} to {vehicle_params["mileage_range"][1]:,.0f} km
- Fuel types: {", ".join(vehicle_params["fuel_types"]) if vehicle_params["fuel_types"] else "None selected"}
- Purchase price range: {fmt_eur(vehicle_params["price_range"][0])} to {fmt_eur(vehicle_params["price_range"][1])}
- Minimum commercial attractiveness score: {fmt_pct(vehicle_params["min_top_price_probability"])}
- Minimum expected discount: {fmt_pct(vehicle_params["min_expected_discount_pct"])}

**Commercial strategy**
- Resale haircut: {fmt_pct(pricing_params["resale_haircut"])}
- Financing take-up: {fmt_pct(pricing_params["financing_take_up_rate"])}
- Expected effective APR after weighted cross-sell discounts: {fmt_pct(metadata["effective_customer_apr"])}
- Expected weighted APR discount: {fmt_pct(metadata["expected_total_apr_discount"])}
- Product APR discounts: insurance {fmt_pct(pricing_params["insurance_apr_discount"])}, fuel card {fmt_pct(pricing_params["fuel_card_apr_discount"])}, payroll transfer {fmt_pct(pricing_params["payroll_apr_discount"])}
- Insurance attach rate: {fmt_pct(pricing_params["insurance_attach_rate"])} base -> {fmt_pct(metadata["adjusted_insurance_attach_rate"])} expected after discount effect
- Fuel card attach rate: {fmt_pct(pricing_params["fuel_card_attach_rate"])} base -> {fmt_pct(metadata["adjusted_fuel_card_attach_rate"])} expected after discount effect
- Payroll transfer attach rate: {fmt_pct(pricing_params["payroll_attach_rate"])} base -> {fmt_pct(metadata["adjusted_payroll_attach_rate"])} expected after discount effect

{strategy_interpretation(vehicle_preset, pricing_preset)}
"""


def component_chart(summary: dict[str, float]) -> None:
    components = pd.DataFrame(
        {
            "Component": [
                "Gross resale spread",
                "Reconditioning cost",
                "Transaction cost",
                "Financing margin",
                "Insurance income",
                "Fuel card income",
                "Payroll income",
                "Inventory funding cost",
            ],
            "Amount": [
                summary["gross_resale_spread"],
                -summary["reconditioning_cost"],
                -summary["transaction_cost"],
                summary["finance_margin"],
                summary["insurance_income"],
                summary["fuel_card_income"],
                summary["payroll_income"],
                -summary["inventory_funding_cost"],
            ],
        }
    )
    fig = px.bar(components, x="Component", y="Amount", title="Expected profit components")
    fig.update_layout(xaxis_title="", yaxis_title="EUR", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def selected_table(selected: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "listing_id",
        "make",
        "model",
        "fuel_type",
        "actual_price_eur",
        "expected_price_eur",
        "expected_discount_pct",
        "top_price_probability",
        "vehicle_margin",
        "finance_margin",
        "insurance_income",
        "fuel_card_income",
        "payroll_income",
        "cross_sell_income",
        "expected_total_profit",
        "expected_roi",
        "investment_score",
        "recommended_action",
        "price_outlier_iqr",
        "mileage_outlier_iqr",
        "power_outlier_iqr",
        "logical_issue",
    ]
    table = selected[[column for column in columns if column in selected.columns]].copy()
    return table.rename(
        columns={
            "listing_id": "Listing ID",
            "make": "Make",
            "model": "Model",
            "fuel_type": "Fuel type",
            "actual_price_eur": "Purchase price",
            "expected_price_eur": "Expected market price",
            "expected_discount_pct": "Expected discount %",
            "top_price_probability": "Commercial attractiveness score",
            "vehicle_margin": "Vehicle margin",
            "finance_margin": "Financing margin",
            "insurance_income": "Insurance income",
            "fuel_card_income": "Fuel card income",
            "payroll_income": "Payroll income",
            "cross_sell_income": "Cross-sell income",
            "expected_total_profit": "Expected profit",
            "expected_roi": "Expected ROI",
            "investment_score": "Investment score",
            "recommended_action": "Recommended action",
            "price_outlier_iqr": "Price outlier flag",
            "mileage_outlier_iqr": "Mileage outlier flag",
            "power_outlier_iqr": "Power outlier flag",
            "logical_issue": "Logical issue flag",
        }
    )


def scatter_chart(eligible_with_actions: pd.DataFrame) -> None:
    if eligible_with_actions.empty:
        st.info("No eligible vehicles are available for the current strategy.")
        return
    plot_df = eligible_with_actions.copy()
    plot_df["selection_status"] = np.where(plot_df["selected"], "Selected", "Not selected")
    fig = px.scatter(
        plot_df,
        x="expected_discount_pct",
        y="top_price_probability",
        size="actual_price_eur",
        color="selection_status",
        hover_data=[
            "listing_id",
            "make",
            "model",
            "fuel_type",
            "actual_price_eur",
            "expected_price_eur",
            "expected_discount_pct",
            "top_price_probability",
            "expected_total_profit",
            "expected_roi",
        ],
        title="Opportunity map: expected discount vs commercial attractiveness score",
        labels={
            "top_price_probability": "Commercial attractiveness score",
            "expected_discount_pct": "Expected discount %",
            "actual_price_eur": "Actual price EUR",
            "expected_price_eur": "Expected price EUR",
            "expected_total_profit": "Expected total profit",
            "expected_roi": "Expected ROI",
        },
    )
    fig.update_layout(xaxis_title="Expected discount %", yaxis_title="Commercial attractiveness score")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def build_memo_prompt(
    vehicle_preset: str,
    pricing_preset: str,
    overview: str,
    summary: dict[str, float],
    pricing_params: dict[str, Any],
    warnings: list[str],
    selected: pd.DataFrame,
) -> str:
    top_candidates = selected_table(selected).head(10).to_dict(orient="records")
    return f"""
Write a concise committee-ready memo in English for a consumer finance business.

Required sections:
1. Selected Strategy and Investment Mandate
2. Executive Recommendation
3. Portfolio Rationale
4. Key Assumptions
5. Expected Financial Result
6. Cross-Sell and Loyalty Logic
7. Main Risks and Model Limitations
8. Suggested Next Validation Steps

Selected vehicle strategy: {vehicle_preset}
Selected pricing and cross-sell strategy: {pricing_preset}

Selected strategy overview:
{overview}

Aggregated metrics:
{summary}

Pricing and cross-sell assumptions:
{pricing_params}

Warnings:
{warnings}

Top 10 selected candidates:
{top_candidates}

The memo must explain the strategy that produced the portfolio, not simply summarize selected cars.
Use "commercial attractiveness score" when referring to the classification model signal from `top_price_probability`.
Include this exact warning:
"This is a simulated decision-support tool. It is not a final acquisition, credit, compliance, or risk approval."

Keep the memo concise, executive, and non-technical.
"""


def call_gemini(prompt: str) -> str:
    text, error = generate_gemini_content(prompt)
    if error:
        return f"{error} Deterministic/default dashboard behavior is still available."
    return text or "Gemini returned an empty response. Deterministic/default dashboard behavior is still available."


def allowed_bi_view(project_id: str, dataset_id: str) -> str:
    return ALLOWED_BI_VIEW_TEMPLATE.format(project_id=project_id, dataset_id=dataset_id)


def build_demo_queries(project_id: str, dataset_id: str) -> dict[str, dict[str, Any]]:
    view = allowed_bi_view(project_id, dataset_id)
    return {
        "Which listings should the committee review first?": {
            "sql": f"""
SELECT
  listing_id,
  make,
  model,
  fuel_type,
  actual_price_eur,
  expected_price_eur,
  expected_price_gap_eur,
  top_price_probability,
  decision_flag,
  price_outlier_iqr,
  mileage_outlier_iqr,
  power_outlier_iqr,
  logical_issue
FROM {view}
WHERE decision_flag = 'high_priority_review'
  AND expected_price_gap_eur < 0
ORDER BY top_price_probability DESC, expected_price_gap_eur ASC
LIMIT 25
""",
            "business_intent": "Rank the first listings the investment committee should review.",
            "expected_output": "Listing-level priority queue with model signals and quality flags.",
            "chart_type": "table",
            "confidence": "high",
            "limitations": ["Demo SQL uses the governed BI view and does not include local portfolio simulation economics."],
        },
        "Which fuel types concentrate the strongest price opportunities?": {
            "sql": f"""
SELECT
  fuel_type,
  COUNT(*) AS listing_count,
  AVG(expected_price_gap_eur) AS avg_expected_price_gap_eur,
  AVG(top_price_probability) AS avg_top_price_probability,
  COUNTIF(decision_flag = 'high_priority_review') AS high_priority_count
FROM {view}
WHERE expected_price_gap_eur < 0
GROUP BY fuel_type
ORDER BY high_priority_count DESC, avg_expected_price_gap_eur ASC
LIMIT 20
""",
            "business_intent": "Identify fuel categories where below-expected-price opportunities concentrate.",
            "expected_output": "Fuel-type aggregation by opportunity count, gap and top-price signal.",
            "chart_type": "bar",
            "confidence": "high",
            "limitations": ["Averages can hide listing-level quality issues."],
        },
        "Where do regression and classification signals agree?": {
            "sql": f"""
SELECT
  decision_flag,
  COUNT(*) AS listing_count,
  AVG(expected_price_gap_eur) AS avg_expected_price_gap_eur,
  AVG(top_price_probability) AS avg_top_price_probability
FROM {view}
WHERE expected_price_gap_eur IS NOT NULL
  AND top_price_probability IS NOT NULL
GROUP BY decision_flag
ORDER BY avg_top_price_probability DESC, avg_expected_price_gap_eur ASC
LIMIT 20
""",
            "business_intent": "Compare regression price-gap signals with classification top-price probability.",
            "expected_output": "Decision-flag groups showing both model signals side by side.",
            "chart_type": "scatter",
            "confidence": "high",
            "limitations": ["Agreement is interpreted from aggregate averages, not a statistical test."],
        },
        "Which opportunities have quality or logical risk flags?": {
            "sql": f"""
SELECT
  listing_id,
  make,
  model,
  fuel_type,
  actual_price_eur,
  expected_price_gap_eur,
  top_price_probability,
  decision_flag,
  price_outlier_iqr,
  mileage_outlier_iqr,
  power_outlier_iqr,
  logical_issue
FROM {view}
WHERE expected_price_gap_eur < 0
  AND (
    price_outlier_iqr = TRUE
    OR mileage_outlier_iqr = TRUE
    OR power_outlier_iqr = TRUE
    OR logical_issue = TRUE
  )
ORDER BY top_price_probability DESC, expected_price_gap_eur ASC
LIMIT 50
""",
            "business_intent": "Surface promising listings that still need quality or logic-risk review.",
            "expected_output": "Listing-level opportunity queue with risk flags visible.",
            "chart_type": "table",
            "confidence": "high",
            "limitations": ["Quality flags explain risk, not final acquisition rejection."],
        },
        "How many listings fall into each decision flag?": {
            "sql": f"""
SELECT
  decision_flag,
  COUNT(*) AS listing_count,
  AVG(actual_price_eur) AS avg_actual_price_eur,
  AVG(expected_price_gap_eur) AS avg_expected_price_gap_eur,
  AVG(top_price_probability) AS avg_top_price_probability
FROM {view}
GROUP BY decision_flag
ORDER BY listing_count DESC
LIMIT 20
""",
            "business_intent": "Summarize the portfolio by governed BI decision flag.",
            "expected_output": "Decision-flag distribution with average price and model signals.",
            "chart_type": "bar",
            "confidence": "high",
            "limitations": ["This summarizes BI-view signals only, before local Streamlit strategy filters."],
        },
    }


def build_semantic_layer(project_id: str, dataset_id: str) -> str:
    field_lines = []
    for column in REQUIRED_COLUMNS:
        meaning = BI_FIELD_MEANINGS.get(column, "allowed BI view field")
        field_lines.append(f"- {column}: {meaning}")
    flag_lines = [f"- {flag}: {meaning}" for flag, meaning in DECISION_FLAG_MEANINGS.items()]
    return f"""
Governed BI query surface:
Allowed table: {allowed_bi_view(project_id, dataset_id)}

Allowed fields:
{chr(10).join(field_lines)}

Decision flag meanings:
{chr(10).join(flag_lines)}

Quality and risk flags should not be ignored when making recommendations:
price_outlier_iqr, mileage_outlier_iqr, power_outlier_iqr, logical_issue.
"""


def build_sql_generation_prompt(question: str, project_id: str, dataset_id: str) -> str:
    return f"""
You are generating BigQuery Standard SQL for a read-only BI assistant.
This is a conversational decision-support layer over a governed BI data product.
The LLM writes candidate SQL. The application owns validation. BigQuery owns the data. The user owns the decision.

{build_semantic_layer(project_id, dataset_id)}

Return JSON only with this schema:
{{
  "sql": "SELECT ...",
  "business_intent": "short explanation of the user's business question",
  "expected_output": "what the query is expected to return",
  "chart_type": "table|bar|scatter|line|none",
  "confidence": "high|medium|low",
  "limitations": ["limitation 1", "limitation 2"]
}}

Rules:
- Return JSON only.
- Generate one SQL statement only.
- Generate SELECT queries only.
- Never generate INSERT, UPDATE, DELETE, MERGE, CREATE, DROP, ALTER, TRUNCATE, GRANT, REVOKE, CALL or EXECUTE.
- Never query INFORMATION_SCHEMA.
- Never include SQL comments.
- Never use SELECT *.
- Never invent columns.
- Use only the columns listed in the semantic layer.
- Prefer aggregation for broad questions.
- For listing-level results, always include LIMIT 100 or less.
- Use BigQuery-safe snake_case aliases only. Do not use spaces, punctuation, parentheses, or backticks in aliases.
- For "best opportunities", prioritize decision_flag = 'high_priority_review', expected_price_gap_eur < 0,
  high top_price_probability, and no logical_issue when relevant.
- If the question is ambiguous, make a conservative assumption and state it in the JSON limitations.
- Do not answer with SQL that requires local Streamlit-calculated fields such as expected_total_profit,
  expected_roi, vehicle_margin, finance_margin, cross_sell_income or investment_score.

User question:
{question}
"""


def parse_gemini_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not text or not text.strip():
        return None, "Gemini returned an empty response."

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start == -1:
        return None, "Gemini response did not contain a JSON object."

    depth = 0
    in_string = False
    escape = False
    end = -1
    for position, char in enumerate(cleaned[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = position + 1
                break

    if end == -1:
        return None, "Gemini response contained incomplete JSON."

    try:
        parsed = json.loads(cleaned[start:end])
    except json.JSONDecodeError as exc:
        return None, f"Gemini JSON could not be parsed: {exc.msg}."
    if not isinstance(parsed, dict):
        return None, "Gemini JSON must be an object."
    return parsed, None


def generate_sql_with_gemini(question: str, project_id: str, dataset_id: str) -> tuple[dict[str, Any] | None, str | None]:
    text, error = generate_gemini_content(build_sql_generation_prompt(question, project_id, dataset_id))
    if error:
        return None, error
    return parse_gemini_json(text or "")


def strip_sql_comments(sql: str) -> tuple[str, bool]:
    cleaned: list[str] = []
    index = 0
    in_single_quote = False
    in_double_quote = False
    removed = False

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            cleaned.append(char)
            index += 1
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            cleaned.append(char)
            index += 1
            continue

        if not in_single_quote and not in_double_quote and char == "-" and next_char == "-":
            removed = True
            index += 2
            while index < len(sql) and sql[index] not in "\r\n":
                index += 1
            continue

        if not in_single_quote and not in_double_quote and char == "/" and next_char == "*":
            removed = True
            index += 2
            while index + 1 < len(sql) and not (sql[index] == "*" and sql[index + 1] == "/"):
                index += 1
            index += 2
            continue

        cleaned.append(char)
        index += 1

    normalized = "\n".join(line.rstrip() for line in "".join(cleaned).splitlines() if line.strip())
    return normalized, removed


def safe_bigquery_alias(alias: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", alias.strip()).strip("_").lower()
    if not cleaned:
        cleaned = "field_alias"
    if cleaned[0].isdigit():
        cleaned = f"field_{cleaned}"
    return cleaned[:300]


def sanitize_sql_aliases(sql: str) -> tuple[str, bool]:
    changed = False

    def replace_alias(match: re.Match) -> str:
        nonlocal changed
        alias = match.group(1)
        safe_alias = safe_bigquery_alias(alias)
        if alias != safe_alias:
            changed = True
        return f"AS {safe_alias}"

    sanitized = re.sub(r"(?is)\bAS\s+`([^`]+)`", replace_alias, sql)
    return sanitized, changed


def validate_sql(sql: str, project_id: str, dataset_id: str, chart_type: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    normalized_sql = (sql or "").strip()

    if not normalized_sql:
        return {"is_valid": False, "errors": ["SQL is empty."], "warnings": warnings, "normalized_sql": ""}

    if re.search(r"(--|/\*|\*/)", normalized_sql):
        errors.append("Comments are not allowed in generated SQL.")

    if normalized_sql.count(";") > 1 or ";" in normalized_sql.rstrip(";"):
        errors.append("Multiple SQL statements or semicolon followed by more text are not allowed.")
    normalized_sql = normalized_sql.rstrip().rstrip(";").strip()

    if not re.match(r"(?is)^\s*(SELECT|WITH)\b", normalized_sql):
        errors.append("Only SELECT or WITH queries are allowed.")

    forbidden = r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CALL|EXECUTE)\b"
    if re.search(forbidden, normalized_sql, flags=re.IGNORECASE):
        errors.append("SQL contains a forbidden write, DDL, permission, or execution keyword.")

    if re.search(r"\bINFORMATION_SCHEMA\b", normalized_sql, flags=re.IGNORECASE):
        errors.append("INFORMATION_SCHEMA queries are not allowed.")

    select_star_pattern = (
        r"(?is)(\bSELECT\s+(?:`?[A-Za-z_][A-Za-z0-9_]*`?\.)?\*"
        r"|,\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\.)?\*)"
    )
    if re.search(select_star_pattern, normalized_sql):
        errors.append("SELECT * is not allowed. Select explicit BI fields.")

    allowed_ref = f"{project_id}.{dataset_id}.vw_bi_dashboard"
    allowed_quoted = allowed_bi_view(project_id, dataset_id)

    backtick_refs = re.findall(r"`([^`]+)`", normalized_sql)
    for ref in backtick_refs:
        if "." in ref and ref != allowed_ref:
            errors.append(f"External table reference is not allowed: `{ref}`.")

    cte_names = {
        match.lower()
        for match in re.findall(r"(?is)(?:WITH|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", normalized_sql)
    }
    table_refs = re.findall(r"(?is)\b(?:FROM|JOIN)\s+([`A-Za-z0-9_.-]+)", normalized_sql)
    allowed_unquoted = allowed_ref
    found_allowed_view = False
    for ref in table_refs:
        clean_ref = ref.strip().rstrip(",")
        if clean_ref.startswith("("):
            continue
        if clean_ref == allowed_quoted or clean_ref.strip("`") == allowed_unquoted:
            found_allowed_view = True
            continue
        if clean_ref.lower() in cte_names:
            continue
        errors.append(f"Only the governed BI view can be queried. Unexpected table reference: {clean_ref}.")

    if not found_allowed_view:
        errors.append("Query must read from the governed BI dashboard view.")

    has_group_by = bool(re.search(r"(?is)\bGROUP\s+BY\b", normalized_sql))
    has_limit = bool(re.search(r"(?is)\bLIMIT\s+\d+\b", normalized_sql))
    if not has_limit and not has_group_by:
        warnings.append("Listing-level query had no LIMIT. LIMIT 100 was appended.")
        normalized_sql = f"{normalized_sql}\nLIMIT 100"
    elif has_limit:
        limits = [int(value) for value in re.findall(r"(?is)\bLIMIT\s+(\d+)\b", normalized_sql)]
        if limits and max(limits) > 100 and not has_group_by:
            errors.append("Listing-level queries must use LIMIT 100 or less.")

    if chart_type == "scatter":
        warnings.append("Scatter charts require at least two numeric result columns.")
    elif chart_type in {"bar", "line"}:
        warnings.append(f"{chart_type.title()} charts require a compatible category/date column and numeric value.")

    return {"is_valid": not errors, "errors": errors, "warnings": warnings, "normalized_sql": normalized_sql}


def bigquery_query_error_message(exc: Exception, stage: str) -> str:
    detail = getattr(exc, "message", "") or str(exc)
    if "bytesBilledLimitExceeded" in detail or "bytes billed" in detail.lower():
        return (
            f"{stage} was blocked by the BigQuery bytes-billed cap. "
            f"The current cap is {MAX_BYTES_BILLED / (1024 * 1024):.0f} MB. "
            "Set `BQ_MAX_BYTES_BILLED_MB=100` before launching Streamlit if you want to allow this classroom query."
        )
    return f"{stage} failed because BigQuery rejected the SQL: {detail}"


def run_validated_query(sql: str, project_id: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "estimated_bytes": None,
        "estimated_mb": None,
        "errors": [],
        "warnings": [],
        "executed": False,
    }
    try:
        client = bigquery.Client(project=project_id)
    except DefaultCredentialsError:
        metadata["errors"].append(
            "BigQuery credentials are not configured. Run `gcloud auth application-default login` or configure service account credentials."
        )
        return None, metadata
    except Exception:
        metadata["errors"].append("BigQuery client could not be created. Check Google Cloud credentials and project access.")
        return None, metadata

    dry_run_config = bigquery.QueryJobConfig(
        dry_run=True,
        maximum_bytes_billed=MAX_BYTES_BILLED,
        use_query_cache=False,
    )
    try:
        dry_run_job = client.query(sql, job_config=dry_run_config)
        estimated_bytes = int(dry_run_job.total_bytes_processed or 0)
        metadata["estimated_bytes"] = estimated_bytes
        metadata["estimated_mb"] = estimated_bytes / (1024 * 1024)
    except BadRequest as exc:
        metadata["errors"].append(bigquery_query_error_message(exc, "Dry-run"))
        return None, metadata
    except Forbidden:
        metadata["errors"].append("Dry-run failed because the current credentials do not have BigQuery permission.")
        return None, metadata
    except GoogleAPIError as exc:
        metadata["errors"].append(bigquery_query_error_message(exc, "Dry-run"))
        return None, metadata
    except Exception:
        metadata["errors"].append("Dry-run failed. Check the generated SQL and BigQuery configuration.")
        return None, metadata

    if metadata["estimated_bytes"] is not None and metadata["estimated_bytes"] > MAX_BYTES_BILLED:
        metadata["errors"].append(
            f"Query estimate is {metadata['estimated_mb']:.2f} MB, above the classroom limit of "
            f"{MAX_BYTES_BILLED / (1024 * 1024):.0f} MB."
        )
        return None, metadata

    execution_config = bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES_BILLED)
    try:
        df = client.query(sql, job_config=execution_config).to_dataframe()
        metadata["executed"] = True
        if df.empty:
            metadata["warnings"].append("Query ran successfully but returned no rows.")
        return df, metadata
    except BadRequest as exc:
        metadata["errors"].append(bigquery_query_error_message(exc, "Execution"))
    except Forbidden:
        metadata["errors"].append("Execution failed because the current credentials do not have BigQuery permission.")
    except GoogleAPIError as exc:
        metadata["errors"].append(bigquery_query_error_message(exc, "Execution"))
    except Exception:
        metadata["errors"].append("Execution failed. Check credentials, query permissions, and network access.")
    return None, metadata


def render_ai_chart(df: pd.DataFrame, chart_type: str) -> None:
    if df.empty or chart_type in {"table", "none"}:
        return
    try:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = [
            column for column in df.columns if column not in numeric_columns and not pd.api.types.is_datetime64_any_dtype(df[column])
        ]

        if chart_type == "bar" and categorical_columns and numeric_columns:
            fig = px.bar(df, x=categorical_columns[0], y=numeric_columns[0])
        elif chart_type == "scatter" and len(numeric_columns) >= 2:
            color = categorical_columns[0] if categorical_columns else None
            fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[1], color=color)
        elif chart_type == "line" and numeric_columns:
            x_candidates = [
                column
                for column in df.columns
                if "date" in column.lower() or "year" in column.lower() or pd.api.types.is_datetime64_any_dtype(df[column])
            ]
            if not x_candidates:
                st.info("No date or year-like column was available for the suggested line chart.")
                return
            fig = px.line(df.sort_values(x_candidates[0]), x=x_candidates[0], y=numeric_columns[0])
        else:
            st.info("The suggested chart type is not compatible with the returned columns.")
            return
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("The result table loaded, but chart rendering failed for these columns.")


def deterministic_ai_summary(df: pd.DataFrame, metadata: dict[str, Any]) -> str:
    if df is None or df.empty:
        return "No rows were returned, so there is no business pattern to summarize from this query."

    summary = [f"- Query returned {len(df):,} rows and {len(df.columns):,} columns."]
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        first_numeric = numeric_columns[0]
        top = df.sort_values(first_numeric, ascending=False).head(3)
        values = ", ".join(str(value) for value in top[first_numeric].tolist())
        summary.append(f"- Top observed values for `{first_numeric}` include: {values}.")
    else:
        first_column = df.columns[0]
        values = ", ".join(str(value) for value in df[first_column].head(3).tolist())
        summary.append(f"- First returned `{first_column}` values: {values}.")
    if metadata.get("warnings"):
        summary.append(f"- Caveat: {metadata['warnings'][0]}")
    else:
        summary.append("- Caveat: deterministic fallback summary; no AI interpretation was used.")
    return "\n".join(summary)


def executive_summary(
    question: str,
    ai_payload: dict[str, Any],
    sql: str,
    df: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    if not gemini_available():
        return deterministic_ai_summary(df, metadata)

    records = df.head(20).to_dict(orient="records") if df is not None else []
    prompt = f"""
Write an investment committee interpretation of a governed BI query result.

Rules:
- Maximum 3 bullets.
- Separate facts from interpretation.
- Do not invent numbers.
- Mention limitations.
- Use investment committee language.

Original question: {question}
Business intent: {ai_payload.get("business_intent")}
Generated SQL: {sql}
Dataframe shape: {df.shape if df is not None else None}
First rows as records: {records}
Limitations: {ai_payload.get("limitations", [])}
"""
    result, error = generate_gemini_content(prompt)
    if error:
        return deterministic_ai_summary(df, metadata)
    return result or deterministic_ai_summary(df, metadata)


def render_ai_sql_assistant_tab(project_id: str, dataset_id: str) -> None:
    st.title("Ask the Data — GenAI SQL Assistant")
    st.caption("Natural-language questions translated into validated BigQuery SQL over the BI decision-support view.")
    st.info(
        "This assistant uses Gemini to generate SQL, but BigQuery remains the source of truth. "
        "Every AI-generated query is validated and dry-run checked before execution."
    )
    st.caption("The LLM writes candidate SQL. The application owns validation. BigQuery owns the data. The user owns the decision.")

    demo_mode = not gemini_available()
    demo_queries = build_demo_queries(project_id, dataset_id)
    if demo_mode:
        reason = gemini_unavailable_message() or "Gemini is unavailable."
        st.warning(f"Demo Mode: using predefined SQL. {reason}")
        button_prompts = DEMO_QUESTIONS
    else:
        button_prompts = AI_EXAMPLE_PROMPTS

    st.session_state.setdefault("ai_sql_history", [])
    st.session_state.setdefault("ai_sql_question", "")

    st.write("Example questions")
    button_columns = st.columns(2)
    for index, prompt in enumerate(button_prompts):
        if button_columns[index % 2].button(prompt, key=f"ai_prompt_{index}"):
            st.session_state["ai_sql_question"] = prompt
            st.session_state["ai_sql_pending_question"] = prompt

    with st.form("ai_sql_question_form"):
        question = st.text_input("Ask a business question", value=st.session_state.get("ai_sql_question", ""))
        submitted = st.form_submit_button("Ask the data")

    if submitted:
        st.session_state["ai_sql_question"] = question
        st.session_state["ai_sql_pending_question"] = question

    question = str(st.session_state.pop("ai_sql_pending_question", "")).strip()
    if not question:
        if submitted:
            st.warning("Enter a question or select an example.")
        return

    if demo_mode:
        ai_payload = demo_queries.get(question)
        if ai_payload is None:
            st.error("Demo Mode can only run the predefined question buttons. Configure Gemini to ask custom questions.")
            return
    else:
        with st.spinner("Asking Gemini to generate governed SQL..."):
            ai_payload, generation_error = generate_sql_with_gemini(question, project_id, dataset_id)
        if generation_error:
            st.error(generation_error)
            st.info("Try one of the example prompts or simplify the question.")
            return
        if ai_payload is None:
            st.error("Gemini did not return a usable SQL payload.")
            return

    sql = str(ai_payload.get("sql", "")).strip()
    sql, comments_removed = strip_sql_comments(sql)
    sql, aliases_sanitized = sanitize_sql_aliases(sql)
    chart_type = str(ai_payload.get("chart_type", "table")).lower()
    if chart_type not in {"table", "bar", "scatter", "line", "none"}:
        chart_type = "table"
    validation = validate_sql(sql, project_id, dataset_id, chart_type)
    if comments_removed:
        validation["warnings"].append("Gemini included SQL comments; they were removed before validation.")
    if aliases_sanitized:
        validation["warnings"].append("Gemini included display-style aliases; they were converted to BigQuery-safe snake_case aliases.")

    st.subheader("Business intent")
    st.write(ai_payload.get("business_intent", "No business intent was provided."))
    st.caption(ai_payload.get("expected_output", "No expected output was provided."))

    status_col, scan_col = st.columns(2)
    with status_col:
        if validation["is_valid"]:
            st.success("Validation passed.")
        else:
            st.error("Validation blocked this SQL.")
        for error in validation["errors"]:
            st.error(error)
        for warning in validation["warnings"]:
            st.warning(warning)

    with st.expander("Generated SQL", expanded=True):
        st.code(validation["normalized_sql"] or sql, language="sql")

    if not validation["is_valid"]:
        st.info("No BigQuery dry-run or execution was attempted because validation failed.")
        return

    with st.spinner("Running BigQuery dry-run and executing within the classroom scan limit..."):
        result_df, metadata = run_validated_query(validation["normalized_sql"], project_id)

    with scan_col:
        if metadata.get("estimated_mb") is None:
            st.metric("Estimated BigQuery scan", "Unavailable")
        else:
            st.metric("Estimated BigQuery scan", f"{metadata['estimated_mb']:.2f} MB")
        st.caption(f"Classroom maximum bytes billed: {MAX_BYTES_BILLED / (1024 * 1024):.0f} MB")

    for error in metadata["errors"]:
        st.error(error)
    for warning in metadata["warnings"]:
        st.warning(warning)
    if metadata["errors"] or result_df is None:
        st.info("Adjust the question or credentials, then run the assistant again.")
        return

    st.subheader("Result table")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.subheader("Suggested chart")
    render_ai_chart(result_df, chart_type)

    st.subheader("Executive interpretation")
    st.markdown(executive_summary(question, ai_payload, validation["normalized_sql"], result_df, metadata))

    limitations = ai_payload.get("limitations", [])
    if limitations:
        with st.expander("Caveats and limitations", expanded=False):
            for limitation in limitations:
                st.write(f"- {limitation}")

    st.session_state["ai_sql_history"].append(
        {
            "question": question,
            "sql": validation["normalized_sql"],
            "rows": len(result_df),
            "estimated_mb": metadata.get("estimated_mb"),
        }
    )
    if st.session_state["ai_sql_history"]:
        with st.expander("Session history", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state["ai_sql_history"]).tail(10), use_container_width=True, hide_index=True)


def main() -> None:
    config = load_project_config()
    project_id = config.get("gcp_project_id")
    dataset_id = config.get("bq_dataset")
    if not project_id or not dataset_id:
        st.error("Project configuration is missing `gcp_project_id` or `bq_dataset` in config/project_config.yaml.")
        st.stop()

    df_raw, load_error = load_dashboard_data(project_id, dataset_id)
    if load_error:
        st.error(load_error)
        st.stop()
    if df_raw is None or df_raw.empty:
        st.error("The BI dashboard view returned no rows. Run the upstream pipeline before opening the dashboard.")
        st.stop()

    missing_columns = validate_columns(df_raw)
    if missing_columns:
        st.error("The BI view is missing required columns: " + ", ".join(missing_columns))
        st.stop()

    df = clean_dashboard_data(df_raw)
    vehicle_preset, pricing_preset, vehicle_params, pricing_params, budget_params, customized = render_sidebar(df)
    enriched, metadata = compute_economics(df, vehicle_params, pricing_params, vehicle_preset)
    eligible, counts = apply_filters(enriched, vehicle_params)
    selected = select_portfolio(eligible, budget_params)
    selected_ids = set(selected["listing_id"]) if not selected.empty else set()
    eligible_with_actions = assign_recommended_actions(eligible, selected_ids)
    selected_with_actions = eligible_with_actions.loc[eligible_with_actions["selected"]].copy()
    summary = summarize_portfolio(selected_with_actions, budget_params)
    warnings = portfolio_warnings(enriched, eligible, selected_with_actions, summary, budget_params)
    risk_metrics = risk_confidence_metrics(enriched, eligible, selected_with_actions, counts, summary, budget_params)
    risk_warnings = risk_confidence_warnings(risk_metrics)
    overview = strategy_overview_markdown(
        vehicle_preset,
        pricing_preset,
        vehicle_params,
        pricing_params,
        metadata,
        customized,
    )

    tab_committee, tab_builder, tab_ai = st.tabs(
        [
            "Committee Dashboard",
            "Strategy & Portfolio Builder",
            "Ask the Data — GenAI SQL Assistant",
        ]
    )

    with tab_committee:
        st.title("Vehicle Portfolio Investment Simulator")
        st.caption(
            "Investment strategy, portfolio selection, financing economics and cross-sell value for a consumer finance business."
        )
        st.markdown(
            "**Core business question:** With a fixed investment budget, which vehicle portfolio maximizes expected "
            "business value through resale margin, financing margin and customer relationship value?"
        )
        st.info(
            "Regression predicts expected market price. Classification predicts commercial attractiveness. "
            "Streamlit combines model outputs with business assumptions. This is decision support, not an automatic "
            "acquisition, credit, compliance, or risk approval."
        )

        st.subheader("Selected Strategy Overview")
        st.markdown(overview)
        if metadata["apr_floor_applied"]:
            st.warning("APR floor protection is active because discounts would otherwise reduce APR below the required margin.")

        st.subheader("Executive KPIs")
        render_kpis(summary, budget_params)

        st.subheader("Committee Decision")
        render_committee_decision(summary, risk_warnings)

        st.subheader("Profit Component Chart")
        component_chart(summary)
        st.caption(
            "Profit components reconcile to expected total profit. Resale spread and income components are shown "
            "as positive values; costs are shown as negative values."
        )

        st.subheader("Risk & Confidence")
        render_risk_confidence(risk_metrics, risk_warnings)

        st.subheader("Current Recommendation")
        st.markdown(deterministic_recommendation(summary, warnings))
        for warning in [item for item in warnings if item not in risk_warnings]:
            st.warning(warning)

        st.subheader("Gemini Committee Memo")
        if st.button("Generate committee memo"):
            prompt = build_memo_prompt(
                vehicle_preset,
                pricing_preset,
                overview,
                summary,
                pricing_params,
                warnings + risk_warnings,
                selected_with_actions,
            )
            st.markdown(call_gemini(prompt))

    with tab_builder:
        st.subheader("Strategy Controls Summary")
        st.write(
            {
                "Vehicle strategy": vehicle_preset,
                "Pricing and cross-sell strategy": pricing_preset,
                "Manual overrides active": customized,
            }
        )

        st.subheader("Before and After Filtering Counts")
        count_cols = st.columns(5)
        count_cols[0].metric("Total vehicle universe", f"{counts['total_universe']:,}")
        count_cols[1].metric("Eligible after filters", f"{counts['eligible_after_filters']:,}")
        count_cols[2].metric("Selected vehicles", f"{summary['selected_count']:,}")
        count_cols[3].metric("Excluded by filters", f"{counts['excluded_by_filters']:,}")
        count_cols[4].metric("Missing model outputs excluded", f"{counts['excluded_by_missing_model_outputs']:,}")
        st.caption(f"Excluded by quality flags: {counts['excluded_by_quality_flags']:,}")

        st.subheader("Portfolio Economics Summary")
        econ_cols = st.columns(4)
        econ_cols[0].metric("Capital deployed", fmt_eur(summary["capital_deployed"]))
        econ_cols[1].metric("Remaining budget", fmt_eur(summary["remaining_budget"]))
        econ_cols[2].metric("Expected total profit", fmt_eur(summary["expected_total_profit"]))
        econ_cols[3].metric("Expected ROI", fmt_pct(summary["expected_roi"]))
        econ_cols_2 = st.columns(3)
        econ_cols_2[0].metric("Vehicle margin", fmt_eur(summary["vehicle_margin"]))
        econ_cols_2[1].metric("Financing margin", fmt_eur(summary["finance_margin"]))
        econ_cols_2[2].metric("Cross-sell income", fmt_eur(summary["cross_sell_income"]))

        st.subheader("Portfolio Opportunity Map")
        scatter_chart(eligible_with_actions)

        st.subheader("Selected Portfolio")
        table = selected_table(selected_with_actions).sort_values("Investment score", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download selected portfolio CSV",
            table.to_csv(index=False),
            file_name="selected_vehicle_portfolio.csv",
            mime="text/csv",
        )
        with st.expander("Download full eligible universe"):
            st.download_button(
                "Download eligible universe CSV",
                eligible_with_actions.to_csv(index=False),
                file_name="eligible_vehicle_universe.csv",
                mime="text/csv",
            )

        st.subheader("Business Notes")
        st.markdown(
            """
- The selected portfolio under the current mandate is the result of the current strategy and assumptions.
- Changing filters changes the investment mandate.
- Changing pricing and cross-sell assumptions changes the monetization strategy.
- The model provides decision signals; the business strategy defines the portfolio.
"""
        )

    with tab_ai:
        render_ai_sql_assistant_tab(project_id, dataset_id)


if __name__ == "__main__":
    main()
