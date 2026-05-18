from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery


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
    "Efficient Engine Campaign",
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


def efficient_fuel_defaults(fuel_types: list[str]) -> list[str]:
    efficient_keywords = ["diesel", "hybrid", "electric", "petrol", "gasoline", "d", "b"]
    selected = [
        fuel
        for fuel in fuel_types
        if any(keyword == fuel.lower() or keyword in fuel.lower() for keyword in efficient_keywords)
    ]
    return selected[:3] if selected else fuel_types[:3]


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
        "Efficient Engine Campaign": {
            "age_range": (age_min, min(age_max, q("age_years", 0.75))),
            "mileage_range": (mileage_min, min(mileage_max, q("mileage_km", 0.70))),
            "power_range": (power_min, power_max),
            "price_range": (q("actual_price_eur", 0.10), q("actual_price_eur", 0.80)),
            "fuel_types": efficient_fuel_defaults(fuel_types),
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


def populate_defaults_if_needed(vehicle_preset: str, pricing_preset: str, vehicle_defaults: dict, pricing_defaults: dict) -> None:
    previous = st.session_state.get("_active_presets")
    current = (vehicle_preset, pricing_preset)
    if previous == current:
        return

    for key, value in vehicle_defaults[vehicle_preset].items():
        st.session_state[f"vehicle_{key}"] = value * 100 if key in PERCENT_SLIDER_KEYS else value
    for key, value in pricing_defaults[pricing_preset].items():
        st.session_state[f"pricing_{key}"] = value * 100 if key in PERCENT_SLIDER_KEYS else value
    st.session_state["budget_eur"] = 3_000_000
    st.session_state["max_vehicles"] = 100
    st.session_state["cash_buffer_pct"] = 0.0
    st.session_state["allow_missing_model_outputs"] = False
    st.session_state["_active_presets"] = current


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
            "Minimum top-price probability",
            "vehicle_min_top_price_probability",
            0.0,
            100.0,
            1.0,
        )
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
    result["vehicle_margin"] = (
        result["conservative_resale_price"]
        - result["actual_price_eur"]
        - pricing_params["reconditioning_cost"]
        - pricing_params["transaction_cost"]
    )
    result["capital_deployed"] = (
        result["actual_price_eur"] + pricing_params["reconditioning_cost"] + pricing_params["transaction_cost"]
    )
    result["financed_amount"] = result["conservative_resale_price"] * pricing_params["financed_amount_pct"]

    raw_effective_apr = (
        pricing_params["base_customer_apr"]
        - pricing_params["insurance_apr_discount"]
        - pricing_params["fuel_card_apr_discount"]
        - pricing_params["payroll_apr_discount"]
    )
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

    # APR discounts improve the product bundle. The elasticity translates discount size into attach-rate uplift.
    result["adjusted_insurance_attach_rate"] = clipped(
        pricing_params["insurance_attach_rate"]
        + pricing_params["attach_rate_elasticity"] * pricing_params["insurance_apr_discount"]
    )
    result["adjusted_fuel_card_attach_rate"] = clipped(
        pricing_params["fuel_card_attach_rate"]
        + pricing_params["attach_rate_elasticity"] * pricing_params["fuel_card_apr_discount"]
    )
    result["adjusted_payroll_attach_rate"] = clipped(
        pricing_params["payroll_attach_rate"]
        + pricing_params["attach_rate_elasticity"] * pricing_params["payroll_apr_discount"]
    )

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
        "apr_floor": apr_floor,
        "apr_floor_applied": effective_customer_apr > raw_effective_apr,
        "net_finance_spread": net_finance_spread,
    }
    return result.replace([np.inf, -np.inf], np.nan), metadata


def portfolio_fit_weight(df: pd.DataFrame, vehicle_preset: str, vehicle_params: dict[str, Any]) -> pd.Series:
    if vehicle_preset == "Broad Market":
        return pd.Series(1.0, index=df.index)
    if vehicle_preset == "Young Low-Mileage Core":
        return (0.80 + 0.20 * normalized_inverse(df["age_years"]) + 0.20 * normalized_inverse(df["mileage_km"])).clip(0.8, 1.2)
    if vehicle_preset == "Efficient Engine Campaign":
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
            "finance_margin": 0.0,
            "cross_sell_income": 0.0,
            "insurance_income": 0.0,
            "fuel_card_income": 0.0,
            "payroll_income": 0.0,
            "inventory_funding_cost": 0.0,
            "reconditioning_transaction_cost": 0.0,
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
        "finance_margin": float(selected["finance_margin"].sum()),
        "cross_sell_income": float(selected["cross_sell_income"].sum()),
        "insurance_income": float(selected["insurance_income"].sum()),
        "fuel_card_income": float(selected["fuel_card_income"].sum()),
        "payroll_income": float(selected["payroll_income"].sum()),
        "inventory_funding_cost": float(selected["inventory_funding_cost"].sum()),
        "reconditioning_transaction_cost": float(
            (selected["capital_deployed"] - selected["actual_price_eur"]).sum()
        ),
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


def strategy_interpretation(vehicle_preset: str, pricing_preset: str) -> str:
    if vehicle_preset == "Broad Market":
        priority = "volume and opportunity discovery"
    elif vehicle_preset == "Young Low-Mileage Core":
        priority = "portfolio cleanliness and resale simplicity"
    elif vehicle_preset == "Efficient Engine Campaign":
        priority = "campaign coherence"
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
    row3[1].metric("Average top-price probability", fmt_pct(summary["avg_top_price_probability"]))
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
- Minimum top-price probability: {fmt_pct(vehicle_params["min_top_price_probability"])}
- Minimum expected discount: {fmt_pct(vehicle_params["min_expected_discount_pct"])}

**Commercial strategy**
- Resale haircut: {fmt_pct(pricing_params["resale_haircut"])}
- Financing take-up: {fmt_pct(pricing_params["financing_take_up_rate"])}
- Customer APR: {fmt_pct(metadata["effective_customer_apr"])}
- APR discounts: insurance {fmt_pct(pricing_params["insurance_apr_discount"])}, fuel card {fmt_pct(pricing_params["fuel_card_apr_discount"])}, payroll transfer {fmt_pct(pricing_params["payroll_apr_discount"])}
- Attach rates: insurance {fmt_pct(pricing_params["insurance_attach_rate"])}, fuel card {fmt_pct(pricing_params["fuel_card_attach_rate"])}, payroll transfer {fmt_pct(pricing_params["payroll_attach_rate"])}

{strategy_interpretation(vehicle_preset, pricing_preset)}
"""


def component_chart(summary: dict[str, float]) -> None:
    components = pd.DataFrame(
        {
            "Component": [
                "Vehicle margin",
                "Financing margin",
                "Insurance income",
                "Fuel card income",
                "Payroll income",
                "Inventory funding cost",
                "Reconditioning and transaction costs",
            ],
            "Amount": [
                summary["vehicle_margin"],
                summary["finance_margin"],
                summary["insurance_income"],
                summary["fuel_card_income"],
                summary["payroll_income"],
                -summary["inventory_funding_cost"],
                -summary["reconditioning_transaction_cost"],
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
    return selected[[column for column in columns if column in selected.columns]].copy()


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
        title="Opportunity map: expected discount vs top-price probability",
    )
    fig.update_layout(xaxis_title="Expected discount %", yaxis_title="Top-price probability")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def gemini_api_key() -> str | None:
    try:
        value = st.secrets.get("GEMINI_API_KEY")
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


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
Include this exact warning:
"This is a simulated decision-support tool. It is not a final acquisition, credit, compliance, or risk approval."

Keep the memo concise, executive, and non-technical.
"""


def call_gemini(prompt: str) -> str:
    api_key = gemini_api_key()
    if not api_key:
        return (
            "Gemini API key not configured. Add it to Streamlit secrets or the GEMINI_API_KEY environment variable."
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text or "Gemini returned an empty memo."
    except Exception:
        return "Gemini memo generation failed. Check the API key, Gemini model name, and network access."


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
    overview = strategy_overview_markdown(
        vehicle_preset,
        pricing_preset,
        vehicle_params,
        pricing_params,
        metadata,
        customized,
    )

    tab_committee, tab_builder = st.tabs(["Committee Dashboard", "Strategy & Portfolio Builder"])

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
            "Regression predicts expected market price. Classification predicts top-price attractiveness. "
            "Streamlit combines model outputs with business assumptions. This is decision support, not an automatic "
            "acquisition, credit, compliance, or risk approval."
        )

        st.subheader("Selected Strategy Overview")
        st.markdown(overview)
        if metadata["apr_floor_applied"]:
            st.warning("APR floor protection is active because discounts would otherwise reduce APR below the required margin.")

        st.subheader("Executive KPIs")
        render_kpis(summary, budget_params)

        st.subheader("Profit Component Chart")
        component_chart(summary)

        st.subheader("Current Recommendation")
        st.markdown(deterministic_recommendation(summary, warnings))
        for warning in warnings:
            st.warning(warning)

        st.subheader("Gemini Committee Memo")
        if st.button("Generate committee memo"):
            prompt = build_memo_prompt(
                vehicle_preset,
                pricing_preset,
                overview,
                summary,
                pricing_params,
                warnings,
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
        table = selected_table(selected_with_actions).sort_values("investment_score", ascending=False)
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
- The selected portfolio is the result of the current strategy and assumptions.
- Changing filters changes the investment mandate.
- Changing pricing and cross-sell assumptions changes the monetization strategy.
- The model provides decision signals; the business strategy defines the portfolio.
"""
        )


if __name__ == "__main__":
    main()
