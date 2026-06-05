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
    "Mercado amplio",
    "Núcleo joven con bajo kilometraje",
    "Campaña retail generalista",
    "Margen en vehículos de mayor precio",
    "Riesgo conservador",
]

PRICING_PRESETS = [
    "Caso base",
    "Venta cruzada agresiva",
    "Foco en margen financiero",
    "Foco en fidelización",
    "Escenario de estrés",
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
    "¿Qué listados debería revisar primero el comité?",
    "¿Qué tipos de combustible concentran las mayores oportunidades de precios?",
    "¿Dónde coinciden la brecha expected-price y la probabilidad top-price?",
    "¿Qué oportunidades son riesgosas debido a las banderas de calidad?",
    "Resuma la distribución decision_flag.",
    "Compare el precio real con el esperado por año de registro.",
    "Muestre casos en los que faltan los resultados del modelo.",
]

DEMO_QUESTIONS = [
    "¿Qué listados debería revisar primero el comité?",
    "¿Qué tipos de combustible concentran las mayores oportunidades de precios?",
    "¿Dónde coinciden las señales de regresión y clasificación?",
    "¿Qué oportunidades tienen indicadores de calidad o de riesgo lógico?",
    "¿Cuántos listados caen en cada indicador de decisión?",
]

DECISION_FLAG_MEANINGS = {
    "high_priority_review": "Precio por debajo del esperado y fuerte señal top-price",
    "price_opportunity": "por debajo del precio esperado pero la señal top-price es más débil o falta",
    "top_price_signal": "señal de clasificación positiva, la diferencia de precios no es necesariamente atractiva",
    "review_missing_model_outputs": "Faltan resultados del modelo y es necesario revisarlos.",
    "standard_review": "no hay señal de prioridad fuerte",
}

BI_FIELD_MEANINGS = {
    "actual_price_eur": "precio de cotización observado en el mercado",
    "expected_price_eur": "estimación del modelo de regresión del precio normal de mercado",
    "expected_price_gap_eur": "precio real menos precio esperado; negativo significa por debajo del valor esperado por el modelo",
    "top_price_probability": "probabilidad del modelo de clasificación asociada a la señal externa top-price",
    "predicted_top_price": "Salida del modelo binario para la tarea de clasificación top-price.",
    "decision_flag": "indicador de prioridad empresarial producido en la vista BI",
    "price_outlier_iqr": "Marca de calidad/riesgo de precio basada en IQR",
    "mileage_outlier_iqr": "Indicador de calidad/riesgo de kilometraje basado en IQR",
    "power_outlier_iqr": "Indicador de riesgo y calidad de energía basado en IQR",
    "logical_issue": "Marca de problema de reglas de negocio o calidad de datos lógicos",
}


st.set_page_config(
    page_title="Simulador De Inversión En Portfolio De Vehículos",
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


@st.cache_data(ttl=900, show_spinner="Cargando datos del panel BI desde BigQuery...")
def load_dashboard_data(project_id: str, dataset_id: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        client = bigquery.Client(project=project_id)
        df = client.query(query_text(project_id, dataset_id)).to_dataframe()
        return df, None
    except DefaultCredentialsError:
        return None, (
            "Las credenciales de BigQuery no están configuradas. Autentícate con Google Cloud antes de ejecutar "
            "el panel de control, por ejemplo con `gcloud auth application-default login`, o configurar "
            "las credenciales de la cuenta de servicio utilizadas por su entorno."
        )
    except GoogleAPIError:
        return None, (
            "No se pudo alcanzar BigQuery o el proyecto configurado y el conjunto de datos no están disponibles. "
            "Verifique el proyecto Google Cloud, el conjunto de datos, los permisos y el estado de la canalización ascendente."
        )
    except Exception:
        return None, (
            "El panel no pudo cargar la vista BI. Verifique las credenciales, el acceso a la red y la "
            "proyecto y conjunto de datos BigQuery configurados."
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

    cleaned["fuel_type"] = cleaned["fuel_type"].fillna("Desconocido").astype(str)
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
        "Mercado amplio": {
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
        "Núcleo joven con bajo kilometraje": {
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
        "Campaña retail generalista": {
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
        "Margen en vehículos de mayor precio": {
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
        "Riesgo conservador": {
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
        "Caso base": {
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
        "Venta cruzada agresiva": {
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
        "Foco en margen financiero": {
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
        "Foco en fidelización": {
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
        "Escenario de estrés": {
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
    st.sidebar.title("Configuración de estrategia")
    vehicle_preset = st.sidebar.selectbox("Estrategia de filtrado de vehículos", VEHICLE_PRESETS)
    pricing_preset = st.sidebar.selectbox("Estrategia de precios y venta cruzada", PRICING_PRESETS)

    vehicle_defaults = build_vehicle_preset_defaults(df)
    pricing_defaults = pricing_preset_defaults()
    populate_defaults_if_needed(vehicle_preset, pricing_preset, vehicle_defaults, pricing_defaults)

    if st.sidebar.button("Restablecer todas las suposiciones"):
        reset_all_assumptions(vehicle_preset, pricing_preset, vehicle_defaults, pricing_defaults)
        st.rerun()

    st.sidebar.info(
        "Los ajustes preestablecidos de estrategia son suposiciones comerciales. Definen mandato de inversión y campaña comercial "
        "lógica. No cambian los resultados del modelo subyacente."
    )

    with st.sidebar.expander("Controles de inversión", expanded=True):
        budget = st.number_input("Presupuesto de inversión en euros", min_value=0, step=50_000, key="budget_eur")
        max_vehicles = st.number_input("Número máximo de vehículos", min_value=1, step=1, key="max_vehicles")
        cash_buffer_pct = percent_slider("% mínimo de colchón de efectivo", "cash_buffer_pct", 0.0, 50.0, 1.0)

    age_min, age_max = numeric_bounds(df, "age_years")
    mileage_min, mileage_max = numeric_bounds(df, "mileage_km")
    power_min, power_max = numeric_bounds(df, "power_hp")
    price_min, price_max = numeric_bounds(df, "actual_price_eur")
    fuel_options = sorted(df["fuel_type"].dropna().astype(str).unique().tolist())

    with st.sidebar.expander("Filtros de vehículos", expanded=True):
        age_range = st.slider("rango de edad", age_min, age_max, key="vehicle_age_range")
        mileage_range = st.slider("Rango de kilometraje", mileage_min, mileage_max, key="vehicle_mileage_range")
        power_range = st.slider("Rango de potencia", power_min, power_max, key="vehicle_power_range")
        price_range = st.slider("rango de precio de compra", price_min, price_max, key="vehicle_price_range")
        fuel_types = st.multiselect("Tipos de combustible", fuel_options, key="vehicle_fuel_types")
        min_top_price_probability = percent_slider(
            "Puntuación mínima de atractivo comercial",
            "vehicle_min_top_price_probability",
            0.0,
            100.0,
            1.0,
        )
        st.caption("Esta puntuación proviene del campo del modelo de clasificación ascendente `top_price_probability`.")
        min_expected_discount_pct = percent_slider(
            "% de descuento mínimo esperado",
            "vehicle_min_expected_discount_pct",
            -50.0,
            50.0,
            1.0,
        )
        exclude_price_outlier = st.checkbox("Excluir indicador de valor atípico de precio", key="vehicle_exclude_price_outlier")
        exclude_mileage_outlier = st.checkbox("Excluir indicador de valor atípico de kilometraje", key="vehicle_exclude_mileage_outlier")
        exclude_power_outlier = st.checkbox("Excluir indicador de valor atípico de potencia", key="vehicle_exclude_power_outlier")
        exclude_logical_issue = st.checkbox("Excluir indicador de problema lógico", key="vehicle_exclude_logical_issue")
        allow_missing_model_outputs = st.checkbox(
            "Permitir salidas de modelo faltantes",
            key="allow_missing_model_outputs",
            help="Deshabilitado de forma predeterminada porque la cartera debe utilizar señales tanto de regresión como de clasificación.",
        )

    with st.sidebar.expander("Economía de reventa", expanded=False):
        resale_haircut = percent_slider("Recorte conservador de reventa sobre el precio esperado", "pricing_resale_haircut", 0.0, 40.0)
        reconditioning_cost = st.number_input("Costo de reacondicionamiento por vehículo", min_value=0, step=50, key="pricing_reconditioning_cost")
        transaction_cost = st.number_input("Costo de transacción por vehículo", min_value=0, step=50, key="pricing_transaction_cost")
        inventory_funding_cost_rate = percent_slider("Tasa de costo de financiación de inventario anual", "pricing_inventory_funding_cost_rate", 0.0, 30.0)
        expected_days_to_resale = st.number_input("Días esperados para revender", min_value=1, max_value=365, step=5, key="vehicle_expected_days_to_resale")

    with st.sidebar.expander("Supuestos de financiación", expanded=False):
        financed_amount_pct = percent_slider("Monto financiado esperado como % del precio de reventa", "pricing_financed_amount_pct", 0.0, 100.0)
        base_customer_apr = percent_slider("Cliente base APR", "pricing_base_customer_apr", 0.0, 30.0)
        funding_cost_rate = percent_slider("Tasa de coste de financiación", "pricing_funding_cost_rate", 0.0, 20.0)
        credit_risk_cost_rate = percent_slider("Tasa de coste del riesgo de crédito esperada", "pricing_credit_risk_cost_rate", 0.0, 20.0)
        loan_term_months = st.number_input("Plazo medio del préstamo en meses", min_value=1, max_value=120, step=1, key="pricing_loan_term_months")
        financing_take_up_rate = percent_slider("Tasa de absorción de financiación", "pricing_financing_take_up_rate", 0.0, 100.0)
        min_net_finance_margin_buffer = percent_slider("Colchón mínimo de margen financiero neto", "pricing_min_net_finance_margin_buffer", 0.0, 10.0, 0.25)

    with st.sidebar.expander("Venta cruzada y fidelización", expanded=False):
        insurance_attach_rate = percent_slider("Base de tasa de fijación de seguros", "pricing_insurance_attach_rate", 0.0, 100.0)
        insurance_commission = st.number_input("Comisión de seguro por póliza", min_value=0, step=25, key="pricing_insurance_commission")
        insurance_apr_discount = percent_slider("Descuento en el tipo de interés si se contrata un seguro", "pricing_insurance_apr_discount", 0.0, 10.0, 0.25)
        fuel_card_attach_rate = percent_slider("Base de tasa de conexión de tarjeta de combustible", "pricing_fuel_card_attach_rate", 0.0, 100.0)
        fuel_card_annual_margin = st.number_input("Margen anual de la tarjeta de combustible", min_value=0, step=10, key="pricing_fuel_card_annual_margin")
        fuel_card_years = st.number_input("Duración prevista de la tarjeta de combustible en años", min_value=0.0, max_value=10.0, step=0.25, key="pricing_fuel_card_years")
        fuel_card_apr_discount = percent_slider("Descuento en el tipo de interés si se contrata tarjeta de combustible", "pricing_fuel_card_apr_discount", 0.0, 10.0, 0.25)
        payroll_attach_rate = percent_slider("Base de tasa de fijación de transferencia de nómina", "pricing_payroll_attach_rate", 0.0, 100.0)
        payroll_lifetime_value = st.number_input("Valor de vida del cliente de nómina", min_value=0, step=25, key="pricing_payroll_lifetime_value")
        payroll_apr_discount = percent_slider("Descuento en tasa de interés si se transfiere nómina", "pricing_payroll_apr_discount", 0.0, 10.0, 0.25)
        attach_rate_elasticity = st.number_input(
            "Parámetro de elasticidad",
            min_value=0.0,
            max_value=20.0,
            step=0.25,
            key="pricing_attach_rate_elasticity",
            help="Cada descuento de 1 punto porcentual APR aumenta la tasa de fijación en esa cantidad de puntos porcentuales.",
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
    st.sidebar.caption(f"Cambios manuales activos: {'Sí' if customized else 'No'}")
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

    # Los descuentos APR mejoran el paquete de productos. La elasticidad traduce el tamaño del descuento en un aumento de la tasa de interés.
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

    # La velocidad de reventa es un indicador transparente, no un modelo entrenado: puntuación atractiva, menor edad y menor kilometraje.
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
    if vehicle_preset == "Mercado amplio":
        return pd.Series(1.0, index=df.index)
    if vehicle_preset == "Núcleo joven con bajo kilometraje":
        return (0.80 + 0.20 * normalized_inverse(df["age_years"]) + 0.20 * normalized_inverse(df["mileage_km"])).clip(0.8, 1.2)
    if vehicle_preset == "Campaña retail generalista":
        selected_fuels = set(vehicle_params["fuel_types"])
        fuel_reward = df["fuel_type"].isin(selected_fuels).astype(float)
        moderate_price = 1 - (normalized_positive(df["actual_price_eur"]) - 0.5).abs() * 2
        return (0.85 + 0.20 * fuel_reward + 0.15 * moderate_price).clip(0.75, 1.2)
    if vehicle_preset == "Margen en vehículos de mayor precio":
        return (0.80 + 0.20 * normalized_positive(df["vehicle_margin"]) + 0.20 * normalized_positive(df["conservative_resale_price"])).clip(0.8, 1.25)
    if vehicle_preset == "Riesgo conservador":
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
    result["recommended_action"] = "No priorizar"
    result.loc[
        result["selected"] & (result["expected_total_profit"] > 0) & (~result["quality_issue"]),
        "recommended_action",
    ] = "Candidato a compra"
    result.loc[result["selected"] & result["quality_issue"], "recommended_action"] = "Revisión manual"
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
        warnings.append("Ningún vehículo cumple con los criterios de la estrategia actual.")
    elif len(eligible) < max(10, len(df) * 0.03):
        warnings.append("El universo elegible es pequeño. El mandato puede ser demasiado limitado para una decisión sólida del comité.")
    if selected.empty:
        warnings.append("No se seleccionó ningún vehículo porque ningún candidato comercial positivo se ajustaba al presupuesto y los supuestos.")
    budget_used = summary["capital_deployed"] / budget_params["budget_eur"] if budget_params["budget_eur"] else 0
    if selected.shape[0] > 0 and budget_used < 0.50:
        warnings.append("La cartera seleccionada utiliza menos de la mitad del presupuesto disponible.")
    if selected.shape[0] > 0 and selected["fuel_type"].value_counts(normalize=True).iloc[0] > 0.70:
        warnings.append("El portafolio seleccionado se concentra en un tipo de combustible.")
    if selected.shape[0] > 0:
        price_bands = pd.cut(selected["actual_price_eur"], bins=4, duplicates="drop")
        if price_bands.value_counts(normalize=True).iloc[0] > 0.70:
            warnings.append("La cartera seleccionada se concentra en una banda de precios de compra.")
    if summary["expected_total_profit"] > 0 and summary["cross_sell_income"] / summary["expected_total_profit"] > 0.50:
        warnings.append("El ROI esperado está fuertemente influenciado por los supuestos de lealtad y venta cruzada.")
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
        warnings.append("La cobertura de salida del modelo es inferior al 90%.")
    if metrics["selected_quality_share"] > 0:
        warnings.append("El portafolio seleccionado incluye vehículos con banderas de calidad.")
    if metrics["fuel_concentration"] > 0.70:
        warnings.append("La concentración del tipo de combustible es superior al 70%.")
    if metrics["price_band_concentration"] > 0.70:
        warnings.append("La concentración de la banda de precios es superior al 70%.")
    if metrics["cross_sell_dependency"] > 0.50:
        warnings.append("La dependencia de las ventas cruzadas supera el 50% del beneficio esperado.")
    if metrics["budget_used"] < 0.50:
        warnings.append("La cartera seleccionada utiliza menos del 50% del presupuesto disponible.")
    return warnings


def committee_decision(summary: dict[str, float], risk_warnings: list[str]) -> tuple[str, list[str], list[str]]:
    if summary["selected_count"] == 0 or summary["expected_roi"] < 0.05:
        status = "No avanzar"
    elif summary["expected_roi"] >= 0.12 and not risk_warnings:
        status = "Recomendar"
    else:
        status = "Revisar"

    reasons = [
        f"{summary['selected_count']:,} vehículos seleccionados bajo el mandato actual.",
        f"El ROI esperado del portfolio es {fmt_pct(summary['expected_roi'])}.",
        f"El beneficio total esperado es {fmt_eur(summary['expected_total_profit'])}.",
    ]
    risks = (risk_warnings + [
        "Validar tiempos de reventa, capacidad de reacondicionamiento y condiciones finales de compra.",
        "Confirme que los supuestos sobre la tasa de fijación comercial sean realistas para esta campaña.",
        "Revise los indicadores de calidad de los vehículos individuales antes de cualquier decisión de adquisición.",
    ])[:3]
    return status, reasons[:3], risks


def render_committee_decision(summary: dict[str, float], risk_warnings: list[str]) -> None:
    status, reasons, risks = committee_decision(summary, risk_warnings)
    if status == "Recomendar":
        st.success(f"**Estado de decisión: {status}**")
    elif status == "Revisar":
        st.warning(f"**Estado de decisión: {status}**")
    else:
        st.error(f"**Estado de decisión: {status}**")

    reason_col, risk_col = st.columns(2)
    with reason_col:
        st.markdown("**Razones**")
        for reason in reasons:
            st.markdown(f"- {reason}")
    with risk_col:
        st.markdown("**Riesgos o puntos de validación**")
        for risk in risks:
            st.markdown(f"- {risk}")

    st.caption(
        "El estado de la decisión es un resultado de simulación para la discusión del comité. No es una adquisición definitiva, "
        "aprobación de crédito, cumplimiento o riesgo."
    )


def main_risk_driver(metrics: dict[str, float]) -> tuple[str, str]:
    if metrics["model_output_coverage"] < 0.90:
        return "Cobertura de salida de modelo baja", "Faltan señales modelo para una parte importante del universo de vehículos."
    if metrics["selected_quality_share"] > 0:
        return "Banderas de calidad de vehículos seleccionadas", "Algunos vehículos seleccionados requieren una revisión de calidad manual antes de actuar."
    if metrics["fuel_concentration"] > 0.70:
        return "Concentración del tipo de combustible", "El portafolio seleccionado se concentra en un tipo de combustible."
    if metrics["price_band_concentration"] > 0.70:
        return "Concentración de la banda de precios", "La cartera seleccionada se concentra en una banda de precios de compra."
    if metrics["cross_sell_dependency"] > 0.50:
        return (
            "Dependencia de venta cruzada",
            "La dependencia de las ventas cruzadas significa que las ganancias esperadas dependen materialmente de supuestos de tasas de fijación comerciales.",
        )
    if metrics["budget_used"] < 0.50:
        return (
            "Despliegue de bajo presupuesto",
            "El bajo despliegue presupuestario significa que el mandato actual puede ser demasiado restrictivo para absorber el capital disponible.",
        )
    return "No se ha detectado ningún factor de riesgo importante", "Los indicadores de riesgo actuales no muestran una única preocupación dominante."


def render_risk_confidence(metrics: dict[str, float], warnings: list[str]) -> None:
    row1 = st.columns(3)
    row1[0].metric("Cobertura de salida del modelo", fmt_pct(metrics["model_output_coverage"]))
    row1[1].metric("Se excluyen los resultados del modelo que faltan", f"{metrics['missing_model_outputs_excluded']:,}")
    row1[2].metric("Vehículos elegibles con banderas de calidad.", fmt_pct(metrics["eligible_quality_share"]))

    row2 = st.columns(4)
    row2[0].metric("Vehículos seleccionados con banderas de calidad.", fmt_pct(metrics["selected_quality_share"]))
    row2[1].metric("Mayor concentración de combustible", fmt_pct(metrics["fuel_concentration"]))
    row2[2].metric("Mayor concentración de la banda de precios", fmt_pct(metrics["price_band_concentration"]))
    row2[3].metric("Tasa de dependencia de ventas cruzadas", fmt_pct(metrics["cross_sell_dependency"]))

    driver, explanation = main_risk_driver(metrics)
    st.markdown(f"**Principal driver de riesgo:** {driver}. {explanation}")

    for warning in warnings:
        st.warning(warning)


def strategy_interpretation(vehicle_preset: str, pricing_preset: str) -> str:
    if vehicle_preset == "Mercado amplio":
        priority = "descubrimiento de volumen y oportunidades"
    elif vehicle_preset == "Núcleo joven con bajo kilometraje":
        priority = "limpieza de cartera y simplicidad de reventa"
    elif vehicle_preset == "Campaña retail generalista":
        priority = "Coherencia de la campaña y selección minorista fácil de explicar."
    elif vehicle_preset == "Margen en vehículos de mayor precio":
        priority = "margen unitario con concentración de capital"
    else:
        priority = "control de riesgos y defensa"

    if pricing_preset == "Foco en fidelización":
        priority += " con la economía de la lealtad del cliente"
    elif pricing_preset == "Venta cruzada agresiva":
        priority += " con paquete comercial"
    elif pricing_preset == "Escenario de estrés":
        priority += " bajo supuestos adversos"
    return f"Esta estrategia prioriza {priority}."


def deterministic_recommendation(summary: dict[str, float], warnings: list[str]) -> str:
    roi = summary["expected_roi"]
    if summary["selected_count"] == 0:
        status = "poco atractiva"
        reason = "Ningún vehículo cumple con los criterios actuales de inversión y rentabilidad."
    elif roi >= 0.12:
        status = "atractiva"
        reason = "el ROI esperado del portfolio está por encima de un umbral fuerte para comité"
    elif roi >= 0.05:
        status = "en revisión"
        reason = "el ROI esperado es positivo pero requiere validación cuidadosa"
    else:
        status = "poco atractiva"
        reason = "el ROI esperado es demasiado bajo para los supuestos actuales"

    drivers = {
        "margen de reventa del vehículo": summary["vehicle_margin"],
        "margen de financiación": summary["finance_margin"],
        "ingresos por venta cruzada y fidelización": summary["cross_sell_income"],
    }
    driver = max(drivers, key=drivers.get)
    caution = warnings[0] if warnings else "Valide los precios, el momento de la reventa y la capacidad operativa antes de la aprobación."
    return (
        f"La estrategia actual es **{status}** porque {reason}. "
        f"El principal driver de valor es **{driver}**. Principal cautela: {caution}"
    )


def fmt_eur(value: float) -> str:
    return f"EUR {value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def render_kpis(summary: dict[str, float], budget_params: dict[str, Any]) -> None:
    row1 = st.columns(4)
    row1[0].metric("Presupuesto de inversión", fmt_eur(budget_params["budget_eur"]))
    row1[1].metric("Capital desplegado", fmt_eur(summary["capital_deployed"]))
    row1[2].metric("Presupuesto restante", fmt_eur(summary["remaining_budget"]))
    row1[3].metric("Vehículos seleccionados", f"{summary['selected_count']:,}")

    row2 = st.columns(4)
    row2[0].metric("Beneficio total esperado", fmt_eur(summary["expected_total_profit"]))
    row2[1].metric("Cartera esperada ROI", fmt_pct(summary["expected_roi"]))
    row2[2].metric("Margen de reventa de vehículos", fmt_eur(summary["vehicle_margin"]))
    row2[3].metric("Margen de financiación", fmt_eur(summary["finance_margin"]))

    row3 = st.columns(4)
    row3[0].metric("Ingresos por venta cruzada y fidelización", fmt_eur(summary["cross_sell_income"]))
    row3[1].metric(
        "Puntuación media de atractivo comercial",
        fmt_pct(summary["avg_top_price_probability"]),
        help="Esta puntuación proviene del campo del modelo de clasificación ascendente `top_price_probability`.",
    )
    row3[2].metric("% de descuento esperado promedio", fmt_pct(summary["avg_expected_discount_pct"]))
    row3[3].metric("Beneficio medio por vehículo", fmt_eur(summary["avg_expected_profit_per_vehicle"]))


def strategy_overview_markdown(
    vehicle_preset: str,
    pricing_preset: str,
    vehicle_params: dict[str, Any],
    pricing_params: dict[str, Any],
    metadata: dict[str, Any],
    customized: bool,
) -> str:
    return f"""
**Estrategia de vehículo seleccionada:** {vehicle_preset}  
**Estrategia de pricing y venta cruzada seleccionada:** {pricing_preset}  
**Personalización manual:** {'Sí' if customized else 'No'}

**Mandato de vehículos elegibles**
- Rango de edad: {vehicle_params["age_range"][0]:,.1f} a {vehicle_params["age_range"][1]:,.1f} años
- Rango de kilometraje: {vehicle_params["mileage_range"][0]:,.0f} a {vehicle_params["mileage_range"][1]:,.0f} km
- Tipos de combustible: {", ".join(vehicle_params["fuel_types"]) if vehicle_params["fuel_types"] else "Ninguno seleccionado"}
- Rango de precio de compra: {fmt_eur(vehicle_params["price_range"][0])} a {fmt_eur(vehicle_params["price_range"][1])}
- Score mínimo de atractivo comercial: {fmt_pct(vehicle_params["min_top_price_probability"])}
- Descuento esperado mínimo: {fmt_pct(vehicle_params["min_expected_discount_pct"])}

**Estrategia comercial**
- Haircut de reventa: {fmt_pct(pricing_params["resale_haircut"])}
- Tasa de contratación de financiación: {fmt_pct(pricing_params["financing_take_up_rate"])}
- APR efectivo esperado tras descuentos ponderados de venta cruzada: {fmt_pct(metadata["effective_customer_apr"])}
- Descuento APR ponderado esperado: {fmt_pct(metadata["expected_total_apr_discount"])}
- Descuentos APR por producto: seguro {fmt_pct(pricing_params["insurance_apr_discount"])}, tarjeta de combustible {fmt_pct(pricing_params["fuel_card_apr_discount"])}, nómina {fmt_pct(pricing_params["payroll_apr_discount"])}
- Tasa de contratación de seguro: {fmt_pct(pricing_params["insurance_attach_rate"])} base -> {fmt_pct(metadata["adjusted_insurance_attach_rate"])} esperada tras el efecto del descuento
- Tasa de contratación de tarjeta de combustible: {fmt_pct(pricing_params["fuel_card_attach_rate"])} base -> {fmt_pct(metadata["adjusted_fuel_card_attach_rate"])} esperada tras el efecto del descuento
- Tasa de vinculación de nómina: {fmt_pct(pricing_params["payroll_attach_rate"])} base -> {fmt_pct(metadata["adjusted_payroll_attach_rate"])} esperada tras el efecto del descuento

{strategy_interpretation(vehicle_preset, pricing_preset)}
"""


def component_chart(summary: dict[str, float]) -> None:
    components = pd.DataFrame(
        {
            "Componente": [
                "Diferencial bruto de reventa",
                "Costo de reacondicionamiento",
                "Costo de transacción",
                "Margen de financiación",
                "Ingresos por seguros",
                "Ingresos por tarjeta de combustible",
                "Ingresos de nómina",
                "Costo de financiación del inventario",
            ],
            "Importe": [
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
    fig = px.bar(components, x="Componente", y="Importe", title="Componentes de ganancias esperadas")
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
            "listing_id": "ID de listado",
            "make": "Marca",
            "model": "Modelo",
            "fuel_type": "Tipo de combustible",
            "actual_price_eur": "Precio de compra",
            "expected_price_eur": "Precio de mercado esperado",
            "expected_discount_pct": "% de descuento esperado",
            "top_price_probability": "Puntuación de atractivo comercial",
            "vehicle_margin": "Margen del vehículo",
            "finance_margin": "Margen de financiación",
            "insurance_income": "Ingresos por seguros",
            "fuel_card_income": "Ingresos por tarjeta de combustible",
            "payroll_income": "Ingresos de nómina",
            "cross_sell_income": "Ingresos por venta cruzada",
            "expected_total_profit": "Beneficio esperado",
            "expected_roi": "ROI esperado",
            "investment_score": "Puntuación de inversión",
            "recommended_action": "Acción recomendada",
            "price_outlier_iqr": "Bandera de precio atípico",
            "mileage_outlier_iqr": "Bandera de valor atípico de kilometraje",
            "power_outlier_iqr": "Bandera de valor atípico de potencia",
            "logical_issue": "Marca de problema lógico",
        }
    )


def scatter_chart(eligible_with_actions: pd.DataFrame) -> None:
    if eligible_with_actions.empty:
        st.info("No hay vehículos elegibles disponibles para la estrategia actual.")
        return
    plot_df = eligible_with_actions.copy()
    plot_df["selection_status"] = np.where(plot_df["selected"], "Seleccionado", "No seleccionado")
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
        title="Mapa de oportunidades: descuento esperado versus puntaje de atractivo comercial",
        labels={
            "top_price_probability": "Puntuación de atractivo comercial",
            "expected_discount_pct": "% de descuento esperado",
            "actual_price_eur": "Precio real EUR",
            "expected_price_eur": "Precio esperado EUR",
            "expected_total_profit": "Beneficio total esperado",
            "expected_roi": "ROI esperado",
        },
    )
    fig.update_layout(xaxis_title="% de descuento esperado", yaxis_title="Puntuación de atractivo comercial")
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
Escribe un memo conciso, listo para comité, en castellano, para un negocio de consumer finance.

Secciones obligatorias:
1. Estrategia seleccionada y mandato de inversión
2. Recomendación ejecutiva
3. Racional del portfolio
4. Supuestos clave
5. Resultado financiero esperado
6. Lógica de venta cruzada y fidelización
7. Principales riesgos y limitaciones del modelo
8. Siguientes pasos sugeridos de validación

Estrategia de vehículo seleccionada: {vehicle_preset}
Estrategia de pricing y venta cruzada seleccionada: {pricing_preset}

Resumen de estrategia seleccionada:
{overview}

Métricas agregadas:
{summary}

Supuestos de pricing y venta cruzada:
{pricing_params}

Avisos:
{warnings}

Top 10 candidatos seleccionados:
{top_candidates}

El memo debe explicar la estrategia que produjo el portfolio, no limitarse a resumir coches seleccionados.
Usa "score de atractivo comercial" al referirte a la señal del modelo de clasificación desde `top_price_probability`.
Incluye este aviso exacto:
"Esta es una herramienta simulada de soporte a la decisión. No es una aprobación final de adquisición, crédito, compliance o riesgo."

Mantén el memo conciso, ejecutivo y no técnico.
"""


def call_gemini(prompt: str) -> str:
    text, error = generate_gemini_content(prompt)
    if error:
        return f"{error} El comportamiento determinista/default del dashboard sigue disponible."
    return text or "Gemini devolvió una respuesta vacía. El comportamiento determinista/predeterminado del panel todavía está disponible."


def allowed_bi_view(project_id: str, dataset_id: str) -> str:
    return ALLOWED_BI_VIEW_TEMPLATE.format(project_id=project_id, dataset_id=dataset_id)


def build_demo_queries(project_id: str, dataset_id: str) -> dict[str, dict[str, Any]]:
    view = allowed_bi_view(project_id, dataset_id)
    return {
        "¿Qué listados debería revisar primero el comité?": {
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
            "business_intent": "Clasifique las primeras cotizaciones que debe revisar el comité de inversiones.",
            "expected_output": "Cola de prioridad a nivel de listado con señales de modelo e indicadores de calidad.",
            "chart_type": "table",
            "confidence": "high",
            "limitations": ["La demostración SQL utiliza la vista gobernada BI y no incluye economía de simulación de cartera local."],
        },
        "¿Qué tipos de combustible concentran las mayores oportunidades de precios?": {
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
            "business_intent": "Identifique las categorías de combustible donde se concentran las oportunidades por debajo de expected-price.",
            "expected_output": "Agregación de tipos de combustible por recuento de oportunidades, brecha y señal top-price.",
            "chart_type": "bar",
            "confidence": "high",
            "limitations": ["Los promedios pueden ocultar problemas de calidad a nivel de listado."],
        },
        "¿Dónde coinciden las señales de regresión y clasificación?": {
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
            "business_intent": "Compare las señales de regresión de diferencia de precios con la probabilidad de clasificación top-price.",
            "expected_output": "Grupos de indicadores de decisión que muestran ambas señales del modelo una al lado de la otra.",
            "chart_type": "scatter",
            "confidence": "high",
            "limitations": ["El acuerdo se interpreta a partir de promedios agregados, no de una prueba estadística."],
        },
        "¿Qué oportunidades tienen indicadores de calidad o de riesgo lógico?": {
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
            "business_intent": "Descubra listados prometedores que aún necesitan una revisión de calidad o de riesgo lógico.",
            "expected_output": "Cola de oportunidades a nivel de listado con indicadores de riesgo visibles.",
            "chart_type": "table",
            "confidence": "high",
            "limitations": ["Las banderas de calidad explican el riesgo, no el rechazo final de la adquisición."],
        },
        "¿Cuántos listados caen en cada indicador de decisión?": {
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
            "business_intent": "Resuma la cartera según el indicador de decisión gobernado BI.",
            "expected_output": "Distribución de banderas de decisión con precio promedio y señales de modelo.",
            "chart_type": "bar",
            "confidence": "high",
            "limitations": ["Esto resume solo las señales de vista BI, antes de los filtros de estrategia locales Streamlit."],
        },
    }


def build_semantic_layer(project_id: str, dataset_id: str) -> str:
    field_lines = []
    for column in REQUIRED_COLUMNS:
        meaning = BI_FIELD_MEANINGS.get(column, "campo permitido de la vista BI")
        field_lines.append(f"- {column}: {meaning}")
    flag_lines = [f"- {flag}: {meaning}" for flag, meaning in DECISION_FLAG_MEANINGS.items()]
    return f"""
Superficie de consulta BI gobernada:
Tabla permitida: {allowed_bi_view(project_id, dataset_id)}

Campos permitidos:
{chr(10).join(field_lines)}

Significado de `decision_flag`:
{chr(10).join(flag_lines)}

Los flags de calidad y riesgo no deben ignorarse al hacer recomendaciones:
price_outlier_iqr, mileage_outlier_iqr, power_outlier_iqr, logical_issue.
"""


def build_sql_generation_prompt(question: str, project_id: str, dataset_id: str) -> str:
    return f"""
Estás generando BigQuery Standard SQL para un asistente BI de sólo lectura.
Esta es una capa conversacional de soporte a la decisión sobre un producto de datos BI gobernado.
El LLM escribe SQL candidato. La aplicación controla la validación. BigQuery controla los datos. El usuario toma la decisión.

{build_semantic_layer(project_id, dataset_id)}

Devuelve sólo JSON con este schema:
{{
  "sql": "SELECT ...",
  "business_intent": "explicación breve de la pregunta de negocio del usuario",
  "expected_output": "qué se espera que devuelva la consulta",
  "chart_type": "table|bar|scatter|line|none",
  "confidence": "high|medium|low",
  "limitations": ["limitación 1", "limitación 2"]
}}

Reglas:
- Devuelve sólo JSON.
- Genera una única sentencia SQL.
- Genera sólo consultas SELECT.
- No generes nunca INSERT, UPDATE, DELETE, MERGE, CREATE, DROP, ALTER, TRUNCATE, GRANT, REVOKE, CALL ni EXECUTE.
- No consultes INFORMATION_SCHEMA.
- No incluyas comentarios SQL.
- No uses SELECT *.
- No inventes columnas.
- Usa sólo las columnas listadas en la capa semántica.
- Prefiere agregaciones para preguntas amplias.
- Para resultados a nivel de listing, incluye siempre LIMIT 100 o menos.
- Usa sólo alias snake_case válidos en BigQuery. No uses espacios, puntuación, paréntesis ni backticks en alias.
- Para "mejores oportunidades", prioriza decision_flag = 'high_priority_review', expected_price_gap_eur < 0,
  top_price_probability alto y sin logical_issue cuando sea relevante.
- Si la pregunta es ambigua, haz una suposición conservadora y declárala en las limitaciones JSON.
- No respondas con SQL que requiera campos calculados localmente por Streamlit como expected_total_profit,
  expected_roi, vehicle_margin, finance_margin, cross_sell_income o investment_score.

Pregunta del usuario:
{question}
"""


def parse_gemini_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not text or not text.strip():
        return None, "Gemini devolvió una respuesta vacía."

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start == -1:
        return None, "La respuesta Gemini no contenía un objeto JSON."

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
        return None, "La respuesta Gemini contenía JSON incompleto."

    try:
        parsed = json.loads(cleaned[start:end])
    except json.JSONDecodeError as exc:
        return None, f"No se pudo parsear el JSON de Gemini: {exc.msg}."
    if not isinstance(parsed, dict):
        return None, "Gemini JSON debe ser un objeto."
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
        return {"is_valid": False, "errors": ["SQL está vacío."], "warnings": warnings, "normalized_sql": ""}

    if re.search(r"(--|/\*|\*/)", normalized_sql):
        errors.append("No se permiten comentarios en el SQL generado.")

    if normalized_sql.count(";") > 1 or ";" in normalized_sql.rstrip(";"):
        errors.append("No se permiten varias declaraciones SQL o punto y coma seguidas de más texto.")
    normalized_sql = normalized_sql.rstrip().rstrip(";").strip()

    if not re.match(r"(?is)^\s*(SELECT|WITH)\b", normalized_sql):
        errors.append("Sólo se permiten consultas SELECT o CON.")

    forbidden = r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CALL|EXECUTE)\b"
    if re.search(forbidden, normalized_sql, flags=re.IGNORECASE):
        errors.append("SQL contiene una palabra clave de escritura, DDL, permiso o ejecución prohibida.")

    if re.search(r"\bINFORMATION_SCHEMA\b", normalized_sql, flags=re.IGNORECASE):
        errors.append("Las consultas INFORMACIÓN_SCHEMA no están permitidas.")

    select_star_pattern = (
        r"(?is)(\bSELECT\s+(?:`?[A-Za-z_][A-Za-z0-9_]*`?\.)?\*"
        r"|,\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\.)?\*)"
    )
    if re.search(select_star_pattern, normalized_sql):
        errors.append("SELECCIONAR * no está permitido. Seleccione campos BI explícitos.")

    allowed_ref = f"{project_id}.{dataset_id}.vw_bi_dashboard"
    allowed_quoted = allowed_bi_view(project_id, dataset_id)

    backtick_refs = re.findall(r"`([^`]+)`", normalized_sql)
    for ref in backtick_refs:
        if "." in ref and ref != allowed_ref:
            errors.append(f"No se permite una referencia de tabla externa: `{ref}`.")

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
        errors.append(f"Sílo se puede consultar la vista BI gobernada. Referencia de tabla inesperada: {clean_ref}.")

    if not found_allowed_view:
        errors.append("La consulta debe leerse desde la vista del panel gobernado BI.")

    has_group_by = bool(re.search(r"(?is)\bGROUP\s+BY\b", normalized_sql))
    has_limit = bool(re.search(r"(?is)\bLIMIT\s+\d+\b", normalized_sql))
    if not has_limit and not has_group_by:
        warnings.append("La consulta a nivel de listado no tenía LÍMITE. Se añadió LÍMITE 100.")
        normalized_sql = f"{normalized_sql}\nLIMIT 100"
    elif has_limit:
        limits = [int(value) for value in re.findall(r"(?is)\bLIMIT\s+(\d+)\b", normalized_sql)]
        if limits and max(limits) > 100 and not has_group_by:
            errors.append("Las consultas a nivel de listado deben usar LIMIT 100 o menos.")

    if chart_type == "scatter":
        warnings.append("Los gráficos de dispersión requieren al menos dos columnas de resultados numéricos.")
    elif chart_type in {"bar", "line"}:
        warnings.append(f"Los gráficos de tipo {chart_type} requieren una columna compatible de categoría/fecha y un valor numérico.")

    return {"is_valid": not errors, "errors": errors, "warnings": warnings, "normalized_sql": normalized_sql}


def bigquery_query_error_message(exc: Exception, stage: str) -> str:
    detail = getattr(exc, "message", "") or str(exc)
    if "bytesBilledLimitExceeded" in detail or "bytes facturados" in detail.lower():
        return (
            f"{stage} fue bloqueado por el límite de bytes facturados de BigQuery. "
            f"El límite actual es {MAX_BYTES_BILLED / (1024 * 1024):.0f} MB. "
            "Configura `BQ_MAX_BYTES_BILLED_MB=100` antes de iniciar Streamlit si quieres permitir esta consulta en el aula."
        )
    return f"{stage} falló porque BigQuery rechazó el SQL: {detail}"


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
            "Las credenciales de BigQuery no están configuradas. Ejecuta `gcloud auth application-default login` o configura credenciales de service account."
        )
        return None, metadata
    except Exception:
        metadata["errors"].append("No se pudo crear el cliente de BigQuery. Verifica las credenciales de Google Cloud y el acceso al proyecto.")
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
        metadata["errors"].append("Error en el ensayo porque las credenciales actuales no tienen el permiso BigQuery.")
        return None, metadata
    except GoogleAPIError as exc:
        metadata["errors"].append(bigquery_query_error_message(exc, "Dry-run"))
        return None, metadata
    except Exception:
        metadata["errors"].append("Falló el ensayo. Verifique la configuración SQL y BigQuery generada.")
        return None, metadata

    if metadata["estimated_bytes"] is not None and metadata["estimated_bytes"] > MAX_BYTES_BILLED:
        metadata["errors"].append(
            f"La estimación de la consulta es {metadata['estimated_mb']:.2f} MB, por encima del límite de aula de "
            f"{MAX_BYTES_BILLED / (1024 * 1024):.0f} MB."
        )
        return None, metadata

    execution_config = bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES_BILLED)
    try:
        df = client.query(sql, job_config=execution_config).to_dataframe()
        metadata["executed"] = True
        if df.empty:
            metadata["warnings"].append("La consulta se ejecutó correctamente pero no devolvió filas.")
        return df, metadata
    except BadRequest as exc:
        metadata["errors"].append(bigquery_query_error_message(exc, "Ejecución"))
    except Forbidden:
        metadata["errors"].append("La ejecución falló porque las credenciales actuales no tienen el permiso BigQuery.")
    except GoogleAPIError as exc:
        metadata["errors"].append(bigquery_query_error_message(exc, "Ejecución"))
    except Exception:
        metadata["errors"].append("La ejecución falló. Verifique las credenciales, consulte permisos y acceso a la red.")
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
                st.info("No había ninguna columna de fecha o año disponible para el gráfico de líneas sugerido.")
                return
            fig = px.line(df.sort_values(x_candidates[0]), x=x_candidates[0], y=numeric_columns[0])
        else:
            st.info("El tipo de gráfico sugerido no es compatible con las columnas devueltas.")
            return
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("La tabla de resultados se cargó, pero la representación del gráfico falló para estas columnas.")


def deterministic_ai_summary(df: pd.DataFrame, metadata: dict[str, Any]) -> str:
    if df is None or df.empty:
        return "No se devolvió ninguna fila, por lo que no hay ningún patrón empresarial que resumir a partir de esta consulta."

    summary = [f"- La consulta devolvió {len(df):,} filas y {len(df.columns):,} columnas."]
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        first_numeric = numeric_columns[0]
        top = df.sort_values(first_numeric, ascending=False).head(3)
        values = ", ".join(str(value) for value in top[first_numeric].tolist())
        summary.append(f"- Los principales valores observados de `{first_numeric}` incluyen: {values}.")
    else:
        first_column = df.columns[0]
        values = ", ".join(str(value) for value in df[first_column].head(3).tolist())
        summary.append(f"- Primeros valores devueltos de `{first_column}`: {values}.")
    if metadata.get("warnings"):
        summary.append(f"- Advertencia: {metadata['warnings'][0]}")
    else:
        summary.append("- Advertencia: resumen determinista de reserva; no se utilizó ninguna interpretación de IA.")
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
Escribe una interpretación para comité de inversión de un resultado de consulta BI gobernada.

Reglas:
- Máximo 3 bullets.
- Separa hechos de interpretación.
- No inventes números.
- Menciona limitaciones.
- Usa lenguaje de comité de inversión.

Pregunta original: {question}
Intención de negocio: {ai_payload.get("business_intent")}
SQL generado: {sql}
Forma del DataFrame: {df.shape if df is not None else None}
Primeras filas como registros: {records}
Limitaciones: {ai_payload.get("limitations", [])}
"""
    result, error = generate_gemini_content(prompt)
    if error:
        return deterministic_ai_summary(df, metadata)
    return result or deterministic_ai_summary(df, metadata)


def render_ai_sql_assistant_tab(project_id: str, dataset_id: str) -> None:
    st.title("Pregunta A Los Datos - Asistente GenAI SQL")
    st.caption("Preguntas en lenguaje natural traducidas a BigQuery SQL validado en la vista de soporte de decisiones BI.")
    st.info(
        "Este asistente usa Gemini para generar SQL, pero BigQuery sigue siendo la fuente de la verdad. "
        "Cada consulta generada por IA se valida y se verifica en seco antes de su ejecución."
    )
    st.caption("El LLM escribe el SQL candidato. La aplicación controla la validación. BigQuery controla los datos. El usuario toma la decisión.")

    demo_mode = not gemini_available()
    demo_queries = build_demo_queries(project_id, dataset_id)
    if demo_mode:
        reason = gemini_unavailable_message() or "Gemini no está disponible."
        st.warning(f"Modo demo: usando SQL predefinido. {reason}")
        button_prompts = DEMO_QUESTIONS
    else:
        button_prompts = AI_EXAMPLE_PROMPTS

    st.session_state.setdefault("ai_sql_history", [])
    st.session_state.setdefault("ai_sql_question", "")

    st.write("Preguntas de ejemplo")
    button_columns = st.columns(2)
    for index, prompt in enumerate(button_prompts):
        if button_columns[index % 2].button(prompt, key=f"ai_prompt_{index}"):
            st.session_state["ai_sql_question"] = prompt
            st.session_state["ai_sql_pending_question"] = prompt

    with st.form("ai_sql_question_form"):
        question = st.text_input("Haga una pregunta de negocios", value=st.session_state.get("ai_sql_question", ""))
        submitted = st.form_submit_button("Preguntar a los datos")

    if submitted:
        st.session_state["ai_sql_question"] = question
        st.session_state["ai_sql_pending_question"] = question

    question = str(st.session_state.pop("ai_sql_pending_question", "")).strip()
    if not question:
        if submitted:
            st.warning("Ingrese una pregunta o seleccione un ejemplo.")
        return

    if demo_mode:
        ai_payload = demo_queries.get(question)
        if ai_payload is None:
            st.error("El modo de demostración solo puede ejecutar los botones de preguntas predefinidos. Configura Gemini para hacer preguntas personalizadas.")
            return
    else:
        with st.spinner("Solicitando a Gemini que genere SQL gobernado..."):
            ai_payload, generation_error = generate_sql_with_gemini(question, project_id, dataset_id)
        if generation_error:
            st.error(generation_error)
            st.info("Pruebe uno de los ejemplos o simplifique la pregunta.")
            return
        if ai_payload is None:
            st.error("Gemini no devolvió una carga útil SQL utilizable.")
            return

    sql = str(ai_payload.get("sql", "")).strip()
    sql, comments_removed = strip_sql_comments(sql)
    sql, aliases_sanitized = sanitize_sql_aliases(sql)
    chart_type = str(ai_payload.get("chart_type", "table")).lower()
    if chart_type not in {"table", "bar", "scatter", "line", "none"}:
        chart_type = "table"
    validation = validate_sql(sql, project_id, dataset_id, chart_type)
    if comments_removed:
        validation["warnings"].append("Gemini incluyó comentarios SQL; fueron eliminados antes de la validación.")
    if aliases_sanitized:
        validation["warnings"].append("Gemini incluyó alias de estilo de visualización; se convirtieron en alias de casos de serpiente BigQuery seguros.")

    st.subheader("Intención comercial")
    st.write(ai_payload.get("business_intent", "No se proporcionó ninguna intención comercial."))
    st.caption(ai_payload.get("expected_output", "No se proporcionó ningún resultado esperado."))

    status_col, scan_col = st.columns(2)
    with status_col:
        if validation["is_valid"]:
            st.success("Validación aprobada.")
        else:
            st.error("La validación bloqueó este SQL.")
        for error in validation["errors"]:
            st.error(error)
        for warning in validation["warnings"]:
            st.warning(warning)

    with st.expander("SQL generado", expanded=True):
        st.code(validation["normalized_sql"] or sql, language="sql")

    if not validation["is_valid"]:
        st.info("No se intentó realizar ningún ensayo o ejecución BigQuery porque falló la validación.")
        return

    with st.spinner("Ejecutando el ensayo BigQuery y ejecutándolo dentro del límite de escaneo del aula..."):
        result_df, metadata = run_validated_query(validation["normalized_sql"], project_id)

    with scan_col:
        if metadata.get("estimated_mb") is None:
            st.metric("Escaneo estimado BigQuery", "No disponible")
        else:
            st.metric("Escaneo estimado BigQuery", f"{metadata['estimated_mb']:.2f} MB")
        st.caption(f"Máximo de bytes facturados en aula: {MAX_BYTES_BILLED / (1024 * 1024):.0f} MB")

    for error in metadata["errors"]:
        st.error(error)
    for warning in metadata["warnings"]:
        st.warning(warning)
    if metadata["errors"] or result_df is None:
        st.info("Ajuste la pregunta o las credenciales y luego ejecute el asistente nuevamente.")
        return

    st.subheader("Tabla de resultados")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.subheader("Gráfico sugerido")
    render_ai_chart(result_df, chart_type)

    st.subheader("Interpretación ejecutiva")
    st.markdown(executive_summary(question, ai_payload, validation["normalized_sql"], result_df, metadata))

    limitations = ai_payload.get("limitations", [])
    if limitations:
        with st.expander("Advertencias y limitaciones", expanded=False):
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
        with st.expander("Historial de sesiones", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state["ai_sql_history"]).tail(10), use_container_width=True, hide_index=True)


def main() -> None:
    config = load_project_config()
    project_id = config.get("gcp_project_id")
    dataset_id = config.get("bq_dataset")
    if not project_id or not dataset_id:
        st.error("Falta la configuración del proyecto `gcp_project_id` o `bq_dataset` en config/project_config.yaml.")
        st.stop()

    df_raw, load_error = load_dashboard_data(project_id, dataset_id)
    if load_error:
        st.error(load_error)
        st.stop()
    if df_raw is None or df_raw.empty:
        st.error("La vista del panel BI no devolvió filas. Ejecute la canalización ascendente antes de abrir el panel.")
        st.stop()

    missing_columns = validate_columns(df_raw)
    if missing_columns:
        st.error("A la vista BI le faltan las columnas obligatorias: " + ", ".join(missing_columns))
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
            "Dashboard Del Comité",
            "Estrategia Y Constructor De Portfolio",
            "Pregunta A Los Datos - Asistente GenAI SQL",
        ]
    )

    with tab_committee:
        st.title("Simulador De Inversión En Portfolio De Vehículos")
        st.caption(
            "Estrategia de inversión, selección de cartera, economía financiera y valor de venta cruzada para un negocio de financiación al consumo."
        )
        st.markdown(
            "**Pregunta central de negocio:** con un presupuesto de inversión fijo, ¿qué portfolio de vehículos maximiza el valor esperado de negocio mediante margen de reventa, margen de financiación y valor de relación con el cliente?"
        )
        st.info(
            "La regresión predice el precio de mercado esperado. La clasificación predice el atractivo comercial. Streamlit combina las salidas del modelo con supuestos de negocio. Esto es soporte a la decisión, no aprobación automática de adquisición, crédito, compliance o riesgo."
        )

        st.subheader("Descripción general de la estrategia seleccionada")
        st.markdown(overview)
        if metadata["apr_floor_applied"]:
            st.warning("APR la protección del piso está activa porque, de lo contrario, los descuentos reducirían APR por debajo del margen requerido.")

        st.subheader("KPI ejecutivos")
        render_kpis(summary, budget_params)

        st.subheader("Decisión del Comité")
        render_committee_decision(summary, risk_warnings)

        st.subheader("Gráfico de componentes de ganancias")
        component_chart(summary)
        st.caption(
            "Los componentes de beneficio concilian con el beneficio total esperado. El diferencial de reventa y los ingresos se muestran como valores positivos; los costes se muestran como valores negativos."
        )

        st.subheader("Riesgo y confianza")
        render_risk_confidence(risk_metrics, risk_warnings)

        st.subheader("Recomendación actual")
        st.markdown(deterministic_recommendation(summary, warnings))
        for warning in [item for item in warnings if item not in risk_warnings]:
            st.warning(warning)

        st.subheader("Gemini Memorándum del Comité")
        if st.button("Generar memorando del comité"):
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
        st.subheader("Resumen de controles estratégicos")
        st.write(
            {
                "Estrategia del vehículo": vehicle_preset,
                "Estrategia de precios y venta cruzada": pricing_preset,
                "Anulaciones manuales activas": customized,
            }
        )

        st.subheader("Recuentos de filtrado antes y después")
        count_cols = st.columns(5)
        count_cols[0].metric("Universo total de vehículos", f"{counts['total_universe']:,}")
        count_cols[1].metric("Elegible después de los filtros", f"{counts['eligible_after_filters']:,}")
        count_cols[2].metric("Vehículos seleccionados", f"{summary['selected_count']:,}")
        count_cols[3].metric("Excluido por filtros", f"{counts['excluded_by_filters']:,}")
        count_cols[4].metric("Se excluyen los resultados del modelo que faltan", f"{counts['excluded_by_missing_model_outputs']:,}")
        st.caption(f"Excluidos por flags de calidad: {counts['excluded_by_quality_flags']:,}")

        st.subheader("Resumen de economía de cartera")
        econ_cols = st.columns(4)
        econ_cols[0].metric("Capital desplegado", fmt_eur(summary["capital_deployed"]))
        econ_cols[1].metric("Presupuesto restante", fmt_eur(summary["remaining_budget"]))
        econ_cols[2].metric("Beneficio total esperado", fmt_eur(summary["expected_total_profit"]))
        econ_cols[3].metric("ROI esperado", fmt_pct(summary["expected_roi"]))
        econ_cols_2 = st.columns(3)
        econ_cols_2[0].metric("Margen del vehículo", fmt_eur(summary["vehicle_margin"]))
        econ_cols_2[1].metric("Margen de financiación", fmt_eur(summary["finance_margin"]))
        econ_cols_2[2].metric("Ingresos por venta cruzada", fmt_eur(summary["cross_sell_income"]))

        st.subheader("Mapa de oportunidades de cartera")
        scatter_chart(eligible_with_actions)

        st.subheader("Portafolio Seleccionado")
        table = selected_table(selected_with_actions).sort_values("Puntuación de inversión", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar portafolio seleccionado CSV",
            table.to_csv(index=False),
            file_name="selected_vehicle_portfolio.csv",
            mime="text/csv",
        )
        with st.expander("Descargar universo elegible completo"):
            st.download_button(
                "Descargar universo elegible CSV",
                eligible_with_actions.to_csv(index=False),
                file_name="eligible_vehicle_universe.csv",
                mime="text/csv",
            )

        st.subheader("Notas comerciales")
        st.markdown(
            "\n- La cartera seleccionada bajo el mandato actual es el resultado de la estrategia y los supuestos actuales.\n- Cambiar filtros cambia el mandato de inversión.\n- Cambiar los supuestos de precios y ventas cruzadas cambia la estrategia de monetización.\n- El modelo proporciona señales de decisión; la estrategia de negocio define la cartera.\n"
        )

    with tab_ai:
        render_ai_sql_assistant_tab(project_id, dataset_id)


if __name__ == "__main__":
    main()
