# Head Of Data 101 - Repositorio Base En Español

Este repositorio es la **réplica en castellano** del baseline docente de Head Of Data 101.
Mantiene la misma base de datos, las mismas claves de configuración, los mismos datos y los mismos contratos BI que el repositorio original.

El objetivo es que los alumnos puedan recorrer el flujo completo de forma clara, notebook-first y sin capas innecesarias de arquitectura.

No es un proyecto de producción. Está diseñado para enseñar el ciclo completo de un producto de datos de forma legible y revisable.

## Caso De Negocio

El curso simula una unidad de datos dentro de un banco retail / consumer finance que evalúa oportunidades de adquisición de vehículos usados para reventa y financiación.

El baseline mantiene esta narrativa:

- el precio real viene de los datos scrapeados del marketplace
- la regresión estima `expected_price`
- la clasificación estima la señal/probabilidad externa `top_price`
- BI combina precio real, brecha frente a `expected_price`, probabilidad `top_price` y supuestos de negocio
- Streamlit ofrece la demo final de soporte a la decisión

El modelo no decide la estrategia. Ayuda al comité a comparar estrategias con datos.

## Qué Cubre Este Repo

El pipeline obligatorio cubre:

1. adquisición de datos mediante scraping
2. preprocesamiento y controles de calidad
3. tablas de warehouse en BigQuery
4. vistas SQL analíticas para ML y BI
5. modelo de regresión para `expected_price`
6. modelo de clasificación para `top_price`
7. vista BI-ready de soporte a la decisión

El dashboard de Streamlit es opcional para ejecutar el pipeline, pero recomendado como demo final cuando el warehouse y las tablas de predicción ya existen.

## Orden Obligatorio De Ejecución

Ejecuta los notebooks principales en este orden:

1. `notebooks/01_scraping_audi_a3_germany.ipynb`
2. `notebooks/02_preprocessing_audi_a3_germany.ipynb`
3. `notebooks/03_sqlqueries_audi_a3_germany.ipynb`
4. `notebooks/04_regression_audi_a3_germany.ipynb`
5. `notebooks/05_classification_audi_a3_germany.ipynb`

Estos notebooks son el único set final obligatorio del baseline. No hay separación entre versión de clase y versión completa.

## Laboratorios Opcionales

Los notebooks opcionales están separados del pipeline obligatorio:

- `notebooks/01b_raw_data_eda_before_preprocessing_audi_a3_germany.ipynb`:
  apoyo de Session 03 para inspeccionar datos raw antes del preprocesamiento. No guarda salidas limpias ni sustituye al Notebook 02.

- `notebooks/04b_regression_challenge_lab_audi_a3_germany.ipynb`:
  laboratorio de regresión de Session 06 desde CSV procesados, sin BigQuery. No sustituye al Notebook 04.

- `notebooks/05b_classification_challenge_lab_audi_a3_germany.ipynb`:
  laboratorio de clasificación de Session 07 desde CSV procesados, sin escrituras en BigQuery. No sustituye al Notebook 05.

## Activos SQL

SQL está ordenado explícitamente en `sql/` para ejecución en clase:

1. `00_create_dataset.sql`
2. `01_create_staging.sql`
3. `02_build_dimensions.sql`
4. `03_build_fact.sql`
5. `04_vw_regression_dataset.sql`
6. `05_vw_classification_dataset.sql`
7. `06_vw_bi_dashboard.sql`

La carpeta SQL define el dataset de BigQuery, staging, dimensiones, tabla de hechos, vistas para ML, tablas de predicción y vista de dashboard BI. Conserva estos nombres porque los notebooks y BI dependen de ellos.

## BI / Streamlit

La app de Streamlit en `bi/` es la capa final de soporte a la decisión. Lee la vista gobernada de BigQuery `vw_bi_dashboard`, combina señales de modelo con supuestos editables de negocio y ayuda a comparar estrategias de portfolio.

No es un motor de aprobación y no sustituye el criterio del comité.

Consulta `bi/README.md` para la configuración específica del dashboard y notas de demo.

## Funcionalidad Opcional Con Gemini

Las funciones de Gemini son opcionales. El dashboard de Streamlit puede usar Gemini para generar el memo de comité y el asistente GenAI SQL cuando:

- `google-genai` está instalado
- `GEMINI_API_KEY` o `GOOGLE_API_KEY` está definida en el entorno del proceso

Configura una clave localmente en PowerShell con una de estas opciones:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

o:

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

`GEMINI_API_KEY` tiene prioridad sobre `GOOGLE_API_KEY`. Si no hay ninguna clave, el dashboard sigue funcionando con comportamiento determinista/default. Las credenciales de Gemini no se leen desde `.streamlit/secrets.toml`, secretos de Streamlit Cloud, `.env` ni archivos de credenciales versionados.

## Instalación

Crea y activa un entorno Python, y después instala las dependencias mínimas del curso:

```bash
pip install -r requirements_min.in
```

Para notebooks y Streamlit respaldados por BigQuery, autentícate con Google Cloud en tu entorno local:

```bash
gcloud auth application-default login
```

La configuración por defecto vive en:

- `config/project_config.yaml`

## Ejecutar El Baseline

1. Revisa `config/project_config.yaml`.
2. Ejecuta los cinco notebooks obligatorios en orden.
3. Ejecuta los archivos SQL en el orden indicado arriba, incluyendo la carga del CSV procesado en `stg_listings_clean`.
4. Confirma que el Notebook 04 escribe `fact_expected_price_predictions`.
5. Confirma que el Notebook 05 escribe `fact_top_price_predictions`.
6. Opcionalmente ejecuta el dashboard de soporte a la decisión:

```bash
streamlit run bi/streamlit_app.py
```

## Qué Esperar

Espera un baseline docente legible que conserva el flujo completo del curso y el contrato de datos BI-ready.

No esperes orquestación de producción, CI/CD, Docker, estructura de paquete, MLOps avanzado ni despliegue cloud endurecido. Son extensiones válidas, pero quedan fuera de este repositorio base.
