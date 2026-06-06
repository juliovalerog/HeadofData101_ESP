# Dashboard BI En Streamlit

## Propósito

Este dashboard es la capa final de decisión de negocio del curso. Convierte las salidas de los modelos previos en un simulador ejecutivo de inversión para un negocio de consumer finance que evalúa oportunidades de adquisición de vehículos usados.

El dashboard es opcional para ejecutar el baseline, pero recomendado como demo final de soporte a la decisión cuando el warehouse y las tablas de salida de modelo ya están pobladas.

Está diseñado para un comité de inversión, comité de portfolio o unidad de financiación de vehículos. No es un notebook técnico, ni un motor de aprobación en producción, ni una herramienta automática de decisión de compra.

## Pregunta De Negocio

Con un presupuesto fijo de inversión, ¿qué portfolio de vehículos debería adquirir y revender el negocio mediante operaciones financiadas, considerando margen de reventa, margen financiero, ingresos de venta cruzada, valor de fidelización, riesgo y coherencia de campaña comercial?

## Fuente De Datos

Fuente principal:

- vista BigQuery: `vw_bi_dashboard`

La ruta por defecto de la app requiere acceso a BigQuery. No usa un dataset mock empaquetado.

La app lee `gcp_project_id` y `bq_dataset` desde:

- `config/project_config.yaml`

El pipeline previo proporciona:

- salida de regresión: `expected_price_eur`
- salida de clasificación: score de atractivo comercial desde `top_price_probability`
- vista BI: perfil del listing, precio real, brecha frente a `expected_price`, señal de atractivo comercial y flag de decisión

Las métricas específicas de inversión, financiación, venta cruzada, riesgo y selección de portfolio se calculan dentro de Streamlit.

## Cómo Ejecutarlo

Instala las dependencias mínimas:

```bash
pip install -r requirements_min.in
```

Ejecuta el dashboard:

```bash
streamlit run bi/streamlit_app.py
```

## Autenticación BigQuery

La ruta por defecto carga datos desde BigQuery. Autentícate antes de ejecutar la app:

```bash
gcloud auth application-default login
```

También puedes configurar credenciales de service account en el entorno de ejecución usando las variables estándar de autenticación de Google Cloud.

Si faltan credenciales o no son válidas, la app muestra un error legible de negocio en lugar de un traceback crudo.

El asistente GenAI SQL hace dry-run de cada consulta y aplica un límite docente de bytes facturados. El valor por defecto es 100 MB. Si BigQuery informa de que una consulta sobre la vista gobernada requiere un mínimo mayor, lanza Streamlit con un límite local más alto:

```powershell
$env:BQ_MAX_BYTES_BILLED_MB="200"
streamlit run bi/streamlit_app.py
```

## Clave API De Gemini

Gemini es opcional. El memo de comité y el asistente GenAI SQL usan el paquete `google-genai` cuando está instalado y hay una clave API disponible.

Este repo lee credenciales de Gemini sólo desde variables de entorno. No lee claves de Gemini desde `.streamlit/secrets.toml`, secretos de Streamlit Cloud, `.env` ni ningún archivo de credenciales versionado.

En PowerShell, configura una de estas variables:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

o:

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

`GEMINI_API_KEY` tiene prioridad sobre `GOOGLE_API_KEY`. Si no está definida ninguna variable, el dashboard sigue funcionando con comportamiento determinista/default. La app no crea, imprime, registra, hardcodea ni commitea claves API.

## Estructura Del Dashboard

La app tiene tres pestañas:

1. `Dashboard Del Comité`
2. `Estrategia Y Constructor De Portfolio`
3. `Pregunta A Los Datos - Asistente GenAI SQL`

La primera pestaña es la pantalla ejecutiva de decisión. La segunda es la pantalla interactiva para ajustar la estrategia y construir el portfolio. La tercera es un asistente text-to-SQL gobernado sobre la vista BI del dashboard.

## Presets De Estrategia De Vehículo

Los presets de estrategia de vehículo definen el mandato de inversión. No cambian las salidas de los modelos.

- `Mercado amplio`: descubrimiento amplio de oportunidades con filtrado mínimo.
- `Núcleo joven con bajo kilometraje`: portfolio más limpio y fácil de vender con filtros más estrictos de edad y kilometraje.
- `Campaña retail generalista`: campaña coherente alrededor de vehículos generalistas y fáciles de explicar.
- `Margen en vehículos de mayor precio`: mayor beneficio unitario con más riesgo de concentración de capital.
- `Riesgo conservador`: defensibilidad por encima del volumen, con umbrales más fuertes de calidad y señal de modelo.

Después de seleccionar un preset, todos los supuestos siguen siendo editables en la barra lateral.

## Presets De Pricing Y Venta Cruzada

Los presets de pricing y venta cruzada definen la estrategia de monetización.

- `Caso base`: supuestos equilibrados por defecto.
- `Venta cruzada agresiva`: economía de bundle más fuerte y mayores tasas de contratación.
- `Foco en margen financiero`: mayor margen directo de financiación con menor énfasis en venta cruzada.
- `Foco en fidelización`: mayor valor de relación a largo plazo y descuentos APR más generosos.
- `Escenario de estrés`: supuestos adversos de reventa, funding, riesgo y comerciales.

Los cambios manuales se detectan y se muestran en la barra lateral y en el resumen de estrategia.

## Cálculos Principales De Negocio

Para cada vehículo elegible, la app estima:

- descuento esperado frente al precio de mercado predicho por el modelo
- precio de reventa conservador
- margen de reventa del vehículo
- capital desplegado
- margen financiero tras descuentos APR ponderados
- valor de seguro, tarjeta de combustible, nómina y fidelización
- coste de funding de inventario
- beneficio total esperado
- ROI esperado
- proxy de velocidad de reventa
- peso de encaje en portfolio
- score de inversión

El portfolio recomendado bajo la estrategia actual se selecciona con una regla de ranking transparente. Los vehículos se ordenan por score de inversión y se seleccionan hasta alcanzar el presupuesto, el buffer de caja o el número máximo de vehículos. No es un modelo de optimización matemática completo.

## Memo De Comité Con Gemini

El memo opcional de Gemini recibe sólo contexto agregado de estrategia, supuestos, avisos y los principales candidatos seleccionados. No envía datos raw innecesarios.

El memo incluye:

1. estrategia seleccionada y mandato de inversión
2. recomendación ejecutiva
3. racional del portfolio
4. supuestos clave
5. resultado financiero esperado
6. lógica de venta cruzada y fidelización
7. principales riesgos y limitaciones de modelo
8. siguientes pasos sugeridos de validación

## Demo En Vivo De 10 Minutos

1. Empieza con `Caso base` y `Mercado amplio`.
2. Explica la caja de decisión del comité.
3. Cambia a `Venta cruzada agresiva` y muestra cómo cambian los drivers de valor.
4. Cambia a `Riesgo conservador` y muestra cómo se reduce el universo elegible.
5. Ajusta el presupuesto de inversión y muestra cómo cambia el portfolio seleccionado.
6. Modifica filtros de edad y kilometraje y explica el mandato de inversión.
7. Genera el memo de comité con Gemini.
8. Concluye: "El modelo no decide la estrategia; permite al comité comparar estrategias con datos."

No expliques cada fórmula durante la demo. Enfócate en cómo los cambios de estrategia de negocio cambian el universo elegible, el portfolio seleccionado, los drivers de valor y la recomendación del comité.

## Mensaje Docente

- El dashboard es la capa final de decisión del pipeline.
- La regresión crea el precio de mercado esperado.
- La clasificación crea el atractivo comercial.
- Los supuestos de negocio crean la estrategia de inversión.
- BI convierte todo esto en un producto de decisión listo para comité.

El modelo no decide la estrategia; permite al comité comparar estrategias con datos.

## Disclaimer De Soporte A La Decisión

Esta herramienta simula soporte a la decisión. No es una aprobación final de adquisición, crédito, compliance o riesgo.

Las salidas de regresión y clasificación vienen del pipeline previo del curso. Streamlit combina esas señales de modelo con supuestos editables de negocio para que el comité pueda discutir estrategia, economía y coherencia de portfolio.
