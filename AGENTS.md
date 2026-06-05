# Reglas AGENTS Para Este Repositorio

## Alcance

Estas reglas aplican a ediciones asistidas por IA en este repositorio docente.

## Reglas Principales

- No sobrediseñar el código.
- Mantener los notebooks simples, legibles y educativos.
- Mantener los notebooks como la interfaz principal de enseñanza.
- Preservar los contratos BI-ready y los nombres de tablas/vistas de salida.
- Preservar un único set final de notebooks.
- No reintroducir duplicación de notebooks de clase/completos.

## Guardarraíles Narrativos

- Mantener la historia prevista:
  - la regresión predice `expected_price`
  - la clasificación predice `top_price`
  - BI combina precio real, brecha frente a `expected_price` y salidas de `top_price`
- No desviarse de la narrativa `expected_price` + `top_price`.
- No reintroducir la antigua historia de bargain/clasificación desde predicción.

## Límites De Refactor

- Preferir cambios mínimos y enseñables frente a rediseños arquitectónicos.
- Evitar helpers ocultos, packaging complejo o abstracciones innecesarias.
- Mantener la documentación en castellano.
- Mantener los activos SQL ordenados y explícitos para uso en clase.
