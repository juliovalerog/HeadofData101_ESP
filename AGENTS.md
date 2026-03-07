# AGENTS Rules For This Repository

## Scope

These rules apply to AI-assisted edits in this teaching repository.

## Core Rules

- Do not overengineer the codebase.
- Keep notebooks simple, readable, and educational.
- Keep notebooks as the main teaching interface.
- Preserve BI-ready contracts and output table/view names.
- Preserve one single final notebook set only.
- Do not reintroduce class/full notebook duplication.

## Narrative Guardrails

- Keep the intended storyline:
  - regression predicts `expected_price`
  - classification predicts `top_price`
  - BI combines actual price, expected-price gap, and top-price outputs
- Do not drift from the `expected_price` + `top_price` narrative.
- Do not reintroduce the old bargain/classification-from-prediction storyline.

## Refactor Boundaries

- Prefer minimal, teachable changes over architectural redesign.
- Avoid hidden helpers, complex packaging, or unnecessary abstractions.
- Keep documentation in English.
- Keep SQL assets ordered and explicit for classroom use.
