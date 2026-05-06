# model-explainability-lite

Standalone post-training explainability for XGBoost binary classifiers. Decoupled from any training pipeline — consumes only approved model artifacts and point-in-time scoring data.

**Produces:**
- **Global explainability** — feature importance overall and by segment
- **Local explainability** — per-record top-N reason codes with business-readable text
- **Audit artifacts** — run manifests, validation checks, SHA-256 file hashes

---

## Requirements

Python 3.10+, XGBoost ≥ 1.7

```bash
pip install xgboost shap pandas polars numpy matplotlib openpyxl pyarrow
```

---

## Quick Start

```bash
# Generate a blank metadata template (or use the included demo)
python -m model_explainability_lite --create-template metadata/explainability_metadata.xlsx

# Edit configs/explainability_config.yaml to point at your model and data

# Run
python -m model_explainability_lite --config configs/explainability_config.yaml
```

A demo metadata file (`metadata/explainability_metadata.xlsx`) with 6 example features is included so you can test immediately with a toy model.

---

## Inputs

| Input | Format | Notes |
|---|---|---|
| Model artifact | `.ubj` / `.json` / `.bin` / `.model` / `.pkl` | XGBoost Booster or sklearn wrapper |
| Scoring dataset | `.parquet` / `.csv` | ID column + model features + optional labels/segments |
| Feature list | `.json` (`feature_names` key) | Optional; falls back to model-embedded names |
| Metadata Excel | `.xlsx` | Optional; enables reason text, typologies, score bands |
| Config YAML | `.yaml` | Controls all paths, sampling, and scoring settings |

---

## Configuration

Key fields in `configs/explainability_config.yaml`:

```yaml
paths:
  model_file: artifacts/my_model.ubj
  scoring_data_file: data/oot_dataset.parquet
  metadata_excel_file: metadata/explainability_metadata.xlsx
  output_root: outputs

dataset:
  id_column: transaction_id
  label_column: label             # null if unavailable
  segment_columns: [channel, customer_tier]

sampling:
  global_sample_size: 50000       # max rows for global SHAP
  local_max_rows: 10000
  shap_chunk_size: 20000          # tune to available memory

scoring:
  score_scale: "0_1000"           # or "0_100"
  top_n_reasons: 5
```

---

## Metadata Excel

Optional but recommended. Generate a blank template with `--create-template`, then fill these tabs:

| Tab | Key columns |
|---|---|
| `feature_dictionary` | `feature_name`, `feature_business_name`, `description` |
| `reason_mapping` | `feature_name`, `reason_code`, `business_explanation_template`, `positive/negative_risk_direction` |
| `typology_mapping` | `feature_name`, `typology_id`, `typology_name` |
| `feature_grouping` | `feature_name`, `group_name`, `domain` |
| `thresholds_bands` | `score_min`, `score_max`, `band`, `operational_meaning` |
| `report_config` | `parameter`, `value` |

---

## Outputs

Written to `<output_root>/<run_id>/`:

| File | Description |
|---|---|
| `scored_transactions.parquet` | All rows: probability, score, band, threshold flag |
| `local_explanations.parquet` | Per-record top-N reason codes with business text |
| `global_feature_importance.parquet` | Mean \|SHAP\| per feature, ranked |
| `global_feature_importance_by_segment.parquet` | Importance broken out by segment |
| `reports/explainability_summary.xlsx` | Excel report: global importance + local samples |
| `validation_checks.csv` | All checks with PASS/WARN status |
| `run_manifest.json` | File hashes, config snapshot, sample sizes |
| `plots/global_importance.png` | Top-20 importance bar chart |
| `logs/explainability.log` | Execution log |

---

## Tests

```bash
pytest tests/ -v
```

Builds a toy XGBoost model and verifies the full pipeline end-to-end.

---

## Architecture

```
model_explainability_lite/
├── config.py                # ExplainabilityConfig (YAML-backed dataclasses)
├── service.py               # ExplainabilityService — main orchestrator
├── metadata.py              # Metadata Excel loading, FeatureSpec
├── reasoning.py             # SHAP → business reason text
├── io_utils.py              # File I/O helpers
├── run_explainability.py    # CLI entry point
├── configs/explainability_config.yaml
├── metadata/explainability_metadata.xlsx
└── tests/
```

**Pipeline:** load model → validate dataset → score in chunks → Tree SHAP → reason codes → global importance → plots + Excel report + manifest

---

## License

MIT
