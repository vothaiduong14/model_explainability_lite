from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from model_explainability.config import ExplainabilityConfig
from model_explainability.metadata import create_metadata_template
from model_explainability.service import ExplainabilityService


def _build_toy_inputs(tmp_path: Path) -> ExplainabilityConfig:
    train = pd.DataFrame(
        {
            "f_velocity": [0, 1, 2, 6, 7, 8],
            "f_amount_ratio": [0.1, 0.2, 0.3, 1.4, 1.5, 1.8],
            "label": [0, 0, 0, 1, 1, 1],
        }
    )
    dtrain = xgb.DMatrix(train[["f_velocity", "f_amount_ratio"]], label=train["label"])
    model = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 2,
            "eta": 0.3,
            "verbosity": 0,
        },
        dtrain,
        num_boost_round=8,
    )

    model_path = tmp_path / "model.ubj"
    model.save_model(model_path)

    score_data = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "txn_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
            ),
            "channel": ["pos", "ecom", "ecom", "atm"],
            "customer_tier": ["standard", "gold", "gold", "standard"],
            "label": [0, 0, 1, 1],
            "f_velocity": [0, 2, 7, 8],
            "f_amount_ratio": [0.1, 0.3, 1.5, 1.8],
        }
    )
    score_path = tmp_path / "oot.parquet"
    score_data.to_parquet(score_path, index=False)

    feature_list_path = tmp_path / "model_signature.json"
    feature_list_path.write_text(json.dumps({"feature_names": ["f_velocity", "f_amount_ratio"]}))

    metadata_path = tmp_path / "metadata.xlsx"
    create_metadata_template(metadata_path)
    with pd.ExcelWriter(metadata_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        pd.DataFrame(
            [
                {
                    "feature_name": "f_velocity",
                    "feature_business_name": "Short-term velocity",
                    "description": "Recent transaction activity.",
                },
                {
                    "feature_name": "f_amount_ratio",
                    "feature_business_name": "Amount deviation",
                    "description": "Transaction amount vs customer norm.",
                },
            ]
        ).to_excel(writer, sheet_name="feature_dictionary", index=False)
        pd.DataFrame(
            [
                {
                    "feature_name": "f_velocity",
                    "reason_code": "RC_VELOCITY",
                    "reason_title": "Elevated short-term velocity",
                    "business_explanation_template": "Velocity is elevated at {value}.",
                    "positive_risk_direction": "increases fraud risk",
                    "negative_risk_direction": "reduces fraud risk",
                },
                {
                    "feature_name": "f_amount_ratio",
                    "reason_code": "RC_AMOUNT",
                    "reason_title": "Amount above norm",
                    "business_explanation_template": "Amount deviation is {value}.",
                    "positive_risk_direction": "increases fraud risk",
                    "negative_risk_direction": "reduces fraud risk",
                },
            ]
        ).to_excel(writer, sheet_name="reason_mapping", index=False)
        pd.DataFrame(
            [
                {"feature_name": "f_velocity", "typology_name": "Rapid spend"},
                {"feature_name": "f_amount_ratio", "typology_name": "Amount anomaly"},
            ]
        ).to_excel(writer, sheet_name="typology_mapping", index=False)
        pd.DataFrame(
            [
                {"feature_name": "f_velocity", "group_name": "velocity"},
                {"feature_name": "f_amount_ratio", "group_name": "amount anomaly"},
            ]
        ).to_excel(writer, sheet_name="feature_grouping", index=False)

    return ExplainabilityConfig(
        run_id="unit_test_run",
        paths={
            "model_file": model_path,
            "scoring_data_file": score_path,
            "feature_list_file": feature_list_path,
            "metadata_excel_file": metadata_path,
            "output_root": tmp_path / "outputs",
        },
        dataset={
            "id_column": "transaction_id",
            "score_ts_column": "txn_timestamp",
            "label_column": "label",
            "segment_columns": ["channel", "customer_tier"],
        },
        sampling={
            "scoring_chunk_rows": 2,
            "shap_chunk_size": 10,
            "global_sample_size": 4,
            "global_top_score_percentile": 0.25,
            "local_max_rows": 4,
            "local_non_alert_sample_size": 1,
        },
        scoring={
            "model_version": "toy_model_v1",
            "score_scale": "0_1000",
            "alert_probability_threshold": 0.55,
            "top_n_reasons": 2,
            "top_n_negative_reasons": 1,
        },
    )


def test_explainability_service_end_to_end(tmp_path: Path):
    config = _build_toy_inputs(tmp_path)
    service = ExplainabilityService(config)
    manifest = service.run()

    run_dir = tmp_path / "outputs" / "unit_test_run"
    assert manifest["run_id"] == "unit_test_run"
    assert (run_dir / "scored_transactions.parquet").exists()
    assert (run_dir / "local_explanations.parquet").exists()
    assert (run_dir / "global_feature_importance.parquet").exists()
    assert (run_dir / "global_feature_importance_by_segment.parquet").exists()
    assert (run_dir / "mapping_coverage_report.csv").exists()
    assert (run_dir / "validation_checks.csv").exists()
    assert (run_dir / "reports" / "explainability_summary.xlsx").exists()
    assert (run_dir / "plots" / "global_importance.png").exists()

    local_df = pd.read_parquet(run_dir / "local_explanations.parquet")
    assert not local_df.empty
    assert "top1_reason_code" in local_df.columns
    assert local_df["explanation_quality_flag"].isin(["OK", "LOW_MAPPING_COVERAGE"]).all()

    global_df = pd.read_parquet(run_dir / "global_feature_importance.parquet")
    assert set(global_df["feature_name"]) == {"f_velocity", "f_amount_ratio"}
