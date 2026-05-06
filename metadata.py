"""Metadata and feature-contract loading for explainability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import pandas as pd

def _read_text_with_fallback(path:Path) -> str:
    raw = path.read_bytes()
    tried: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            tried.append(encoding)
    raise UnicodeDecodeError(
        "text",
        raw,
        0,
        1,
        f"Unable to decode file {path} using encodings: {', '.join(tried)}",
    )

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str | None = None
    default_fill: Any = 0.0


@dataclass
class MetadataBundle:
    feature_dictionary: pd.DataFrame
    reason_mapping: pd.DataFrame
    typology_mapping: pd.DataFrame
    feature_grouping: pd.DataFrame
    thresholds_bands: pd.DataFrame
    report_config: pd.DataFrame


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    return frame


def load_feature_specs(path: Path) -> list[FeatureSpec]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        payload = json.loads(_read_text_with_fallback(path))
        if isinstance(payload, dict):
            feature_names = payload.get("feature_names") or payload.get("features") or []
            return [FeatureSpec(name=str(name)) for name in feature_names]
        if isinstance(payload, list):
            return [FeatureSpec(name=str(name)) for name in payload]
    elif suffix in {".txt", ".lst"}:
        return [
            FeatureSpec(name=line.strip()) 
            for line in _read_text_with_fallback(path).splitlines()
            if line.strip()
        ]
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        frame = _normalise_columns(frame)
        feature_col = "feature_name" if "feature_name" in frame.columns else "name"
        if feature_col not in frame.columns:
            raise ValueError("Feature list CSV must contain 'feature_name' or 'name'")
        specs = []
        for row in frame.to_dict(orient="records"):
            specs.append(
                FeatureSpec(
                    name=str(row[feature_col]),
                    dtype=row.get("dtype"),
                    default_fill=row.get("default_fill", 0.0),
                )
            )
        return specs

    raise ValueError(f"Unsupported feature list format: {path.suffix}")


def empty_metadata_bundle() -> MetadataBundle:
    empty_feature = pd.DataFrame(columns=["feature_name", "feature_business_name", "description"])
    return MetadataBundle(
        feature_dictionary=empty_feature.copy(),
        reason_mapping=pd.DataFrame(
            columns=[
                "feature_name",
                "reason_code",
                "reason_title",
                "business_explanation_template",
                "positive_risk_direction",
                "negative_risk_direction",
                "display_priority",
                "evidence_fields",
            ]
        ),
        typology_mapping=pd.DataFrame(
            columns=["feature_name", "typology_id", "typology_name", "typology_description"]
        ),
        feature_grouping=pd.DataFrame(
            columns=["feature_name", "group_name", "subgroup_name", "domain"]
        ),
        thresholds_bands=pd.DataFrame(
            columns=["score_min", "score_max", "band", "operational_meaning"]
        ),
        report_config=pd.DataFrame(columns=["parameter", "value"]),
    )


def load_metadata_bundle(path: Path | None) -> MetadataBundle:
    if path is None:
        return empty_metadata_bundle()

    xls = pd.ExcelFile(path)
    bundle = empty_metadata_bundle()
    for tab_name in xls.sheet_names:
        frame = _normalise_columns(pd.read_excel(path, sheet_name=tab_name))
        key = tab_name.strip().lower()
        if key == "feature_dictionary":
            bundle.feature_dictionary = frame
        elif key == "reason_mapping":
            bundle.reason_mapping = frame
        elif key == "typology_mapping":
            bundle.typology_mapping = frame
        elif key == "feature_grouping":
            bundle.feature_grouping = frame
        elif key == "thresholds_bands":
            bundle.thresholds_bands = frame
        elif key == "report_config":
            bundle.report_config = frame

    return bundle


def create_metadata_template(path: Union[str, Path]) -> None:
    path = Path(path)
    bundle = empty_metadata_bundle()
    sample_feature_dictionary = pd.DataFrame(
        [
            {
                "feature_name": "f_card_txn_count_all_24h",
                "feature_business_name": "24-hour transaction count",
                "description": "Number of transactions observed on the card in the last 24 hours.",
                "unit": "count",
                "direction_interpretation": "Higher values usually increase risk.",
                "calculation_summary": "Point-in-time trailing 24-hour count.",
                "owner_team": "Fraud Analytics",
                "data_source": "feature_store",
                "category": "velocity",
            }
        ]
    )
    sample_reason_mapping = pd.DataFrame(
        [
            {
                "feature_name": "f_card_txn_count_all_24h",
                "reason_code": "RC_VELOCITY_24H",
                "reason_title": "Elevated short-term velocity",
                "business_explanation_template": "Transaction activity on this card in the past 24 hours is materially elevated ({value}).",
                "positive_risk_direction": "increases fraud risk",
                "negative_risk_direction": "reduces fraud risk",
                "display_priority": 1,
                "evidence_fields": "f_card_txn_count_all_24h",
            }
        ]
    )
    sample_typology = pd.DataFrame(
        [
            {
                "feature_name": "f_card_txn_count_all_24h",
                "typology_id": "T001",
                "typology_name": "Rapid spend",
                "typology_description": "Activity patterns consistent with rapid compromise usage.",
                "stage": "alert triage",
                "use_case": "fraud operations",
                "channel_applicability": "all",
            }
        ]
    )
    sample_grouping = pd.DataFrame(
        [
            {
                "feature_name": "f_card_txn_count_all_24h",
                "group_name": "velocity",
                "subgroup_name": "card_velocity",
                "domain": "behavioral_deviation",
            }
        ]
    )
    sample_bands = pd.DataFrame(
        [
            {"score_min": 0, "score_max": 400, "band": "green", "operational_meaning": "Low risk"},
            {"score_min": 401, "score_max": 700, "band": "amber", "operational_meaning": "Review"},
            {"score_min": 701, "score_max": 1000, "band": "red", "operational_meaning": "Alert"},
        ]
    )
    sample_report_config = pd.DataFrame(
        [
            {"parameter": "analyst_summary_template", "value": "High score driven by {themes}."},
            {"parameter": "detail_summary_template", "value": "Risk is driven by {reasons}."},
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sample_feature_dictionary.to_excel(writer, sheet_name="feature_dictionary", index=False)
        sample_reason_mapping.to_excel(writer, sheet_name="reason_mapping", index=False)
        sample_typology.to_excel(writer, sheet_name="typology_mapping", index=False)
        sample_grouping.to_excel(writer, sheet_name="feature_grouping", index=False)
        sample_bands.to_excel(writer, sheet_name="thresholds_bands", index=False)
        sample_report_config.to_excel(writer, sheet_name="report_config", index=False)
