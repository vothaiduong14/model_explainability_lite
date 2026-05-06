"""Business reasoning layer built on top of SHAP outputs."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from model_explainability.metadata import MetadataBundle


def _safe_str(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    return str(value)


def format_feature_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        if math.isnan(value):
            return "missing"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def build_feature_metadata(bundle: MetadataBundle, feature_names: list[str]) -> pd.DataFrame:
    base = pd.DataFrame({"feature_name": feature_names})
    for frame in [
        bundle.feature_dictionary,
        bundle.reason_mapping,
        bundle.typology_mapping,
        bundle.feature_grouping,
    ]:
        if frame.empty or "feature_name" not in frame.columns:
            continue
        deduped = frame.drop_duplicates(subset=["feature_name"])
        base = base.merge(deduped, on="feature_name", how="left")

    if "feature_business_name" not in base.columns:
        base["feature_business_name"] = base["feature_name"]
    base["feature_business_name"] = base["feature_business_name"].fillna(base["feature_name"])
    base["reason_code"] = base.get("reason_code", pd.Series(index=base.index, dtype="object")).fillna(
        base["feature_name"].str.upper().map(lambda name: f"RC_{name}")
    )
    base["reason_title"] = base.get("reason_title", pd.Series(index=base.index, dtype="object")).fillna(
        base["feature_business_name"]
    )
    base["group_name"] = base.get("group_name", pd.Series(index=base.index, dtype="object")).fillna(
        "unmapped"
    )
    base["typology_name"] = base.get("typology_name", pd.Series(index=base.index, dtype="object")).fillna(
        "unmapped"
    )
    base["business_explanation_template"] = base.get(
        "business_explanation_template", pd.Series(index=base.index, dtype="object")
    )
    return base.set_index("feature_name", drop=False)


def render_reason_text(meta_row: pd.Series, shap_value: float, feature_value: Any) -> str:
    template = _safe_str(meta_row.get("business_explanation_template"))
    feature_label = _safe_str(meta_row.get("feature_business_name"), meta_row["feature_name"])
    value_text = format_feature_value(feature_value)

    if template:
        try:
            return template.format(
                feature_name=meta_row["feature_name"],
                feature_business_name=feature_label,
                value=value_text,
                direction="increase" if shap_value >= 0 else "decrease",
            )
        except KeyError:
            pass

    direction_key = "positive_risk_direction" if shap_value >= 0 else "negative_risk_direction"
    direction_text = _safe_str(meta_row.get(direction_key))
    if direction_text:
        return f"{feature_label} ({value_text}) {direction_text}."
    if shap_value >= 0:
        return f"{feature_label} ({value_text}) increases fraud risk."
    return f"{feature_label} ({value_text}) reduces fraud risk."


def build_summary_text(reason_rows: list[dict[str, Any]]) -> str:
    positive = [row for row in reason_rows if row["shap_value"] > 0]
    if not positive:
        return "Score is mainly supported by risk-reducing features."

    themes = []
    for row in positive:
        theme = row.get("group_name") or row.get("reason_title") or row["feature_name"]
        if theme not in themes:
            themes.append(theme)
    themes = themes[:3]

    if len(themes) == 1:
        return f"High score driven mainly by {themes[0]}."
    if len(themes) == 2:
        return f"High score driven mainly by {themes[0]} and {themes[1]}."
    return f"High score driven mainly by {themes[0]}, {themes[1]}, and {themes[2]}."


def explanation_quality_flag(reason_rows: list[dict[str, Any]]) -> str:
    if not reason_rows:
        return "NO_TOP_REASONS"
    if any(row.get("group_name") == "unmapped" for row in reason_rows):
        return "LOW_MAPPING_COVERAGE"
    if any(str(row.get("feature_value")) == "missing" for row in reason_rows):
        return "FEATURE_VALUE_MISSING"
    return "OK"
