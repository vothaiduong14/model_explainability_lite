"""Standalone explainability service implementation."""

from __future__ import annotations

import logging
import pickle
import subprocess
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from model_explainability.config import ExplainabilityConfig
from model_explainability.io_utils import (
    ensure_run_dirs,
    file_sha256,
    generate_run_id,
    scan_dataset,
    write_json,
    write_validation_checks,
    write_yaml,
)
from model_explainability.metadata import (
    FeatureSpec,
    create_metadata_template,
    load_feature_specs,
    load_metadata_bundle,
)
from model_explainability.reasoning import (
    build_feature_metadata,
    build_summary_text,
    explanation_quality_flag,
    format_feature_value,
    render_reason_text,
)

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """Runs post-training explainability on approved model artifacts."""

    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        if not self.config.run_id:
            self.config.run_id = generate_run_id()
        self.run_dirs = ensure_run_dirs(self.config.paths.output_root, self.config.run_id)
        self.validation_checks: list[dict] = []

    def run(self) -> dict:
        self._configure_logging()

        model = self._load_model()
        feature_specs = self._resolve_feature_specs(model)
        feature_names = [spec.name for spec in feature_specs]
        metadata_bundle = load_metadata_bundle(self.config.paths.metadata_excel_file)
        feature_meta = build_feature_metadata(metadata_bundle, feature_names)
        calibrator = self._load_calibrator()

        self._validate_dataset_contract(feature_names, metadata_bundle)
        scored_path, scored_stats = self._score_dataset(model, calibrator, feature_specs, metadata_bundle)
        local_ids = self._select_local_population(scored_path)
        global_ids = self._select_global_population(scored_path)

        local_joined = self._collect_source_rows(feature_specs, local_ids, scored_path)
        global_joined = self._collect_source_rows(feature_specs, global_ids, scored_path)

        local_outputs = self._build_local_outputs(model, feature_specs, feature_meta, local_joined)
        global_outputs = self._build_global_outputs(
            model,
            feature_specs,
            feature_meta,
            global_joined,
        )

        self._write_reporting_pack(local_outputs, global_outputs)
        self._write_top_scored_transactions(scored_path, top_n=10_000)
        mapping_coverage = self._write_mapping_coverage(feature_meta)
        manifest = self._write_manifest(scored_stats, mapping_coverage, feature_specs)
        self._write_validation_checks()

        return manifest

    def _configure_logging(self) -> None:
        log_path = self.run_dirs["logs"] / "explainability.log"
        pkg_logger = logging.getLogger("model_explainability")
        if not pkg_logger.handlers:
            pkg_logger.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            fh = logging.FileHandler(log_path)
            fh.setFormatter(fmt)
            pkg_logger.addHandler(sh)
            pkg_logger.addHandler(fh)

    def _load_model(self) -> xgb.Booster:
        model_path = Path(self.config.paths.model_file)
        suffix = model_path.suffix.lower()

        if suffix in {".ubj", ".json", ".bin", ".model"}:
            model = xgb.Booster()
            model.load_model(str(model_path))
            self._record_check("model_load", "PASS", f"Loaded XGBoost model file {model_path}")
            return model
        
        if suffix == ".pkl":
            with model_path.open("rb") as f:
                obj = pickle.load(f)

            if isinstance(obj, xgb.Booster):
                self._record_check("model_load", "PASS", f"Loaded pickled xgboost.Booster from {model_path}")
                return obj
            
            if hasattr(obj, "get_booster"):
                booster = obj.get_booster()
                if isinstance(booster, xgb.Booster):
                    self._record_check(
                        "model_load",
                        "PASS",
                        f"Loaded pickled XGBoost sklearn wrapper from {model_path} and extracted Booster",
                    )
                    return booster
            
            raise TypeError(
                "Unsupported .pkl model object type. Expected xgboost.Booster or an XGBoost sklearn wrapper"
                "with get_booster()."
            )
        
        raise ValueError(
            f"Unsupported model file extension  '{suffix}'. Supported extensions: .ubj, .json, .bin, .model, .pkl"
        )
    
    def _resolve_feature_specs(self, model: xgb.Booster) -> list[FeatureSpec]:
        feature_list_path = self.config.paths.feature_list_file
        model_feature_names = model.feature_names or []

        if feature_list_path is not None: 
            feature_specs = load_feature_specs(feature_list_path)
            config_feature_names = [spec.name for spec in feature_specs]
            if model_feature_names and config_feature_names != list(model_feature_names):
                raise ValueError(
                    "Configured feature list does not match model.feature_names ordering/ content. "
                    "Use the model-embedded features or update feature_list_file to an exact match. "
                )
            self._record_check("feauture_list_source", "PASS", f"Loaded feature list from {feature_list_path}")
            return feature_specs

        if not model_feature_names:
            raise ValueError(
                "feauture_list_file is not provided and model does not contain feature_names."
                "Provide paths.feature_list_file in config."
            )
        
        self._record_check("feature_list_source", "PASS", "Using feature names embedded in model artifact")
        return [FeatureSpec(name=str(name)) for name in model_feature_names]
    

    def _load_calibrator(self):
        path = self.config.paths.calibration_file
        if path is None:
            return None
        import joblib

        calibrator = joblib.load(path)
        self._record_check("calibrator_load", "PASS", f"Loaded {path}")
        return calibrator

    def _validate_dataset_contract(self, feature_names: list[str], metadata_bundle) -> None:
        lazy_frame = scan_dataset(self.config.paths.scoring_data_file)
        available_cols = lazy_frame.collect_schema().names()
        required_cols = {self.config.dataset.id_column, *feature_names}
        if self.config.dataset.score_ts_column:
            required_cols.add(self.config.dataset.score_ts_column)
        if self.config.dataset.label_column:
            required_cols.add(self.config.dataset.label_column)

        missing = sorted(required_cols.difference(available_cols))
        if missing:
            raise ValueError(f"Missing required scoring columns: {missing}")
        self._record_check("feature_presence", "PASS", f"Validated {len(feature_names)} model features")

        duplicates = (
            lazy_frame.select(self.config.dataset.id_column)
            .group_by(self.config.dataset.id_column)
            .len()
            .filter(pl.col("len") > 1)
            .limit(1)
            .collect()
        )
        if len(duplicates) and not self.config.dataset.allow_duplicate_ids:
            raise ValueError("Duplicate transaction ids found in scoring data")
        self._record_check("duplicate_keys", "PASS", "No duplicate transaction ids found")

        null_exprs = [
            pl.col(feature).is_null().mean().alias(feature)
            for feature in feature_names
        ]
        null_rates = lazy_frame.select(null_exprs).collect().to_dicts()[0]
        high_null = sorted(
            [feature for feature, rate in null_rates.items() if rate >= self.config.validation.null_warning_threshold]
        )
        status = "WARN" if high_null else "PASS"
        detail = "High-null features: " + ", ".join(high_null[:20]) if high_null else "Null rates within threshold"
        self._record_check("null_rate_scan", status, detail)

        extra_cols = sorted(set(available_cols).difference(required_cols))
        if extra_cols and not self.config.dataset.allow_extra_columns:
            raise ValueError("Unexpected extra columns found in scoring dataset")
        self._record_check(
            "extra_columns",
            "WARN" if extra_cols else "PASS",
            f"{len(extra_cols)} extra columns present",
        )

        missing_tabs = []
        for tab in self.config.validation.required_metadata_tabs:
            frame = getattr(metadata_bundle, tab, pd.DataFrame())
            if frame.empty:
                missing_tabs.append(tab)
        self._record_check(
            "metadata_tabs",
            "WARN" if missing_tabs else "PASS",
            f"Missing metadata tabs: {missing_tabs}" if missing_tabs else "Required metadata tabs present",
        )

    def _score_dataset(
        self,
        model: xgb.Booster,
        calibrator,
        feature_specs: list[FeatureSpec],
        metadata_bundle,
    ) -> tuple[Path, dict]:
        feature_names = [spec.name for spec in feature_specs]
        available_cols = scan_dataset(self.config.paths.scoring_data_file).collect_schema().names()
        segment_cols = [col for col in self.config.dataset.segment_columns if col in available_cols]
        keep_cols = [self.config.dataset.id_column]
        if self.config.dataset.score_ts_column:
            keep_cols.append(self.config.dataset.score_ts_column)
        if self.config.dataset.label_column:
            keep_cols.append(self.config.dataset.label_column)
        keep_cols.extend(segment_cols)
        keep_cols.extend(feature_names)
        keep_cols = list(dict.fromkeys(keep_cols))

        lazy_frame = scan_dataset(self.config.paths.scoring_data_file).select(keep_cols)
        total_rows = lazy_frame.select(pl.len()).collect().item()
        chunk_rows = self.config.sampling.scoring_chunk_rows
        scored_path = self.run_dirs["base"] / "scored_transactions.parquet"

        writer = None
        total_alerts = 0
        positive_labels = 0

        try:
            for offset in range(0, total_rows, chunk_rows):
                chunk = lazy_frame.slice(offset, chunk_rows).collect()
                feature_frame = chunk.select(feature_names).to_pandas()
                X = self._standardise_features(feature_specs, feature_frame)
                dmatrix = xgb.DMatrix(X, feature_names=feature_names)
                probabilities = model.predict(dmatrix)
                margins = model.predict(dmatrix, output_margin=True)
                calibrated = self._calibrate_probabilities(calibrator, probabilities)
                score_final = self._score_scale(calibrated)
                score_band = self._apply_score_band(score_final, metadata_bundle)
                threshold_flag = self._threshold_flag(calibrated, score_band)

                result = chunk.select(
                    [
                        col
                        for col in [
                            self.config.dataset.id_column,
                            self.config.dataset.score_ts_column,
                            self.config.dataset.label_column,
                            *segment_cols,
                        ]
                        if col and col in chunk.columns
                    ]
                ).to_pandas()
                result["run_id"] = self.config.run_id
                result["model_version"] = self.config.scoring.model_version
                result["prediction_margin"] = margins
                result["prediction_probability"] = probabilities
                result["calibrated_probability"] = calibrated
                result["score_final"] = score_final
                result["score_band"] = score_band
                result["threshold_flag"] = threshold_flag
                result["explanation_status"] = "PENDING"
                result["explanation_warning"] = ""

                table = pa.Table.from_pandas(result, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(scored_path, table.schema)
                writer.write_table(table)

                total_alerts += int(np.asarray(threshold_flag, dtype=np.int8).sum())
                if self.config.dataset.label_column and self.config.dataset.label_column in result.columns:
                    positive_labels += int(result[self.config.dataset.label_column].fillna(0).astype(int).sum())
                logger.info("Scored chunk %s-%s / %s", offset, min(offset + chunk_rows, total_rows), total_rows)
                del chunk, feature_frame, X, dmatrix, result, table
        finally:
            if writer is not None:
                writer.close()

        stats = {
            "scored_rows": int(total_rows),
            "alerts": total_alerts,
            "positive_labels": positive_labels,
            "scored_output": str(scored_path),
        }
        self._record_check("scoring_output", "PASS", f"Wrote {total_rows} scored rows")
        return scored_path, stats

    def _standardise_features(self, feature_specs: list[FeatureSpec], frame: pd.DataFrame) -> np.ndarray:
        # standardised = pd.DataFrame(index=frame.index)
        # for spec in feature_specs:
        #     series = pd.to_numeric(frame[spec.name], errors="coerce")
        #     fill_value = 0.0 if spec.default_fill is None or pd.isna(spec.default_fill) else spec.default_fill
        #     if spec.dtype and str(spec.dtype).lower().startswith("int"):
        #         standardised[spec.name] = series.fillna(fill_value).astype(np.int64)
        #     else:
        #         standardised[spec.name] = series.fillna(fill_value).astype(np.float32)
        # return standardised[[spec.name for spec in feature_specs]].to_numpy(dtype=np.float32)
        feature_names = [spec.name for spec in feature_specs]
        return frame.loc[:, feature_names].to_numpy()

    def _calibrate_probabilities(self, calibrator, probabilities: np.ndarray) -> np.ndarray:
        if calibrator is None:
            return probabilities
        if hasattr(calibrator, "predict_proba"):
            return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        if hasattr(calibrator, "predict"):
            return calibrator.predict(probabilities)
        raise TypeError("Unsupported calibration artifact: missing predict/predict_proba")

    def _score_scale(self, probabilities: np.ndarray) -> np.ndarray:
        if self.config.scoring.score_scale == "0_1000":
            return np.round(probabilities * 1000).astype(int)
        if self.config.scoring.score_scale == "0_100":
            return np.round(probabilities * 100).astype(int)
        return probabilities

    def _apply_score_band(self, score_final: np.ndarray, metadata_bundle) -> np.ndarray:
        result = np.full(len(score_final), None, dtype=object)
        if metadata_bundle.thresholds_bands.empty:
            return result
        bands_df = metadata_bundle.thresholds_bands.dropna(subset=["score_min", "score_max"])
        for _, row in bands_df.iterrows():
            mask = (score_final >= row["score_min"]) & (score_final <= row["score_max"])
            result[mask] = row.get("band")
        return result

    def _threshold_flag(self, calibrated_probability: np.ndarray, score_band: np.ndarray) -> np.ndarray:
        if self.config.scoring.alert_probability_threshold is not None:
            return calibrated_probability >= self.config.scoring.alert_probability_threshold
        return np.isin(score_band, ["red", "amber"])
    
    def _get_ranked_scored_frame(self, scored_path: Path) -> pl.LazyFrame:
        scored_lazy = pl.scan_parquet(scored_path)
        ranking_file = self.config.paths.ranking_score_file
        ranking_score_column = self.config.scoring.ranking_score_column

        if ranking_file is None or ranking_score_column is None:
            return scored_lazy
        
        ranking_id_column = self.config.scoring.ranking_score_id_column or self.config.dataset.id_column
        ranking_lazy = (
            scan_dataset(ranking_file)
            .select(
                pl.col(ranking_id_column).cast(pl.Utf8).alias(self.config.dataset.id_column),
                pl.col(ranking_score_column).cast(pl.Float64).fill_null(0.0).alias("ranking_score"),
            )
            .group_by(self.config.dataset.id_column)
            .agg(pl.col("ranking_score").max().alias("ranking_score"))
        )
        return scored_lazy.join(ranking_lazy, on=self.config.dataset.id_column, how="left").with_columns(
            pl.col("ranking_score").fill_null(0.0)
        )

    def _get_scored_sort_spec(self, available_cols: list[str]) -> tuple[list[str], list[bool]]:
        id_col = self.config.dataset.id_column
        sort_cols: list[str] = []
        descending: list[bool] = []

        for column in ["ranking_score", "prediction_probability", "calibrated_probability", "score_final"]:
            if column in available_cols:
                sort_cols.append(column)
                descending.append(True)
            
            if id_col in available_cols:
                sort_cols.append(id_col)
                descending.append(False)
            
            return sort_cols, descending

    def _select_local_population(self, scored_path: Path) -> list[str]:
        frame = self._get_ranked_scored_frame(scored_path)
        id_col = self.config.dataset.id_column

        available_cols = frame.collect_schema().names()
        rank_cols, rank_descending =self._get_scored_sort_spec(available_cols)
        base_cols = list(dict.fromkeys([id_col, *rank_cols]))
        selected_frames = []
        top_score_n = max(0, self.config.sampling.local_top_score_rows)
        if top_score_n and rank_cols:
            selected_frames.append(
                frame.select(base_cols)
                .sort(rank_cols, descending=rank_descending)
                .head(top_score_n)
            )
        if self.config.sampling.include_all_alerts_in_local:
            selected_frames.append(
                frame.filter(pl.col("threshold_flag")).select(base_cols)
            )
        if self.config.dataset.label_column and self.config.sampling.include_all_positive_labels_in_local:
            selected_frames.append(
                frame.filter(pl.col(self.config.dataset.label_column) == 1).select(base_cols)
            )

        if selected_frames:
            selected = pl.concat(selected_frames).unique(subset=[id_col]).collect()
        else:
            selected = pl.DataFrame(
                schema={
                    id_col: pl.Utf8,
                    "ranking_score": pl.Float64,
                    "prediction_probability": pl.Float64,
                    "score_final": pl.Int64, 
                    "calibrated_probability": pl.Float64
                    }
                )

        limit = max(self.config.sampling.local_max_rows, top_score_n)
        if rank_cols:
            selected = selected.sort(rank_cols, descending=rank_descending)

        if limit and len(selected) > limit and rank_cols:
            selected = selected.head(limit)

        random_n = self.config.sampling.local_non_alert_sample_size
        remaining_slots = max(0, limit - len(selected)) if limit else random_n
        if random_n > 0 and remaining_slots > 0:
            already = set(selected[id_col].to_list())
            random_sample = (
                frame.filter(~pl.col(id_col).is_in(list(already)) & ~pl.col("threshold_flag"))
                .select(base_cols)
                .collect()
            )
            if len(random_sample):
                n = min(random_n, remaining_slots, len(random_sample))
                random_sample = random_sample.sample(n=n, seed=self.config.sampling.random_seed)
                selected = pl.concat([selected, random_sample]).unique(subset=[id_col])

        if rank_cols:
            selected = selected.sort(rank_cols, descending=rank_descending)

        ids = [str(value) for value in selected[id_col].to_list()]
        self._record_check("local_population", "PASS", f"Selected {len(ids)} rows for local explanations")
        return ids

    def _select_global_population(self, scored_path: Path) -> list[str]:
        lazy = pl.scan_parquet(scored_path)
        id_col = self.config.dataset.id_column
        base_cols = [id_col, "calibrated_probability"]

        selected_frames: list[pl.DataFrame] = []
        if self.config.dataset.label_column and self.config.sampling.include_all_positive_labels_in_global:
            selected_frames.append(
                lazy.filter(pl.col(self.config.dataset.label_column) == 1).select(base_cols).collect()
            )

        total_rows = lazy.select(pl.len()).collect().item()
        if total_rows:
            threshold = float(
                lazy.select(
                    pl.col("calibrated_probability").quantile(
                        1 - self.config.sampling.global_top_score_percentile
                    )
                ).collect().item()
            )
            selected_frames.append(
                lazy.filter(pl.col("calibrated_probability") >= threshold).select(base_cols).collect()
            )

        selected = (
            pl.concat(selected_frames).unique(subset=[id_col])
            if selected_frames
            else pl.DataFrame(schema={id_col: pl.Utf8, "calibrated_probability": pl.Float64})
        )
        remaining_slots = max(0, self.config.sampling.global_sample_size - len(selected))
        if remaining_slots > 0:
            already = selected[id_col].to_list() if len(selected) else []
            remainder = lazy.filter(~pl.col(id_col).is_in(already)).select(base_cols).collect()
            if len(remainder):
                n = min(remaining_slots, len(remainder))
                remainder = remainder.sample(n=n, seed=self.config.sampling.random_seed)
                selected = pl.concat([selected, remainder]).unique(subset=[id_col])

        ids = [str(value) for value in selected[id_col].to_list()]
        self._record_check("global_population", "PASS", f"Selected {len(ids)} rows for global explanations")
        return ids

    def _collect_source_rows(
        self,
        feature_specs: list[FeatureSpec],
        ids: list[str],
        scored_path: Path,
    ) -> pd.DataFrame:
        if not ids:
            return pd.DataFrame()

        feature_names = [spec.name for spec in feature_specs]
        source_keep = [self.config.dataset.id_column]
        if self.config.dataset.score_ts_column:
            source_keep.append(self.config.dataset.score_ts_column)
        if self.config.dataset.label_column:
            source_keep.append(self.config.dataset.label_column)
        source_keep.extend(self.config.dataset.segment_columns)
        source_keep.extend(feature_names)
        source_keep = list(dict.fromkeys(source_keep))

        source_lazy = scan_dataset(self.config.paths.scoring_data_file)
        available_source_cols = source_lazy.collect_schema().names()
        source = (
            source_lazy
            .select([col for col in source_keep if col in available_source_cols])
            .with_columns(pl.col(self.config.dataset.id_column).cast(pl.Utf8))
            .filter(pl.col(self.config.dataset.id_column).is_in(ids))
            .collect()
        )
        scored = (
            self._get_ranked_scored_frame(scored_path)
            .with_columns(pl.col(self.config.dataset.id_column).cast(pl.Utf8))
            .filter(pl.col(self.config.dataset.id_column).is_in(ids))
            .collect()
        )
        joined = source.join(scored, on=self.config.dataset.id_column, how="inner")
        joined_df = joined.to_pandas()
        rank_order = pd.Series(range(len(ids)), index = pd.Index(ids, dtype='object'))
        joined_df["__rank_order"] = joined_df[self.config.dataset.id_column].astype(str).map(rank_order)
        joined_df = joined_df.sort_values("__rank_order", kind="stable").drop(columns="__rank_order")
        return joined_df.reset_index(drop=True)

    def _build_local_outputs(
        self,
        model: xgb.Booster,
        feature_specs: list[FeatureSpec],
        feature_meta: pd.DataFrame,
        joined: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        if joined.empty:
            empty_local = pd.DataFrame()
            empty_long = pd.DataFrame()
            empty_local.to_parquet(self.run_dirs["base"] / "local_explanations.parquet", index=False)
            empty_long.to_parquet(self.run_dirs["base"] / "shap_values_long.parquet", index=False)
            return {"local": empty_local, "shap_long": empty_long}

        feature_names = [spec.name for spec in feature_specs]
        raw_feature_values = joined[feature_names].copy()
        X = self._standardise_features(feature_specs, joined[feature_names].copy())
        shap_values, base_value = self._compute_shap(model, feature_names, X)
        reconstruction = base_value + shap_values.sum(axis=1)
        delta = np.abs(reconstruction - joined["prediction_margin"].to_numpy())
        if delta.max() > self.config.validation.shap_reconstruction_tolerance:
            status = "WARN"
            detail = f"Max SHAP reconstruction delta {delta.max():.6f}"
        else:
            status = "PASS"
            detail = f"Max SHAP reconstruction delta {delta.max():.6f}"
        self._record_check("shap_reconstruction", status, detail)

        top_n = self.config.scoring.top_n_reasons
        negative_n = self.config.scoring.top_n_negative_reasons
        local_rows: list[dict] = []

        # Pre-compute argsort for all rows at once (avoids per-row np.argsort)
        abs_shap = np.abs(shap_values)
        all_ranked = np.argsort(-abs_shap, axis=1)
        all_pos_sorted = np.argsort(-shap_values, axis=1)
        all_neg_sorted = np.argsort(shap_values, axis=1)
        feature_names_arr = np.array(feature_names)

        joined_reset = joined.reset_index(drop=True)
        for row_idx in range(len(joined_reset)):
            row = joined_reset.iloc[row_idx]
            row_shap = shap_values[row_idx]
            top_idx = all_ranked[row_idx, :top_n]
            pos_idx = all_pos_sorted[row_idx, :top_n]
            neg_idx = all_neg_sorted[row_idx, :negative_n]
            reason_rows = []

            for feat_idx in top_idx:
                feat_name = feature_names[feat_idx]
                meta_row = feature_meta.loc[feat_name]
                feature_value = raw_feature_values.iloc[row_idx, feat_idx]
                reason_rows.append({
                    "feature_name": feat_name,
                    "feature_value": format_feature_value(feature_value),
                    "shap_value": float(row_shap[feat_idx]),
                    "reason_code": meta_row["reason_code"],
                    "reason_title": meta_row["reason_title"],
                    "reason_text": render_reason_text(meta_row, float(row_shap[feat_idx]), feature_value),
                    "typology_name": meta_row["typology_name"],
                    "group_name": meta_row["group_name"],
                })

            local_record = {
                "run_id": self.config.run_id,
                self.config.dataset.id_column: row[self.config.dataset.id_column],
                "model_version": row["model_version"],
                "base_value": float(base_value),
                "prediction_margin": float(row["prediction_margin"]),
                "prediction_probability": float(row["prediction_probability"]),
                "calibrated_probability": float(row["calibrated_probability"]),
                "score_final": row["score_final"],
                "score_band": row.get("score_band"),
                "net_positive_contribution": float(row_shap[row_shap > 0].sum()) if np.any(row_shap > 0) else 0.0,
                "net_negative_contribution": float(row_shap[row_shap < 0].sum()) if np.any(row_shap < 0) else 0.0,
                "explanation_summary_text": build_summary_text(reason_rows),
                "explanation_quality_flag": explanation_quality_flag(reason_rows),
                "top_positive_features": ",".join(feature_names_arr[idx] for idx in pos_idx if row_shap[idx] > 0),
                "top_negative_features": ",".join(feature_names_arr[idx] for idx in neg_idx if row_shap[idx] < 0),
            }
            if"ranking_score" in row.index:
                local_record["ranking_score"] = float(row["ranking_score"])
            for rank, reason in enumerate(reason_rows, start=1):
                local_record[f"top{rank}_feature"] = reason["feature_name"]
                local_record[f"top{rank}_feature_value"] = reason["feature_value"]
                local_record[f"top{rank}_shap"] = reason["shap_value"]
                local_record[f"top{rank}_reason_code"] = reason["reason_code"]
                local_record[f"top{rank}_reason_text"] = reason["reason_text"]
                local_record[f"top{rank}_typology"] = reason["typology_name"]
                local_record[f"top{rank}_group_name"] = reason["group_name"]
            local_rows.append(local_record)

        local_df = pd.DataFrame(local_rows)
        local_df.to_parquet(self.run_dirs["base"] / "local_explanations.parquet", index=False)

        # Build SHAP long DataFrame vectorized (avoid n_rows × n_features inner loop)
        n_rows, n_features = shap_values.shape
        id_values = joined_reset[self.config.dataset.id_column].values
        shap_flat = shap_values.flatten()
        feature_value_flat = pd.Series(raw_feature_values.values.flatten())
        feature_value_flat = feature_value_flat.where(feature_value_flat.notna(), None).astype(str)
        shap_long_df = pd.DataFrame({
            "run_id": self.config.run_id,
            self.config.dataset.id_column: np.repeat(id_values, n_features),
            "feature_name": np.tile(feature_names, n_rows),
            "feature_value": feature_value_flat,
            "shap_value": shap_flat,
            "abs_shap_value": np.abs(shap_flat),
            "contribution_direction": np.where(shap_flat >= 0, "positive", "negative"),
        })
        shap_long_df.to_parquet(self.run_dirs["base"] / "shap_values_long.parquet", index=False)
        return {"local": local_df, "shap_long": shap_long_df}

    def _build_global_outputs(
        self,
        model: xgb.Booster,
        feature_specs: list[FeatureSpec],
        feature_meta: pd.DataFrame,
        joined: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        if joined.empty:
            empty = pd.DataFrame()
            empty.to_parquet(self.run_dirs["base"] / "global_feature_importance.parquet", index=False)
            empty.to_parquet(self.run_dirs["base"] / "global_feature_importance_by_segment.parquet", index=False)
            return {"overall": empty, "by_segment": empty}

        feature_names = [spec.name for spec in feature_specs]
        X = self._standardise_features(feature_specs, joined[feature_names].copy())
        shap_values, _ = self._compute_shap(model, feature_names, X)

        overall = pd.DataFrame(
            {
                "run_id": self.config.run_id,
                "feature_name": feature_names,
                "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
                "mean_shap": np.mean(shap_values, axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        overall["rank_overall"] = np.arange(1, len(overall) + 1)
        overall = overall.merge(
            feature_meta.reset_index(drop=True)[["feature_name", "group_name", "typology_name"]],
            on="feature_name",
            how="left",
        )
        overall.to_parquet(self.run_dirs["base"] / "global_feature_importance.parquet", index=False)

        by_segment_rows = []
        for segment_col in self.config.dataset.segment_columns:
            if segment_col not in joined.columns:
                continue
            segment_series = joined[segment_col].fillna("MISSING")
            for segment_value in segment_series.astype(str).unique():
                mask = segment_series.astype(str) == segment_value
                if not mask.any():
                    continue
                segment_mean = np.mean(np.abs(shap_values[mask]), axis=0)
                segment_frame = pd.DataFrame(
                    {
                        "run_id": self.config.run_id,
                        "segment_name": segment_col,
                        "segment_value": segment_value,
                        "feature_name": feature_names,
                        "mean_abs_shap": segment_mean,
                    }
                ).sort_values("mean_abs_shap", ascending=False)
                segment_frame["feature_rank_in_segment"] = np.arange(1, len(segment_frame) + 1)
                by_segment_rows.append(segment_frame)

        by_segment = pd.concat(by_segment_rows, ignore_index=True) if by_segment_rows else pd.DataFrame()
        by_segment.to_parquet(self.run_dirs["base"] / "global_feature_importance_by_segment.parquet", index=False)

        self._write_global_plots(overall, joined, shap_values, feature_names)
        return {"overall": overall, "by_segment": by_segment}

    def _compute_shap(
        self,
        model: xgb.Booster,
        feature_names: list[str],
        X: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        chunk_size = self.config.sampling.shap_chunk_size
        outputs = []
        for start in range(0, len(X), chunk_size):
            end = min(start + chunk_size, len(X))
            dmatrix = xgb.DMatrix(X[start:end], feature_names=feature_names)
            contribs = model.predict(dmatrix, pred_contribs=True)
            # contribs shape: (n_samples, n_features + 1); last col = base value
            outputs.append(contribs[:, :-1])
            if start == 0:
                base_value = float(contribs[0, -1])
        shap_values = np.concatenate(outputs, axis=0)
        logger.info("Computed SHAP for %s rows x %s features", len(X), len(feature_names))
        return shap_values, base_value

    def _write_global_plots(
        self,
        overall: pd.DataFrame,
        joined: pd.DataFrame,
        shap_values: np.ndarray,
        feature_names: list[str],
    ) -> None:
        if overall.empty:
            return

        top = overall.head(20).sort_values("mean_abs_shap", ascending=True)
        plt.figure(figsize=(10, 7))
        plt.barh(top["feature_name"], top["mean_abs_shap"])
        plt.xlabel("Mean absolute SHAP")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(self.run_dirs["plots"] / "global_importance.png", dpi=150)
        plt.close()

        dependence_dir = self.run_dirs["plots"] / "top_feature_dependence_plots"
        dependence_dir.mkdir(parents=True, exist_ok=True)
        for feature_name in overall.head(self.config.sampling.dependence_top_n)["feature_name"]:
            idx = feature_names.index(feature_name)
            plt.figure(figsize=(8, 5))
            plt.scatter(joined[feature_name], shap_values[:, idx], s=10, alpha=0.5)
            plt.xlabel(feature_name)
            plt.ylabel("SHAP value")
            plt.tight_layout()
            plt.savefig(dependence_dir / f"{feature_name}.png", dpi=150)
            plt.close()

    def _write_reporting_pack(self, local_outputs: dict, global_outputs: dict) -> None:
        excel_path = self.run_dirs["reports"] / "explainability_summary.xlsx"
        global_top = global_outputs["overall"].head(100)
        segment_top = global_outputs["by_segment"].head(500)
        local_sample = local_outputs["local"].head(500)
        local_full = local_outputs["local"]
        checks_df = pd.DataFrame(self.validation_checks)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            global_top.to_excel(writer, sheet_name="global_importance", index = False)
            if not global_outputs["by_segment"].empty:
                segment_top.to_excel(
                    writer, sheet_name="importance_by_segment", index=False
                )
            local_sample.to_excel(writer, sheet_name="local_explanations", index=False)
            checks_df.to_excel(writer, sheet_name="validation_checks", index=False)
            
        #CSV exports
        global_top.to_csv(self.run_dirs["reports"] / "global_importance.csv", index = False)
        segment_top.to_csv(self.run_dirs["reports"] / "importance_by_segment.csv", index = False)
        local_full.to_csv(self.run_dirs["reports"] / "local_explanations.csv", index = False)
        checks_df.to_csv(self.run_dirs["reports"] / "validation_checks.csv", index = False)
    
    def _write_top_scored_transactions(self, scored_path: Path, top_n: int=10_000) -> None:
        scored_lazy = self._get_ranked_scored_frame(scored_path)
        available_cols = scored_lazy.collect_schema().names()
        sort_cols, descending = self._get_scored_sort_spec(available_cols)

        if sort_cols:
            top_df = scored_lazy.sort(by=sort_cols, descending=descending).head(top_n).collect()
        else:
            top_df = scored_lazy.head(top_n).collect()
        
        out_path = self.run_dirs["reports"] / f"top_scored_transactions_top{top_n}.csv"
        top_df.write_csv(out_path)
        self._record_check("top_scored_export", "PASS", f"Exported {len(top_df)} rows to {out_path}")

    def _write_mapping_coverage(self, feature_meta: pd.DataFrame) -> dict:
        coverage = feature_meta.reset_index(drop=True).copy()
        coverage["has_feature_dictionary"] = coverage["description"].notna()
        coverage["has_reason_mapping"] = coverage["business_explanation_template"].notna()
        coverage["has_typology_mapping"] = coverage["typology_name"].ne("unmapped")
        coverage["has_feature_grouping"] = coverage["group_name"].ne("unmapped")
        coverage["coverage_ratio"] = coverage[
            [
                "has_feature_dictionary",
                "has_reason_mapping",
                "has_typology_mapping",
                "has_feature_grouping",
            ]
        ].mean(axis=1)
        coverage.to_csv(self.run_dirs["base"] / "mapping_coverage_report.csv", index=False)
        return {
            "feature_count": int(len(coverage)),
            "fully_mapped_features": int((coverage["coverage_ratio"] == 1.0).sum()),
            "top20_fully_mapped": int(
                (coverage.head(20)["coverage_ratio"] == 1.0).sum()
            ),
        }

    def _write_manifest(self, scored_stats: dict, mapping_coverage: dict, feature_specs: list[FeatureSpec]) -> dict:
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            git_hash = "unknown"

        manifest = {
            "run_id": self.config.run_id,
            "model_version": self.config.scoring.model_version,
            "config": self._to_serializable(asdict(self.config)),
            "artifacts": {
                "model_file": str(self.config.paths.model_file),
                "scoring_data_file": str(self.config.paths.scoring_data_file),
                "ranking_score_file": str(self.config.paths.ranking_score_file)
                if self.config.paths.ranking_score_file
                else None,
                "feature_list_file": str(self.config.paths.feature_list_file)
                if self.config.paths.feature_list_file
                else None,
                "metadata_excel_file": str(self.config.paths.metadata_excel_file)
                if self.config.paths.metadata_excel_file
                else None,
                "calibration_file": str(self.config.paths.calibration_file)
                if self.config.paths.calibration_file
                else None,
            },
            "hashes": {
                "model_sha256": file_sha256(self.config.paths.model_file),
                "scoring_data_sha256": file_sha256(self.config.paths.scoring_data_file),
                "ranking_score_sha256": file_sha256(self.config.paths.ranking_score_file),
                "feature_list_sha256": file_sha256(self.config.paths.feature_list_file),
                "metadata_sha256": file_sha256(self.config.paths.metadata_excel_file),
            },
            "feature_count": len(feature_specs),
            "scored_stats": scored_stats,
            "mapping_coverage": mapping_coverage,
            "validation_summary": self.validation_checks,
            "code_version": git_hash,
        }
        write_json(self.run_dirs["base"] / "run_manifest.json", manifest)
        write_yaml(self.run_dirs["base"] / "execution_config.yml", manifest["config"])
        return manifest

    def _write_validation_checks(self) -> None:
        write_validation_checks(
            self.run_dirs["base"] / "validation_checks.csv",
            self.validation_checks,
        )

    def _record_check(self, check_name: str, status: str, detail: str) -> None:
        self.validation_checks.append(
            {
                "check_name": check_name,
                "status": status,
                "detail": detail,
            }
        )

    def _to_serializable(self, value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: self._to_serializable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_serializable(item) for item in value]
        return value


__all__ = ["ExplainabilityService", "create_metadata_template"]
