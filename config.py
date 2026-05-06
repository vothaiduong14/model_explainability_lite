"""Configuration models for the standalone explainability module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

def _read_text_with_fallback(path:str | Path) -> str:
    p = Path(path)
    raw = p.read_bytes()
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

@dataclass
class ExplainabilityPaths:
    model_file: Path | None = None
    scoring_data_file: Path | None = None
    ranking_score_file: Path | None = None
    feature_list_file: Path | None = Path("artifacts/run_default/final/model_signature.json")
    metadata_excel_file: Path | None = None
    calibration_file: Path | None = None
    output_root: Path = Path("model_explainability/outputs")
    _config_dir: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.model_file = Path(self.model_file)
        self.scoring_data_file = Path(self.scoring_data_file)
        self.ranking_score_file = Path(self.ranking_score_file) if self.ranking_score_file else None
        self.feature_list_file = Path(self.feature_list_file) if self.feature_list_file else None
        self.metadata_excel_file = (
            Path(self.metadata_excel_file) if self.metadata_excel_file else None
        )
        self.calibration_file = (
            Path(self.calibration_file) if self.calibration_file else None
        )
        self.output_root = Path(self.output_root)

    @staticmethod
    def _resolve_path(path_value: Path, base_dirs: list[Path]) -> Path:
        if path_value.is_absolute():
            return path_value
        
        candidates: list[Path] = []
        for base_dir in base_dirs:
            candidate = (base_dir / path_value).resolve()
            if candidate.exists():
                return candidate
            candidates.append(candidate)

        return candidates[0] if candidates else path_value        

    def resolve_paths(self, config_file_path: Path | None = None) -> None:
        if config_file_path is None:
            return
        
        config_path = Path(config_file_path).resolve()
        config_dir = config_path.parent
        module_dir = config_dir.parent
        workspace_dir = module_dir.parent
        self._config_dir = config_dir

        base_dirs = [config_dir, module_dir, workspace_dir]

        self.model_file = self._resolve_path(self.model_file, base_dirs)
        self.scoring_data_file = self._resolve_path(self.scoring_data_file, base_dirs)

        if self.ranking_score_file:
            self.ranking_score_file = self._resolve_path(self.ranking_score_file, base_dirs)
        if not self.model_file.is_absolute():
            self.model_file = (config_dir / self.model_file).resolve()


        if self.feature_list_file:
            self.feature_list_file = self._resolve_path(self.scoring_data_file, base_dirs)

        if self.metadata_excel_file:
            self.metadata_excel_file = self._resolve_path(self.metadata_excel_file, base_dirs)

        if self.calibration_file:
            self.calibration_file = self._resolve_path(self.calibration_file, base_dirs)


@dataclass
class DatasetConfig:
    id_column: str = "transaction_id"
    score_ts_column: str = "txn_timestamp"
    label_column: str | None = "label"
    segment_columns: list[str] = field(default_factory=lambda: ["channel", "customer_tier"])
    allow_duplicate_ids: bool = False
    allow_extra_columns: bool = True


@dataclass
class SamplingConfig:
    scoring_chunk_rows: int = 250_000
    shap_chunk_size: int = 20_000
    global_sample_size: int = 50_000
    global_top_score_percentile: float = 0.01
    include_all_positive_labels_in_global: bool = True
    local_top_score_rows: int = 10_000
    local_max_rows: int = 10_000
    local_non_alert_sample_size: int = 500
    include_all_alerts_in_local: bool = True
    include_all_positive_labels_in_local: bool = True
    dependence_top_n: int = 5
    random_seed: int = 42


@dataclass
class ScoringConfig:
    model_version: str = "xgb_fraud_model"
    score_scale: str = "0_1000"
    alert_probability_threshold: float | None = 0.5
    ranking_score_column: str | None = None
    ranking_score_id_column: str | None = None
    top_n_reasons: int = 5
    top_n_negative_reasons: int = 2


@dataclass
class ValidationConfig:
    null_warning_threshold: float = 0.80
    shap_reconstruction_tolerance: float = 1e-4
    required_metadata_tabs: list[str] = field(default_factory=lambda: ["feature_dictionary"])


@dataclass
class ExplainabilityConfig:
    run_id: str | None = None
    paths: ExplainabilityPaths = field(default_factory=ExplainabilityPaths)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self):
        if isinstance(self.paths, dict):
            self.paths = ExplainabilityPaths(**self.paths)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(**self.dataset)
        if isinstance(self.sampling, dict):
            self.sampling = SamplingConfig(**self.sampling)
        if isinstance(self.scoring, dict):
            self.scoring = ScoringConfig(**self.scoring)
        if isinstance(self.validation, dict):
            self.validation = ValidationConfig(**self.validation)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExplainabilityConfig":
        config_path = Path(path)
        raw = yaml.safe_load(_read_text_with_fallback(config_path)) or {}
        instance = cls(**raw)

        instance.paths.resolve_paths(config_path)
        return instance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
