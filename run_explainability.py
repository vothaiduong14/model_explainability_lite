"""Run the standalone explainability module.

Usage:
    python -m model_explainability.run_explainability --config model_explainability/configs/explainability_config.yaml
    python -m model_explainability.run_explainability --create-template model_explainability/metadata/explainability_metadata.xlsx
"""

from __future__ import annotations

import argparse

from model_explainability.config import ExplainabilityConfig
from model_explainability.metadata import create_metadata_template
from model_explainability.service import ExplainabilityService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone explainability")
    parser.add_argument("--config", help="Path to explainability YAML config")
    parser.add_argument(
        "--create-template",
        help="Optional path for generating a blank metadata Excel template",
    )
    args = parser.parse_args()

    if args.create_template:
        create_metadata_template(args.create_template)
        print(f"Metadata template written to {args.create_template}")
        return

    if not args.config:
        parser.error("--config is required unless --create-template is provided")

    config = ExplainabilityConfig.from_yaml(args.config)
    service = ExplainabilityService(config)
    manifest = service.run()
    print(f"Explainability run complete: {manifest['run_id']}")


if __name__ == "__main__":
    main()
