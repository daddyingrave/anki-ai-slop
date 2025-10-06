"""
Run this from your IDE or CLI. Loads configuration from config-prod.yaml.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from mistral_ocr.config_models import MistralOcrConfig
from mistral_ocr.pdf_to_markdown import convert_pdf

# Default config file path
DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> MistralOcrConfig:
    """Load and validate mistral_ocr configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Validated MistralOcrConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid
        KeyError: If mistral_ocr section is missing
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        yaml_data = yaml.safe_load(f)

    if not yaml_data or "pipelines" not in yaml_data:
        raise KeyError("'pipelines' key not found in configuration file")

    if "mistral_ocr" not in yaml_data["pipelines"]:
        raise KeyError("'mistral_ocr' pipeline not found in configuration file")

    mistral_config_data = yaml_data["pipelines"]["mistral_ocr"]

    try:
        return MistralOcrConfig(**mistral_config_data)
    except ValidationError as e:
        print(f"Configuration validation error:\n{e}")
        raise


def main(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    """Main entry point for Mistral OCR pipeline.

    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)

    print(f"Processing PDF: {config.input_file_path}")
    if config.page_start is not None or config.page_end is not None:
        print(f"Page range: {config.page_start or 0} to {config.page_end or 'end'}")

    # Run conversion
    out_path = convert_pdf(
        input_file_path=config.input_file_path,
        output_dir=config.output_dir,
        output_format=config.output_format,
        override_existing=config.override_existing,
        page_start=config.page_start,
        page_end=config.page_end,
    )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
