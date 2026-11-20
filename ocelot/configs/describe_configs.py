"""Display configuration schemas: python -m ocelot.configs.describe_configs."""

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ocelot.configs.config_base import ConfigBase
from ocelot.configs.inference_config import InferenceConfig
from ocelot.configs.instrument_config import InstrumentCatalogConfig
from ocelot.configs.model_config import ModelConfig
from ocelot.configs.pipeline_config import PipelineConfig
from ocelot.configs.training_config import TrainingConfig


def main(argv: list[str] | None = None) -> None:
    config_types: dict[str, type[ConfigBase]] = {
        "model": ModelConfig,
        "training": TrainingConfig,
        "pipeline": PipelineConfig,
        "instrument": InstrumentCatalogConfig,
        "inference": InferenceConfig,
    }
    parser = argparse.ArgumentParser(
        description="Display configuration fields, defaults, and nested choices."
    )
    parser.add_argument(
        "config",
        nargs="?",
        choices=["all", *config_types],
        help="configuration type to describe; use 'all' to display every type",
    )
    args = parser.parse_args(argv)
    if args.config is None:
        parser.print_help()
        return
    selected = (
        config_types.values() if args.config == "all" else [config_types[args.config]]
    )
    for config_type in selected:
        print(config_type.describe(recursive=True))


if __name__ == "__main__":
    main()
