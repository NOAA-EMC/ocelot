import re
from types import SimpleNamespace

import pytest
import yaml

from ocelot.configs.config_base import ConfigError
from ocelot.configs.pipeline_config import PipelineConfig, SubsamplingConfig


class InstrumentCatalog:
    def __init__(self, instruments):
        self._instruments = instruments

    @property
    def names(self):
        return set(self._instruments)

    def get(self, name):
        return self._instruments[name]


@pytest.fixture
def catalog():
    return InstrumentCatalog(
        {
            "atms": SimpleNamespace(kind="satellite", feature_names=["brightness_temperature"]),
            "aircraft": SimpleNamespace(kind="conventional", feature_names=["air_temperature"]),
        }
    )


def make_subsampling(satellite=None, conventional=None):
    config = SubsamplingConfig()
    config.load(
        {
            "satellite": satellite
            or {"_default": {"factor": 30, "mode": "random"}},
            "conventional": conventional
            or {"_default": {"factor": 1, "mode": "stride"}},
        }
    )
    return config


def test_subsampling_resolves_overrides_and_defaults(catalog):
    config = make_subsampling(
        satellite={
            "_default": {"factor": 30, "mode": "random"},
            "atms": {"factor": 6},
        }
    )

    config.validate_instruments(catalog)

    assert config.resolve("atms", "satellite").factor == 6
    assert config.resolve("atms", "satellite").mode == "random"
    assert config.resolve("aircraft", "conventional").factor == 1


@pytest.mark.parametrize(
    ("satellite", "message"),
    [
        (
            {
                "_default": {"factor": 30, "mode": "random"},
                "missing": {"factor": 2},
            },
            "Unknown satellite sampling instrument(s): missing",
        ),
        (
            {
                "_default": {"factor": 30, "mode": "random"},
                "aircraft": {"factor": 2},
            },
            "Instrument(s) in the satellite sampling group have a different kind: aircraft",
        ),
    ],
)
def test_subsampling_validates_its_catalog_references(catalog, satellite, message):
    config = make_subsampling(satellite=satellite)

    with pytest.raises(ConfigError, match=re.escape(message)):
        config.validate_instruments(catalog)


def test_pipeline_delegates_subsampling_catalog_validation(tmp_path, catalog):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "enabled_instruments": ["atms"],
                "subsampling": {
                    "satellite": {
                        "_default": {"factor": 30, "mode": "random"},
                        "missing": {"factor": 2},
                    },
                    "conventional": {
                        "_default": {"factor": 1, "mode": "stride"}
                    },
                },
                "outputs": {},
            }
        )
    )

    config = PipelineConfig(str(config_path))

    with pytest.raises(
        ConfigError,
        match=re.escape("Unknown satellite sampling instrument(s): missing"),
    ):
        config.validate_instruments(catalog)