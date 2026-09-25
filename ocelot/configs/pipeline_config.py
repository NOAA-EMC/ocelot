from dataclasses import dataclass

import yaml

from ocelot.configs.config_base import (
    BoolField,
    ConfigBase,
    FloatField,
    IntField,
    ListField,
    MapField,
    Optional,
    StrField,
    ConfigError,
)


SAMPLING_MODES = {'stride', 'random', 'none'}
STANDARD_PRESSURE_LEVELS = [
    1000, 925, 850, 700, 500, 400, 300, 250,
    200, 150, 100, 70, 50, 30, 20, 10,
]


class SamplingPolicyConfig(ConfigBase):
    factor = Optional(IntField())
    mode = Optional(StrField())

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        if self.factor is not None and self.factor < 1:
            raise ConfigError("Sampling factor must be positive")
        if self.mode is not None and self.mode not in SAMPLING_MODES:
            raise ConfigError(
                f"Sampling mode must be one of: {', '.join(sorted(SAMPLING_MODES))}"
            )


@dataclass(frozen=True)
class ResolvedSamplingPolicy:
    factor: int
    mode: str


class SubsamplingConfig(ConfigBase):
    seed = Optional(IntField(), default=12345)
    satellite = MapField(SamplingPolicyConfig())
    conventional = MapField(SamplingPolicyConfig())

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        for kind, policies in self._policy_groups():
            if "_default" not in policies:
                raise ConfigError(
                    f"Missing '_default' entry in {kind} sampling policies"
                )
            default = policies["_default"]
            if default.factor is None or default.mode is None:
                raise ConfigError(
                    f"The {kind} default sampling policy requires factor and mode"
                )
            for name, policy in policies.items():
                if name != "_default" and policy.factor is None and policy.mode is None:
                    raise ConfigError(
                        f"Sampling override for {name} must set factor or mode"
                    )

    def validate_instruments(self, catalog) -> None:
        for kind, policies in self._policy_groups():
            instrument_names = set(policies) - {"_default"}
            unknown = instrument_names - catalog.names
            if unknown:
                raise ConfigError(
                    f"Unknown {kind} sampling instrument(s): "
                    f"{', '.join(sorted(unknown))}"
                )

            wrong_kind = {
                name
                for name in instrument_names
                if catalog.get(name).kind != kind
            }
            if wrong_kind:
                raise ConfigError(
                    f"Instrument(s) in the {kind} sampling group have a different "
                    f"kind: {', '.join(sorted(wrong_kind))}"
                )

    def resolve(self, instrument: str, kind: str) -> ResolvedSamplingPolicy:
        policies_by_kind = dict(self._policy_groups())
        if kind not in policies_by_kind:
            raise ConfigError("Kind must be either 'satellite' or 'conventional'")

        policies = policies_by_kind[kind]
        default = policies["_default"]
        override = policies.get(instrument)
        return ResolvedSamplingPolicy(
            factor=(
                override.factor
                if override is not None and override.factor is not None
                else default.factor
            ),
            mode=(
                override.mode
                if override is not None and override.mode is not None
                else default.mode
            ),
        )

    def _policy_groups(self):
        return (
            ("satellite", self.satellite),
            ("conventional", self.conventional),
        )



class MeshPredictionConfig(ConfigBase):
    enabled = Optional(BoolField(), default=False)
    pressure_level = Optional(FloatField(), default=1000)
    variables = Optional(MapField(ListField(StrField())), default={})

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        if self.pressure_level not in STANDARD_PRESSURE_LEVELS:
            levels = ', '.join(str(level) for level in STANDARD_PRESSURE_LEVELS)
            raise ConfigError(f"pressure_level must be one of: {levels} hPa")

    @property
    def pressure_level_index(self) -> int:
        return STANDARD_PRESSURE_LEVELS.index(self.pressure_level)

    def validate_instruments(self, catalog) -> None:
        for name, variables in self.variables.items():
            if name not in catalog.names:
                raise ConfigError(
                    f"Mesh prediction references unknown instrument: {name}"
                )
            instrument = catalog.get(name)
            unknown_variables = set(variables) - set(instrument.feature_names)
            if unknown_variables:
                raise ConfigError(
                    f"Unknown mesh variable(s) for {name}: "
                    f"{', '.join(sorted(unknown_variables))}"
                )


class OutputConfig(ConfigBase):
    mesh_prediction = Optional(MeshPredictionConfig(), default={})


class PipelineConfig(ConfigBase):
    enabled_instruments = ListField(StrField())
    subsampling = SubsamplingConfig()
    outputs = Optional(OutputConfig(), default={})

    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as config_file:
            self.load(yaml.safe_load(config_file))

    def validate_instruments(self, catalog) -> None:
        if len(self.enabled_instruments) != len(set(self.enabled_instruments)):
            raise ConfigError("Enabled instrument names must be unique")
        unknown = set(self.enabled_instruments) - catalog.names
        if unknown:
            raise ConfigError(
                "Unknown enabled instrument(s): "
                f"{', '.join(sorted(unknown))}"
            )

        self.subsampling.validate_instruments(catalog)
        self.outputs.mesh_prediction.validate_instruments(catalog)

    def enabled(self, catalog):
        self.validate_instruments(catalog)
        for name in self.enabled_instruments:
            yield name, catalog.get(name)

    def instrument_name_to_id(self, catalog) -> dict[str, int]:
        self.validate_instruments(catalog)
        return {name: index for index, name in enumerate(self.enabled_instruments)}

    def instrument_weights(self, catalog) -> dict[int, float]:
        name_to_id = self.instrument_name_to_id(catalog)
        return {
            name_to_id[name]: instrument.weight
            for name, instrument in self.enabled(catalog)
        }

    def channel_weights(self, catalog):
        name_to_id = self.instrument_name_to_id(catalog)
        return {
            name_to_id[name]: instrument.channel_weights
            for name, instrument in self.enabled(catalog)
        }

    def feature_stats(self, catalog) -> dict[str, dict[str, list[float]]]:
        return {
            name: instrument.feature_stats
            for name, instrument in self.enabled(catalog)
        }
