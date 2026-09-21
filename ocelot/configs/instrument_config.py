import yaml

from ocelot.configs.config_base import (
    BoolField,
    Choices,
    ConfigBase,
    FloatField,
    IntField,
    ListField,
    MapField,
    Optional,
    StrField,
)


class NormalizationConfig(ConfigBase):
    mean = FloatField()
    std = FloatField()

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        if self.std <= 0:
            raise ValueError("Normalization std must be greater than zero")


class VariableConfig(ConfigBase):
    units = StrField()
    normalization = NormalizationConfig()


class FeatureConfig(VariableConfig):
    name = StrField()
    weight = Optional(FloatField(), default=1.0)


class InstrumentModelConfig(ConfigBase):
    encoder_hidden_layers = IntField()
    decoder_hidden_layers = IntField()

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        if self.encoder_hidden_layers < 1 or self.decoder_hidden_layers < 1:
            raise ValueError("Instrument model layer counts must be positive")


class QualityFilterConfig(ConfigBase):
    units = Optional(StrField())
    range = Optional(ListField(FloatField()))
    clip = Optional(BoolField(), default=False)
    qm_flag_col = Optional(StrField())
    keep = Optional(ListField(IntField()))
    reject = Optional(ListField(IntField()))
    require_flag_column = Optional(BoolField(), default=False)

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        if self.range is not None and len(self.range) != 2:
            raise ValueError("QC range must contain exactly two values")
        if self.keep is not None and self.reject is not None:
            raise ValueError("QC filter cannot define both keep and reject")


class PressureHeightRelationConfig(ConfigBase):
    enable = Optional(BoolField(), default=False)
    scale_height_m = Optional(FloatField(), default=8000.0)
    tolerance_hpa = Optional(FloatField(), default=100.0)


class QualityRelationsConfig(ConfigBase):
    dewpoint_le_temp = Optional(BoolField(), default=False)
    max_temp_dewpoint_spread = Optional(FloatField())
    rh_from_td_consistency_pct = Optional(FloatField())
    pressure_vs_height = Optional(PressureHeightRelationConfig())


class LevelSelectionConfig(ConfigBase):
    filter_col = StrField()
    matching_mode = StrField()
    levels = Optional(ListField(FloatField()), default=[])

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        if self.matching_mode not in {'all', 'nearest', 'exact'}:
            raise ValueError(
                "matching_mode must be one of: all, nearest, exact"
            )


class InstrumentConfig(ConfigBase):
    kind = Choices(['satellite', 'conventional'])
    weight = Optional(FloatField(), default=1.0)
    source = Optional(StrField())
    zarr_name = Optional(StrField())
    satellite_ids = Optional(ListField(IntField()))
    scan_angle_channels = Optional(IntField(), default=1)
    metadata = Optional(ListField(StrField()), default=[])
    model = InstrumentModelConfig()
    features = ListField(FeatureConfig())
    require_all_flag_columns = Optional(BoolField(), default=False)
    qc_filters = Optional(MapField(QualityFilterConfig()), default={})
    qc_relations = Optional(QualityRelationsConfig(), default={})
    level_selection = Optional(LevelSelectionConfig())
    auxiliary_variables = Optional(MapField(VariableConfig()), default={})

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        feature_names = self.feature_names
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Instrument feature names must be unique")
        if self.scan_angle_channels < 1:
            raise ValueError("scan_angle_channels must be positive")

    @property
    def feature_names(self) -> list[str]:
        return [feature.name for feature in self.features]

    @property
    def target_dim(self) -> int:
        return len(self.features)

    @property
    def input_dim(self) -> int:
        satellite_id_dim = len(self.satellite_ids or [])
        return 7 + len(self.metadata) + self.target_dim + satellite_id_dim

    @property
    def feature_stats(self) -> dict[str, list[float]]:
        stats = {
            feature.name: [feature.normalization.mean, feature.normalization.std]
            for feature in self.features
        }
        stats.update({
            name: [variable.normalization.mean, variable.normalization.std]
            for name, variable in self.auxiliary_variables.items()
        })
        return stats

    @property
    def channel_weights(self) -> list[float]:
        return [feature.weight for feature in self.features]


class InstrumentCatalogConfig(ConfigBase):
    instruments = MapField(InstrumentConfig())

    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as config_file:
            self.load(yaml.safe_load(config_file))

    def get(self, name: str) -> InstrumentConfig:
        try:
            return self.instruments[name]
        except KeyError as exc:
            raise KeyError(f"Unknown instrument: {name}") from exc

    def items(self):
        yield from self.instruments.items()

    @property
    def names(self) -> set[str]:
        return set(self.instruments)


InstrumentCatalog = InstrumentCatalogConfig
