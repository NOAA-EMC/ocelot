from copy import deepcopy

import yaml

from .config_base import (
    BoolField,
    Choices,
    ConfigBase,
    ConfigField,
    FloatField,
    IntField,
    ListField,
    Optional,
    StrField,
)


class RawField(ConfigField):
    def load(self, value):
        self.value = deepcopy(value)


class MappingField(ConfigField):
    def __init__(self, value_type: ConfigField | ConfigBase):
        super().__init__()
        self.value_type = value_type

    def load(self, value):
        if not isinstance(value, dict):
            raise ValueError(f"Expected value of type dict but got {type(value).__name__}")

        for item in value.values():
            validator = deepcopy(self.value_type)
            validator.load(item)
        self.value = deepcopy(value)


class StructuredField(ConfigField):
    def __init__(self, config_type: ConfigBase):
        super().__init__()
        self.config_type = config_type

    def load(self, value):
        validator = deepcopy(self.config_type)
        validator.load(value)
        self.value = deepcopy(value)


class SubsampleModeConfig(ConfigBase):
    satellite = Optional(MappingField(Choices(['stride', 'random', 'none'])), default={})
    conventional = Optional(MappingField(Choices(['stride', 'random', 'none'])), default={})
    snow_cover = Optional(Choices(['stride', 'random', 'none']))


class SubsampleConfig(ConfigBase):
    seed = Optional(IntField(), default=12345)
    satellite = Optional(MappingField(IntField()), default={})
    conventional = Optional(MappingField(IntField()), default={})
    snow_cover = Optional(IntField())
    mode = Optional(StructuredField(SubsampleModeConfig()), default={})


class PipelineConfig(ConfigBase):
    subsample = Optional(StructuredField(SubsampleConfig()), default={})


class QualityFilterConfig(ConfigBase):
    range = Optional(ListField(FloatField()))
    clip = Optional(BoolField(), default=False)
    qm_flag_col = Optional(StrField())
    keep = Optional(ListField(IntField()))
    reject = Optional(ListField(IntField()))
    strict_flags = Optional(BoolField(), default=False)


class LevelSelectionConfig(ConfigBase):
    filter_col = StrField()
    matching_mode = Choices(['all', 'nearest', 'exact'])


class InstrumentConfig(ConfigBase):
    source = Optional(StrField())
    zarr_name = Optional(StrField())
    sat_ids = Optional(ListField(IntField()))
    scan_angle_channels = Optional(IntField(), default=1)
    features = ListField(StrField())
    metadata = ListField(StrField())
    input_dim = IntField()
    target_dim = IntField()
    encoder_hidden_layers = IntField()
    decoder_hidden_layers = IntField()
    qc_strict_flags = Optional(BoolField(), default=False)
    qc_filters = Optional(MappingField(QualityFilterConfig()), default={})
    qc_relations = Optional(MappingField(RawField()), default={})
    level_selection = Optional(StructuredField(LevelSelectionConfig()))


class ObservationGroupsConfig(ConfigBase):
    satellite = Optional(MappingField(InstrumentConfig()), default={})
    conventional = Optional(MappingField(InstrumentConfig()), default={})


class MeshVariablesConfig(ConfigBase):
    enable_mesh_pred = Optional(BoolField(), default=True)
    variables = MappingField(ListField(StrField()))
    mesh_pressure_level_idx = Optional(IntField(), default=0)


class ObservationConfig(ConfigBase):
    pipeline = Optional(StructuredField(PipelineConfig()), default={})
    instrument_weights = MappingField(FloatField())
    channel_weights = MappingField(ListField(FloatField()))
    observation_config = StructuredField(ObservationGroupsConfig())
    feature_stats = MappingField(MappingField(ListField(FloatField())))
    mesh_config = Optional(StructuredField(MeshVariablesConfig()), default={})

    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as config_file:
            self.load(yaml.safe_load(config_file))
