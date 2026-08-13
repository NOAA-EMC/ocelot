import yaml

from .config_base import (
    BoolField,
    ConfigBase,
    DatetimeField,
    IntField,
    Optional,
    StrField,
)


class InferenceDataConfig(ConfigBase):
    path = Optional(StrField())
    start_date = DatetimeField()
    end_date = DatetimeField()
    batch_size = Optional(IntField(), default=1)

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        if self.start_date >= self.end_date:
            raise ValueError("Inference start_date must be before end_date")


class InferenceResourcesConfig(ConfigBase):
    devices = Optional(IntField(), default=1)
    num_nodes = Optional(IntField(), default=1)
    limit_batches = Optional(IntField())


class InferenceConfig(ConfigBase):
    experiment_name = StrField()
    checkpoint = StrField()
    output_dir = Optional(StrField(), default='predictions')
    verbose = Optional(BoolField(), default=False)
    eval_mode = Optional(BoolField(), default=False)
    data = InferenceDataConfig()
    resources = Optional(InferenceResourcesConfig(), default={})

    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as config_file:
            self.load(yaml.safe_load(config_file))
