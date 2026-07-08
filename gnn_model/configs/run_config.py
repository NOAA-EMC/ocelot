import yaml
from .config_base import ConfigBase, Choices, IntField, FloatField, StrField, Optional


class DataConfig(ConfigBase):
    window_hours = IntField()
    latent_step_hours = IntField()
    max_rollout_steps = IntField()
    zarr_root = StrField()


class TrainingConfig(ConfigBase):
    learning_rate = FloatField()
    weight_decay = FloatField()
    max_epochs = IntField()
    precision = Choices(['fp32', 'fp16', '16-mixed'])


class LossConfig(ConfigBase):
    type = Choices(['mse', 'huber'])
    huber_delta = Optional(IntField())


class ScheduleConfig(ConfigBase):
    type = Choices(['plateau', 'cosine_warmup'])
    warmup_pct = FloatField()
    min_lr = FloatField()


class RunConfig(ConfigBase):
    experiment_name = StrField()
    checkpoint_dir = StrField()
    data = DataConfig()
    training = TrainingConfig()
    loss = LossConfig()
    schedule = ScheduleConfig()

    def __init__(self, config_path: str):
        super().load(yaml.safe_load(open(config_path)))
