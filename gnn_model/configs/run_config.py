import yaml
from .config_base import ConfigBase, Choices, IntField, BoolField, FloatField, StrField, Optional


class DataConfig(ConfigBase):
    window_hours = Optional(IntField(), default=12)
    latent_step_hours = Optional(IntField(), default=3)
    max_rollout_steps = Optional(IntField(), default=1)
    zarr_root = Optional(StrField(), default="/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7")


class TrainingConfig(ConfigBase):
    lr = Optional(FloatField(), default=5e-4)
    lr_scheduler = Optional(Choices(['plateau', 'cosine_warmup']), default='plateau')
    weight_decay = Optional(FloatField(), default=1e-5)
    max_epochs = Optional(IntField())
    precision = Optional(Choices(['16-mixed', 'fp32', 'fp16']), default='16-mixed')


class LossConfig(ConfigBase):
    type = Optional(Choices(['mse', 'huber']), default='mse')
    huber_delta = Optional(IntField())


class ScheduleConfig(ConfigBase):
    type = Choices(['plateau', 'cosine_warmup'])
    warmup_pct = Optional(FloatField(), default=0.05)
    warmup_start_factor = Optional(FloatField(), default=0.01)
    min_lr = Optional(FloatField(), default=1e-6)


class RunConfig(ConfigBase):
    experiment_name = StrField()
    checkpoint_dir = StrField()

    switch_to_sequential_after_epochs = Optional(IntField())
    auto_switch_to_sequential = Optional(BoolField(), default=False)
    auto_switch_to_metric = Optional(Choices(['val_loss']), default='val_loss')
    auto_switch_patience_epochs = Optional(IntField(), default=10)
    auto_switch_min_delta = Optional(FloatField(), default=0.0)


    data = DataConfig()
    training = TrainingConfig()
    loss = LossConfig()
    schedule = ScheduleConfig()

    def __init__(self, config_path: str):
        super().load(yaml.safe_load(open(config_path)))
