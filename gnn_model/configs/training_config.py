from datetime import timedelta

import yaml

from .config_base import (
    BoolField,
    Choices,
    ConfigBase,
    DatetimeField,
    FloatField,
    IntField,
    Optional,
    StrField,
)


class SplitRatioDateRangeConfig(ConfigBase):
    full_start_date = Optional(DatetimeField(), default="2015-01-01")
    full_end_date = Optional(DatetimeField(), default="2025-01-01")
    train_val_split_ratio = Optional(FloatField(), default=0.9)

    @property
    def train_start_date(self):
        return self.full_start_date

    @property
    def train_end_date(self):
        total_days = (self.full_end_date - self.full_start_date).days
        train_days = int(total_days * self.train_val_split_ratio)
        return self.full_start_date + timedelta(days=train_days)

    @property
    def val_start_date(self):
        return self.train_end_date

    @property
    def val_end_date(self):
        return self.full_end_date


class PredefinedDateRangeConfig(ConfigBase):
    train_start_date = Optional(DatetimeField(), default="2015-01-01")
    train_end_date = Optional(DatetimeField(), default="2025-01-01")
    val_start_date = Optional(DatetimeField(), default="2025-01-01")
    val_end_date = Optional(DatetimeField(), default="2025-12-31")


class SamplingConfig(ConfigBase):
    date_range = Choices({
        'split_ratio': SplitRatioDateRangeConfig(),
        'predefined': PredefinedDateRangeConfig(),
    })


class RandomSamplingConfig(SamplingConfig):
    train_window_days = Optional(IntField(), default=14)
    seed = Optional(IntField())
    sequential_stride_days = Optional(IntField(), default=1)
    switch_to_sequential_after_epochs = Optional(IntField())
    auto_switch_to_sequential = Optional(BoolField(), default=False)
    auto_switch_metric = Optional(Choices(['val_loss']), default='val_loss')
    auto_switch_patience_epochs = Optional(IntField(), default=10)
    auto_switch_min_delta = Optional(FloatField(), default=0.0)


class SequentialSamplingConfig(SamplingConfig):
    window_days = Optional(IntField(), default=7)
    stride_days = Optional(IntField(), default=1)


class TrainingDataConfig(ConfigBase):
    path = StrField()
    sampler = Choices({
        'random': RandomSamplingConfig(),
        'sequential': SequentialSamplingConfig(),
    })
    window_hours = Optional(IntField(), default=12)
    latent_step_hours = Optional(IntField(), default=3)
    max_rollout_steps = Optional(IntField(), default=1)
    batch_size = Optional(IntField(), default=1)
    num_neighbors = Optional(IntField(), default=3)

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        if self.window_hours % self.latent_step_hours != 0:
            raise ValueError(
                f"window_hours ({self.window_hours}) must be divisible by "
                f"latent_step_hours ({self.latent_step_hours})"
            )


class EarlyStoppingConfig(ConfigBase):
    enabled = Optional(BoolField(), default=True)
    patience = Optional(IntField(), default=25)
    min_delta = Optional(FloatField(), default=1e-5)
    start_epoch = Optional(IntField(), default=0)


class TrainerConfig(ConfigBase):
    max_epochs = Optional(IntField(), default=100)
    accelerator = Optional(Choices(['auto', 'cpu', 'gpu']), default='auto')
    devices = Optional(IntField(), default=1)
    num_nodes = Optional(IntField(), default=1)
    precision = Optional(
        Choices(['16-mixed', 'bf16-mixed', '16-true', '32-true', '64-true']),
        default='16-mixed',
    )
    gradient_clip_val = Optional(FloatField(), default=0.5)
    log_every_n_steps = Optional(IntField(), default=1)
    num_sanity_val_steps = Optional(IntField(), default=2)
    limit_train_batches = Optional(IntField())
    limit_val_batches = Optional(IntField())
    early_stopping = Optional(EarlyStoppingConfig(), default={})


class OptimizerConfig(ConfigBase):
    pass


class AdamWConfig(OptimizerConfig):
    lr = Optional(FloatField(), default=5e-4)
    weight_decay = Optional(FloatField(), default=1e-5)


class CsvOutputConfig(ConfigBase):
    output_dir = Optional(StrField(), default='val_csv')
    disable = Optional(BoolField(), default=False)
    num_batches = Optional(IntField(), default=1)
    every_n_epochs = Optional(IntField(), default=1)
    max_rows = Optional(IntField())
    seed = Optional(IntField(), default=0)


class ValidationModeConfig(ConfigBase):
    window_days = Optional(IntField(), default=8)
    update_every_n_epochs = Optional(IntField(), default=5)


class FixedValidationConfig(ConfigBase):
    window_days = Optional(IntField(), default=8)


class RandomValidationConfig(ValidationModeConfig):
    pass


class SequentialValidationConfig(ValidationModeConfig):
    stride_days = Optional(IntField(), default=8)


class ValidationConfig(ConfigBase):
    mode = Choices({
        'fixed': FixedValidationConfig(),
        'random': RandomValidationConfig(),
        'sequential': SequentialValidationConfig(),
    })
    cache_windows = Optional(BoolField(), default=True)
    cache_max_entries = Optional(IntField(), default=16)
    csv = Optional(CsvOutputConfig(), default={})


class LossConfig(ConfigBase):
    pass


class MseLossConfig(LossConfig):
    pass


class HuberLossConfig(LossConfig):
    delta = Optional(FloatField(), default=0.1)


class ScheduleConfig(ConfigBase):
    pass


class PlateauScheduleConfig(ScheduleConfig):
    factor = Optional(FloatField(), default=0.5)
    patience = Optional(IntField(), default=3)
    min_lr = Optional(FloatField(), default=1e-6)


class CosineWarmupScheduleConfig(ScheduleConfig):
    warmup_pct = Optional(FloatField(), default=0.05)
    warmup_start_factor = Optional(FloatField(), default=0.01)
    min_lr = Optional(FloatField(), default=1e-6)


class TrainingConfig(ConfigBase):
    experiment_name = StrField()
    checkpoint_dir = StrField()
    observation_config_path = Optional(StrField(), default='configs/observation_config.yaml')
    mesh_variable_config_path = Optional(StrField(), default='configs/mesh_config.yaml')
    seed = Optional(IntField())
    verbose = Optional(BoolField(), default=False)
    debug_mode = Optional(BoolField(), default=False)
    resume_from_checkpoint = Optional(StrField())
    resume_from_latest = Optional(BoolField(), default=False)
    load_weights_only = Optional(BoolField(), default=False)
    data = TrainingDataConfig()
    trainer = TrainerConfig()
    validation = ValidationConfig()
    optimizer = Choices({'adamw': AdamWConfig()})
    loss = Choices({
        'mse': MseLossConfig(),
        'huber': HuberLossConfig(),
    })
    schedule = Choices({
        'plateau': PlateauScheduleConfig(),
        'cosine_warmup': CosineWarmupScheduleConfig(),
    })

    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as config_file:
            self.load(yaml.safe_load(config_file))
