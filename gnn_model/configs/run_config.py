import yaml
from .config_base import ConfigBase, \
                         BoolField, \
                         Choices, \
                         IntField, \
                         FloatField,\
                         StrField, \
                         DatetimeField, \
                         Optional 
                        

class SequentialSamplingConfig(ConfigBase):
    start_date = DatetimeField()
    end_date = DatetimeField()


class RandomSamplingConfig(ConfigBase):
    start_date = DatetimeField()
    end_date = DatetimeField()
    seed = Optional(IntField())


class DataConfig(ConfigBase):
    path = StrField()

    sampler = Choices({'random': RandomSamplingConfig(), 
                       'sequential': SequentialSamplingConfig()})
    
    window_hours = Optional(IntField(), default=12)
    latent_step_hours = Optional(IntField(), default=3)
    max_rollout_steps = Optional(IntField(), default=1)
    zarr_root = Optional(StrField(), default="/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7")


class TrainingConfig(ConfigBase):
    # start_date = DatetimeField()
    # end_date = DatetimeField()

    use_split_ratio = Optional(BoolField(), default=False)
    train_ratio = Optional(FloatField(), default=0.9)

    window_days = Optional(IntField(), default=12)
    
    lr = Optional(FloatField(), default=5e-4)
    lr_scheduler = Optional(Choices(['plateau', 'cosine_warmup']), default='plateau')
    weight_decay = Optional(FloatField(), default=1e-5)
    max_epochs = Optional(IntField(), default=0)
    precision = Optional(Choices(['16-mixed', 'fp32', 'fp16']), default='16-mixed')


class CsvOutputConfig(ConfigBase):
    output_dir = StrField()

    disable = Optional(BoolField(), default=False)
    num_batches = Optional(IntField(), default=3)
    every_n_epochs = Optional(IntField(), default=10)
    max_rows = Optional(IntField())
    seed = Optional(IntField())


class ValidationConfig(ConfigBase):
    start_date = DatetimeField()
    end_date = DatetimeField()

    mode = Optional(Choices(['fixed', 'random', 'sequential']), default='sequential')
    window_days = Optional(IntField(), default=12)
    stride_days = Optional(IntField(), default=8)
    update_every_n_epochs = Optional(IntField(), default=5)

    cache_windows = Optional(BoolField(), default=True)
    cache_max_entries = Optional(IntField(), default=16)

    csv = CsvOutputConfig()


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

    stride_days = Optional(IntField(), default=1)
    seed = Optional(IntField())

    switch_to_sequential_after_epochs = Optional(IntField())
    auto_switch_to_sequential = Optional(BoolField(), default=False)
    auto_switch_to_metric = Optional(Choices(['val_loss']), default='val_loss')
    auto_switch_patience_epochs = Optional(IntField(), default=10)
    auto_switch_min_delta = Optional(FloatField(), default=0.0)

    data = DataConfig()
    training = TrainingConfig()
    # validation = ValidationConfig()
    loss = LossConfig()
    schedule = ScheduleConfig()

    def __init__(self, config_path: str):
        super().load(yaml.safe_load(open(config_path)))
