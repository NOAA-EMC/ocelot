from ocelot.configs.instrument_config import InstrumentCatalogConfig
from ocelot.configs.model_config import ModelConfig
from ocelot.configs.pipeline_config import PipelineConfig
from ocelot.configs.training_config import TrainingConfig
from ocelot.training.ocelot_training import OcelotTrainingModule
from ocelot.model.ocelot import Ocelot


def make_module(model_config: ModelConfig,
                            training_config: TrainingConfig,
                            instrument_catalog: InstrumentCatalogConfig,
                            pipeline_config: PipelineConfig,
                            verbose: bool = False):
    model = Ocelot(
        model_config=model_config,
        instrument_catalog=instrument_catalog,
        pipeline_config=pipeline_config,
        verbose=verbose)

    return OcelotTrainingModule(model=model, training_config=training_config)

