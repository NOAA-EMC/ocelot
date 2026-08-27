from ocelot.configs.model_config import ModelConfig
from ocelot.configs.training_config import TrainingConfig
from ocelot.configs.observation_config import ObservationConfig
from ocelot.training.ocelot_training import OcelotTrainingModule
from ocelot.model.ocelot import Ocelot


def make_module(model_config: ModelConfig,
                            training_config: TrainingConfig,
                            observation_config: ObservationConfig,
                            verbose: bool = False):
    model = Ocelot(
        model_config=model_config,
        observation_config=observation_config,
        verbose=verbose)

    return OcelotTrainingModule(model=model, training_config=training_config)

