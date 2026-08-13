
from model.ocelot_training import OcelotTrainingModule
from model.ocelot_inference import OcelotInferenceModule
from model.ocelot import Ocelot

from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from configs.inference_config import InferenceConfig
from configs.observation_config import ObservationConfig


class OcelotFactory:
    @staticmethod
    def create_training_module(model_config: ModelConfig,
                               training_config: TrainingConfig,
                               observation_config: ObservationConfig,
                               verbose: bool = False):
        model = Ocelot(
            model_config=model_config,
            observation_config=observation_config,
            verbose=verbose)

        return OcelotTrainingModule(model=model, training_config=training_config)


    @staticmethod
    def create_inference_module(model_config: ModelConfig,
                                inference_config: InferenceConfig,
                                observation_config: ObservationConfig,
                                verbose: bool = False):
        model = Ocelot(
            model_config=model_config,
            observation_config=observation_config,
            verbose=verbose)
        
        return OcelotInferenceModule(model=model, inference_config=inference_config)
