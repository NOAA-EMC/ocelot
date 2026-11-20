import torch

from ocelot.configs.instrument_config import InstrumentCatalogConfig
from ocelot.configs.model_config import ModelConfig
from ocelot.configs.inference_config import InferenceConfig
from ocelot.configs.pipeline_config import PipelineConfig
from ocelot.inference.ocelot_inference import OcelotInferenceModule
from ocelot.model.ocelot import Ocelot


def make_module(model_config: ModelConfig,
                inference_config: InferenceConfig,
                instrument_catalog: InstrumentCatalogConfig,
                pipeline_config: PipelineConfig,
                verbose: bool = False):
    model = Ocelot(
        model_config=model_config,
        instrument_catalog=instrument_catalog,
        pipeline_config=pipeline_config,
        verbose=verbose)

    module = OcelotInferenceModule(model=model,
                                   inference_config=inference_config,
                                   pipeline_config=pipeline_config)

    return module
