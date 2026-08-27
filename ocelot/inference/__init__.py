from ocelot.configs.model_config import ModelConfig
from ocelot.configs.inference_config import InferenceConfig
from ocelot.configs.observation_config import ObservationConfig
from ocelot.inference.ocelot_inference import OcelotInferenceModule
from ocelot.model.ocelot import Ocelot


def make_module(model_config: ModelConfig,
                inference_config: InferenceConfig,
                observation_config: ObservationConfig,
                verbose: bool = False):
                
    model = Ocelot(
        model_config=model_config,
        observation_config=observation_config,
        verbose=verbose)

    try:
        model = Ocelot.load_from_checkpoint(
            args.checkpoint,
            observation_config=observation_config,
            strict=False,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("\nTrying alternative loading method...")

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        hparams = ckpt.get('hyper_parameters', {})

        # Filter hparams to match the current constructor signature (robust to checkpoint drift)
        try:
            sig = inspect.signature(Ocelot.__init__)
            allowed = set(sig.parameters.keys())
            allowed.discard('self')
            filtered_hparams = {k: v for k, v in (hparams or {}).items() if k in allowed}
        except Exception:
            filtered_hparams = dict(hparams or {})

        model = Ocelot(
            observation_config=observation_config,
            **filtered_hparams
        )

        state = ckpt.get('state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            "Model loaded successfully using alternative method (strict=False)! "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )
    
    return OcelotInferenceModule(model=model, inference_config=inference_config)
