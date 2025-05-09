import yaml
import torch


INSTRUMENT_NAME_TO_ID = {
    "atms": 0,
    "surface_pressure": 1,
    "radiosonde": 2,
    # Add more as needed
}


def load_weights_from_yaml(path):

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # Map instrument names to IDs
    if "obs_counts" in config:
        raw_weights = {
            INSTRUMENT_NAME_TO_ID[k]: 1 / (v + 1e-6)
            for k, v in config["obs_counts"].items()
        }
        total = sum(raw_weights.values())
        instrument_weights = {k: w / total for k, w in raw_weights.items()}
    else:
        instrument_weights = {
            INSTRUMENT_NAME_TO_ID[k]: float(v)
            for k, v in config["instrument_weights"].items()
        }

    channel_weights = {
        INSTRUMENT_NAME_TO_ID[k]: torch.tensor(v, dtype=torch.float32)
        for k, v in config["channel_weights"].items()
    }

    return instrument_weights, channel_weights
