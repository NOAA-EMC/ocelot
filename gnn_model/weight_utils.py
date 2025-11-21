import yaml
import torch

INSTRUMENT_NAME_TO_ID = {"atms": 0, "surface_obs": 1, "amsua": 2, "snow_cover": 3, "avhrr": 4, "radiosonde": 5, "ascat": 6}


def load_weights_from_yaml(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    observation_config = config.get("observation_config", {})
    feature_stats = config.get("feature_stats", {})

    if "obs_counts" in config:
        raw = {INSTRUMENT_NAME_TO_ID[k]: 1.0 / (v + 1e-6) for k, v in config["obs_counts"].items()}
        s = sum(raw.values()) or 1.0
        instrument_weights = {k: v / s for k, v in raw.items()}
    else:
        instrument_weights = {INSTRUMENT_NAME_TO_ID[k]: float(v) for k, v in config["instrument_weights"].items()}

    channel_weights = {INSTRUMENT_NAME_TO_ID[k]: torch.tensor(v, dtype=torch.float32) for k, v in config["channel_weights"].items()}

    return observation_config, feature_stats, instrument_weights, channel_weights, INSTRUMENT_NAME_TO_ID
