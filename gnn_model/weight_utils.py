import yaml
import torch


def _build_instrument_map(observation_config: dict) -> dict[str, int]:
    """Build instrument name→id mapping consistent with process_timeseries and the model.

    IDs are assigned by sorted instrument names within each group, in group order:
    satellite then conventional.
    """
    order: list[str] = []
    for group in ("satellite", "conventional"):
        if group in observation_config:
            order += sorted(observation_config[group].keys())
    return {name: i for i, name in enumerate(order)}


def load_weights_from_yaml(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    observation_config = config.get("observation_config", {})
    feature_stats = config.get("feature_stats", {})

    name_to_id = _build_instrument_map(observation_config)

    if "obs_counts" in config:
        raw = {name_to_id[k]: 1.0 / (v + 1e-6) for k, v in config["obs_counts"].items() if k in name_to_id}
        s = sum(raw.values()) or 1.0
        instrument_weights = {k: v / s for k, v in raw.items()}
    else:
        instrument_weights = {name_to_id[k]: float(v) for k, v in config["instrument_weights"].items() if k in name_to_id}

    channel_weights = {
        name_to_id[k]: torch.tensor(v, dtype=torch.float32)
        for k, v in config["channel_weights"].items()
        if k in name_to_id
    }

    return observation_config, feature_stats, instrument_weights, channel_weights, name_to_id
