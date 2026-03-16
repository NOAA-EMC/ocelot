import yaml
import torch
import warnings


def _handle_unknown_instrument_keys(
    *,
    path: str,
    section: str,
    provided_keys: set[str],
    known_keys: set[str],
    on_unknown: str,
) -> None:
    unknown = sorted(provided_keys - known_keys)
    if not unknown:
        return

    msg = (
        f"[WEIGHTS] {path}: '{section}' contains instrument keys that are not present in "
        f"observation_config (will be dropped): {unknown}"
    )

    on_unknown = (on_unknown or "warn").lower()
    if on_unknown == "ignore":
        return
    if on_unknown == "warn":
        warnings.warn(msg)
        return
    if on_unknown == "error":
        raise KeyError(msg)
    raise ValueError(f"Invalid on_unknown={on_unknown!r}; expected 'ignore'|'warn'|'error'.")


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


def load_weights_from_yaml(path, *, on_unknown: str = "warn"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    observation_config = config.get("observation_config", {})
    feature_stats = config.get("feature_stats", {})

    name_to_id = _build_instrument_map(observation_config)
    known_instruments = set(name_to_id.keys())

    if "obs_counts" in config:
        _handle_unknown_instrument_keys(
            path=path,
            section="obs_counts",
            provided_keys=set(map(str, config["obs_counts"].keys())),
            known_keys=known_instruments,
            on_unknown=on_unknown,
        )
        raw = {name_to_id[k]: 1.0 / (v + 1e-6) for k, v in config["obs_counts"].items() if k in name_to_id}
        s = sum(raw.values()) or 1.0
        instrument_weights = {k: v / s for k, v in raw.items()}
    else:
        _handle_unknown_instrument_keys(
            path=path,
            section="instrument_weights",
            provided_keys=set(map(str, config["instrument_weights"].keys())),
            known_keys=known_instruments,
            on_unknown=on_unknown,
        )
        instrument_weights = {name_to_id[k]: float(v) for k, v in config["instrument_weights"].items() if k in name_to_id}

    _handle_unknown_instrument_keys(
        path=path,
        section="channel_weights",
        provided_keys=set(map(str, config["channel_weights"].keys())),
        known_keys=known_instruments,
        on_unknown=on_unknown,
    )

    channel_weights = {
        name_to_id[k]: torch.tensor(v, dtype=torch.float32)
        for k, v in config["channel_weights"].items()
        if k in name_to_id
    }

    return observation_config, feature_stats, instrument_weights, channel_weights, name_to_id
