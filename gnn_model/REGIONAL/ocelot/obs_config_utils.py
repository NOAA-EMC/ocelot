from ocelot.constants import FEATURES_AUX_DIM


def apply_derived_instrument_dims(observation_config: dict) -> None:
    """
    Set target_dim and input_dim on each instrument from features, metadata list length,
    and FEATURES_AUX_DIM. Must match graph_dataset concatenation rules:

    - satellite:     input.x = [features_norm, features_meta, features_aux]
                     -> input_dim = len(features) + len(metadata) + FEATURES_AUX_DIM
    - conventional:  input.x = [features_norm, features_meta?, features_aux, features_valid_mask]
                     -> input_dim = len(features) + len(metadata) + FEATURES_AUX_DIM + len(features)
    """
    for obs_type, instruments in observation_config.items():
        if obs_type not in ("satellite", "conventional"):
            continue
        if not isinstance(instruments, dict):
            continue
        for _name, inst in instruments.items():
            if not isinstance(inst, dict):
                continue
            feats = inst.get("features")
            if feats is None:
                continue
            target_dim = len(feats)
            n_meta = len(inst.get("metadata", []))
            inst["target_dim"] = target_dim
            if obs_type == "satellite":
                inst["input_dim"] = target_dim + n_meta + FEATURES_AUX_DIM
            else:
                inst["input_dim"] = target_dim + n_meta + FEATURES_AUX_DIM + target_dim
