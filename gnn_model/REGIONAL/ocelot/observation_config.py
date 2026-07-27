import importlib

from ocelot.logger import logger
from ocelot.obs_config_utils import apply_derived_instrument_dims


def load_observation_config(
    use_python_config: bool = True,
    exp_type: str = "global",
    config_name: str = "default",
) -> tuple:
    """
    Loads observation configuration and feature statistics.

    Args:
        use_python_config (bool): If True, uses the Python config module.
                                  If False, falls back to YAML loading for backward compatibility.
        exp_type (str): Experiment type controlling the instrument subset.
                        "regional" selects the regional subset; any other value (e.g. "global")
                        selects the global subset.
        config_name (str): Which obs_config variant to load. "default" loads
                           ``ocelot.obs_config``; any other value loads
                           ``ocelot.obs_config_<config_name>`` (e.g. "urma" → obs_config_urma).

    Returns:
        tuple: A tuple containing the observation_config dictionary, the
               feature_stats dictionary, the fill_values list, the
               instrument_weights dictionary (per-instrument loss weight,
               keyed the same as observation_config's instrument names), and
               the increment_stats dictionary (per-variable DA increment
               mean/std, in normalized-field space; see
               ocelot/compute_increment_stats.py).
    """
    if use_python_config:
        module_name = "obs_config" if config_name == "default" else f"obs_config_{config_name}"
        logger.info(f'module_name: {module_name}')
        cfg = importlib.import_module(f"ocelot.{module_name}")
        observation_config = cfg.OBSERVATION_CONFIG
        feature_stats = cfg.FEATURE_STATS
        fill_values = cfg.FILL_VALUES
        instrument_weights = getattr(cfg, 'INSTRUMENT_WEIGHTS', {})
        increment_stats = getattr(cfg, 'STATE_INCREMENT_STATS', {})
    else:
        # Backward compatibility: load from YAML
        import yaml
        path = './ocelot/observation_config.yaml'
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        observation_config = config.get("observation_config", {})
        apply_derived_instrument_dims(observation_config)
        feature_stats = config.get("feature_stats", {})
        fill_values = config.get("fill_values", [-999, -9999, 9.96921e+36])
        instrument_weights = config.get("instrument_weights", {})
        increment_stats = config.get("increment_stats", {})

    if exp_type in ("regional", ):
        # selected_obs = ['atms', 'surface_obs']
        selected_obs = [
            'diag_t',
            'diag_q',
            'diag_uv',
            'diag_ps',
        ]
    elif exp_type in ("regional_da", ):
        selected_obs = [
            'diag_t',
            'diag_q',
            'diag_uv',
            'diag_ps',
            'ges',
        ]
    elif exp_type in ('regional_rrfs',):
        selected_obs = [
            'diag_atms',
            'diag_amsua',
            # 'diag_cris-fsr',
            # 'diag_sst',
            'diag_surface_obs_t',
            # 'diag_surface_obs_uv'
        ]

    else:
        selected_obs = [
            'diag_atms',
            'radiosonde',
            # 'diag_amsua',
            'diag_ssmis',
            'diag_cris-fsr',
            'diag_avhrr',
            'diag_sst',
            'diag_surface_obs_t',
            # 'surface_obs',
            # 'diag_surface_obs_uv'
        ]

    filtered_obs_config = {}
    for obs_type, instruments in observation_config.items():
        filtered_instruments = {
            instrument_name: instrument_config
            for instrument_name, instrument_config in instruments.items()
            if instrument_name in selected_obs
        }
        if filtered_instruments:
            filtered_obs_config[obs_type] = filtered_instruments

    filtered_instrument_weights = {
        instrument_name: weight
        for instrument_name, weight in instrument_weights.items()
        if instrument_name in selected_obs
    }

    logger.debug(f"Filtered observation config: {filtered_obs_config}")
    return filtered_obs_config, feature_stats, fill_values, filtered_instrument_weights, increment_stats
