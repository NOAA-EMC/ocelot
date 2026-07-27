"""
Observation configuration module - Python equivalent of observation_config.yaml

This module provides the same configuration data as observation_config.yaml
but in Python format for programmatic access.
"""

# Values to replace with NaN when loading data
FILL_VALUES = [99999997952.00, 1000000.00, 1000000000, 9999]

# Instrument weights, keyed to match OBSERVATION_CONFIG instrument names below.
INSTRUMENT_WEIGHTS = {
    'diag_atms': 1.0,
    'diag_amsua': 1.0,
    'diag_iasi': 1.0,
    'diag_cris-fsr': 1.0,
    'diag_sst': 1.0,
    'diag_surface_obs_t': 1.0,
    'diag_surface_obs_uv': 1.0,
}

# Channel weights
CHANNEL_WEIGHTS = {
    'diag_t': [1.0] * 22,
    'diag_q': [1.0] * 15,
    'diag_ps': [1.0] * 3,
    'diag_uv': [1.0] * 24,
}

# Channel numbers for specific instruments
ATMS_CHANNELS = list(range(1, 23))  # Channels 1-22
AMSUA_CHANNELS = list(range(1, 16))  # Channels 1-15
AVHRR_CHANNELS = [3, 4, 5]  # Channels 3-5
SSMIS_CHANNELS = list(range(1, 25))  # Channels 1-24
CRIS_FSR_CHANNELS = [28, 95, 132, 158, 400, 496, 626, 678, 748, 874, 1018, 1133, 1596, 1635, 2182]
IASI_CHANNELS = [89, 148, 259, 350, 414, 1027, 1271, 1579, 1710, 2346, 2701, 3027, 3322, 5992, 6182, 6489, 7584]
COMMON_RANGE = [150, 350]

from ocelot.obs_config_utils import apply_derived_instrument_dims  # noqa: E402

# Observation configuration
OBSERVATION_CONFIG = {
    'satellite': {
        'diag_atms': {
            'sat_ids': ['npp', 'n20'],
            'features': [f'observation_channel_{ch}' for ch in ATMS_CHANNELS],
            'metadata': ['sat_zenith_angle', 'sol_zenith_angle', 'sol_azimuth_angle'],
            'dropna_cols': ['sat_zenith_angle', 'sol_zenith_angle', 'sol_azimuth_angle'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            
            'qc_filters': {
                f'observation_channel_{ch}': COMMON_RANGE for ch in ATMS_CHANNELS
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
        'diag_amsua': {
            'sat_ids': ['n15', 'n18', 'n19', 'metop-b', 'metop-c'],
            'features': [f'observation_channel_{ch}' for ch in AMSUA_CHANNELS],
            'metadata': ['sat_zenith_angle', 'sol_zenith_angle', 'sol_azimuth_angle'],
            'dropna_cols': ['sat_zenith_angle', 'sol_zenith_angle', 'sol_azimuth_angle'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                f'observation_channel_{ch}': COMMON_RANGE for ch in AMSUA_CHANNELS
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },

        'diag_iasi': {
            'sat_ids': ['metop-c', 'metop-b'],
            'scan_angle_channels': 2,
            'features': [f'observation_channel_{ch}' for ch in IASI_CHANNELS],
            'metadata': ['sat_zenith_angle', 'sol_zenith_angle', 'sol_azimuth_angle'],
            'dropna_cols': ['sat_zenith_angle', 'sol_zenith_angle', 'sol_azimuth_angle'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                f'observation_channel_{ch}': COMMON_RANGE for ch in IASI_CHANNELS
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,    
                'seed': 12345
            }
        },

        'diag_cris-fsr': {
            'sat_ids': ['n20', 'n21'],
            'scan_angle_channels': 2,
            'features': [f'observation_channel_{ch}' for ch in CRIS_FSR_CHANNELS],
            'metadata': ['sat_zenith_angle', 'sol_zenith_angle', 'sol_azimuth_angle'],
            'dropna_cols': ['sat_zenith_angle', 'sol_zenith_angle', 'sol_azimuth_angle'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                f'observation_channel_{ch}': COMMON_RANGE for ch in CRIS_FSR_CHANNELS
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
    },

    'conventional': {
                'diag_sst': {
            'source': 'zarr',
            'zarr_name': 'diag_sst',
            'features': ['observation'],
            'metadata': ['height'],
            'dropna_cols': [],
            'drop_rows': {
                'analysis_use_flag': [-1]
            },
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                'height': {
                    'range': [0, 20]
                },
                'observation': {
                    'range': [250.00, 327.00]
                }
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
        'diag_surface_obs_t': {
            'source': 'zarr',
            'zarr_name': 'diag_surface_obs_t',
            'features': ['observation', 'pressure'],
            'metadata': ['height'],
            'dropna_cols': ['height'],
            'drop_rows': {
                'analysis_use_flag': [-1]
            },
            'input_dim': 12,
            'target_dim': 2,
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                'height': {
                    'range': [-388, 9000]
                },
                'observation': {
                    'range': [220.00, 350.00]
                },
                'pressure': {
                    'range': [6.21, 7.09]
                }
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
        'diag_surface_obs_uv': {
            'source': 'zarr',
            'zarr_name': 'diag_surface_obs_uv',
            'features': ['u_observation', 'v_observation', 'pressure'],
            'metadata': ['height'],
            'dropna_cols': ['height'], 
            'drop_rows': {
                'analysis_use_flag': [-1]
            },

            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                'height': {
                    'range': [-388, 9000]
                },
                'u_observation': {
                    'range': [-50, 50]
                },
                'v_observation': {
                    'range': [-50, 50]
                },
                'pressure': {
                    'range': [6.68, 6.97]
                }
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
    }
}

apply_derived_instrument_dims(OBSERVATION_CONFIG)

# Feature statistics
FEATURE_STATS = {
       'diag_atms': {
        **{f'observation_channel_{ch}': stats for ch, stats in zip(
            ATMS_CHANNELS,
            [
                (212.26, 42.54),  # channel 1
                (201.69, 44.72),  # channel 2
                (241.46, 21.08),  # channel 3
                (251.77, 15.83),  # channel 4
                (256.53, 12.11),  # channel 5
                (248.98, 9.73),   # channel 6
                (234.25, 6.74),   # channel 7
                (223.89, 5.23),   # channel 8
                (216.70, 5.97),   # channel 9
                (211.66, 8.44),   # channel 10
                (215.00, 7.61),   # channel 11
                (221.60, 7.87),   # channel 12
                (229.98, 8.99),   # channel 13
                (240.70, 9.90),   # channel 14
                (250.66, 9.88),   # channel 15
                (242.96, 27.24),  # channel 16
                (264.69, 24.82),  # channel 17
                (264.98, 18.06),  # channel 18
                (262.01, 14.90),  # channel 19
                (258.27, 12.19),  # channel 20
                (252.66, 10.27),  # channel 21
                (247.10, 8.89),   # channel 22
            ]
        )}
    },
    'diag_amsua': {
        **{f'observation_channel_{ch}': stats for ch, stats in zip(
            AMSUA_CHANNELS,
            [
                (211.21, 42.76),  # channel 1
                (201.29, 44.88),  # channel 2
                (242.21, 20.75),  # channel 3
                (256.70, 12.54),  # channel 4
                (249.54, 9.81),   # channel 5
                (233.98, 6.61),   # channel 6
                (224.40, 5.08),   # channel 7
                (217.77, 5.95),   # channel 8
                (211.77, 8.64),   # channel 9
                (214.95, 7.93),   # channel 10
                (221.15, 8.20),   # channel 11
                (229.52, 9.35),   # channel 12
                (239.89, 10.24),  # channel 13
                (250.10, 10.24),  # channel 14
                (242.18, 27.20),  # channel 15
            ]
        )}
    },
    'diag_cris-fsr': {
        **{f'observation_channel_{ch}': stats for ch, stats in zip(
            CRIS_FSR_CHANNELS,
            [
                (225.18, 6.55),   # channel 28
                (230.25, 5.52),   # channel 95
                (244.95, 8.81),   # channel 132
                (258.28, 12.03),  # channel 158
                (277.74, 19.51),  # channel 400
                (277.95, 19.11),  # channel 496
                (244.97, 12.71),  # channel 626
                (275.72, 18.21),  # channel 678
                (277.62, 19.19),  # channel 748
                (263.07, 12.82),  # channel 874
                (243.97, 8.07),   # channel 1018
                (224.63, 4.68),   # channel 1133
                (273.96, 17.50),  # channel 1596
                (263.64, 14.14),  # channel 1635
                (282.14, 19.49),  # channel 2182
            ]
        )}
    },
    'diag_iasi': {
        **{f'observation_channel_{ch}': stats for ch, stats in zip(
            IASI_CHANNELS,
            [
                (222.98, 7.43),   # channel 89
                (215.24, 7.72),   # channel 148
                (235.90, 6.68),   # channel 259
                (254.57, 11.42),  # channel 350
                (249.70, 9.55),   # channel 414
                (277.62, 19.33),  # channel 1027
                (278.48, 19.36),  # channel 1271
                (244.61, 12.13),  # channel 1579
                (276.34, 18.39),  # channel 1710
                (277.79, 19.18),  # channel 2346
                (244.82, 8.23),   # channel 2701
                (257.42, 10.94),  # channel 3027
                (242.87, 8.32),   # channel 3322
                (279.62, 19.40),  # channel 5992
                (258.80, 13.00),  # channel 6182
                (224.68, 6.98),   # channel 6489
                (282.13, 19.12),  # channel 7584
            ]
        )}
    },
        'diag_sst': {
        'observation': [292.12, 8.12],
        'station_elevation': [3.15, 22.34],
        'height': [1.03, 0.84],
    },
    'diag_surface_obs_t': {
        'observation': [288.87, 12.61],
        'height': [281.12, 487.22],
        'pressure': [6.89, 0.06],
    },
    'diag_surface_obs_uv': {
        'u_observation': [0.12, 3.48],
        'v_observation': [0.15, 3.48],
        'height': [291.12, 485.22],
        'pressure': [6.89, 0.05],
    }
}
