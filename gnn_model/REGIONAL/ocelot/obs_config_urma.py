"""
Observation configuration module - Python equivalent of observation_config.yaml

This module provides the same configuration data as observation_config.yaml
but in Python format for programmatic access.
"""

# Values to replace with NaN when loading data
FILL_VALUES = [99999997952.00, 1000000.00, 1000000000, 9999]

# Instrument weights
INSTRUMENT_WEIGHTS = {
    'diag_t': 1.0,
    'diag_q': 1.0,
    'diag_ps': 1.0,
    'diag_uv': 1.0,
    'ges': 1.0,
}

# Channel weights
CHANNEL_WEIGHTS = {
    'diag_t': [1.0] * 22,
    'diag_q': [1.0] * 15,
    'diag_ps': [1.0] * 3,
    'diag_uv': [1.0] * 24,
}

COMMON_RANGE = [150, 350]

from ocelot.obs_config_utils import apply_derived_instrument_dims  # noqa: E402

# Observation configuration
OBSERVATION_CONFIG = {
    "state": {
        "ges": {
            'source': 'zarr',
            'features': ['tmp_2maboveground', 'pres_surface', 'vgrd_10maboveground', 'ugrd_10maboveground', 'dpt_2maboveground',],
            # 'features': ['tmp_2maboveground', 'hgt_surface', 'pres_surface', 'vgrd_10maboveground',  'ugrd_10maboveground',
            #              'dpt_2maboveground',] ,
            "target_dim": 5,              # number of state variables
            "input_dim": 20,              # 2 * N + FEATURES_AUX_DIM + num_metadata + STATIC_CONTEXT_DIM,
            "anal_zarr_name": "anal_urma",   # parquet file base name for anal
            "zarr_name": "ges_urma",         # parquet file base name for ges
        }
    },
    'satellite': {
    },
    'conventional': {
        'diag_t': {
            'source': 'zarr',
            'zarr_name': 'diag_urma_t',
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
                    'range': [-357, 6000],

                },
                'observation': {
                    'range': [228.15, 312.15],

                },
                'pressure': {
                    'range': [5.56, 7.09],

                },

            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
        'diag_uv': {
            'source': 'zarr',
            'zarr_name': 'diag_urma_uv',
            'features': ['observation', 'pressure'],
            'metadata': ['height'],
            'dropna_cols': ['height'],
            'drop_rows': {
                'analysis_use_flag': [-1]
            },
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                'height': {
                    'range': [-64, 5776],

                },
                'observation': {
                    'range': [-25, 44],

                },
                'pressure': {
                    'range': [5.56, 7.09],

                }
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
        'diag_ps': {
            'source': 'zarr',
            'zarr_name': 'diag_urma_ps',
            'features': ['observation', 'pressure'],
            'metadata': ['height'],
            'dropna_cols': ['height'],
            'drop_rows': {
                'analysis_use_flag': [-1]
            },
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                'height': {
                    'range': [-67, 5612],

                },
                'observation': {
                    'range': [500, 1052],

                },
                'pressure': {
                    'range': [5.56, 7.09],

                }
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
        'diag_q': {
            'source': 'zarr',
            'zarr_name': 'diag_urma_q',
            'features': ['observation', 'pressure'],
            'metadata': ['height'],
            'dropna_cols': ['height'],
            'drop_rows': {
                'analysis_use_flag': [-1]
            },
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                'height': {
                    'range': [-357, 6000],

                },
                'observation': {
                    'range': [0, 30],

                },
                'pressure': {
                    'range': [5.56, 6.97],

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
    'diag_t': {
        'observation': [276.58, 10.81],
        'height': [573.49, 696.63],
        'pressure': [6.89, 0.08],
    },
    'diag_q': {
        'observation': [4.18, 3.04],
        'height': [543.96, 648],
        'pressure': [6.89, 0.08],
    },

    'diag_ps': {
        'observation': [978, 55],
        'height': [349.41, 488.67],
        'pressure': [6.89, 0.08],
    },

    'diag_uv': {
        'observation': [0.57, 2.89],
        'height': [640, 695],
        'pressure': [6.89, 0.08],
    },

    'ges': {
        'hgt_surface': [442, 622],
        'pres_surface': [96000, 6950],
        'vgrd_10maboveground': [0.10, 5.0],
        'ugrd_10maboveground': [1.2, 4.5],
        'dpt_2maboveground': [273.15, 13.5],
        'tmp_2maboveground': [280.14, 13.0],
    }
}

STATE_INCREMENT_STATS = {
    "tmp_2maboveground": [
        0.00416477081149185,
        0.06005789850646087
    ],
    "pres_surface": [
        0.000574698606485636,
        0.006262742829476267
    ],
    "vgrd_10maboveground": [
        -0.0011717678905649221,
        0.15212414347233272
    ],
    "ugrd_10maboveground": [
        -0.010390013714132994,
        0.1720638438077753
    ],
    "hgt_surface": [
        0.0,
        1
    ],
    "dpt_2maboveground": [
        -0.0008408632737459072,
        0.06351003475323037
    ]
}
