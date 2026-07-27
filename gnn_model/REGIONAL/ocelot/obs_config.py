"""
Observation configuration module - Python equivalent of observation_config.yaml

This module provides the same configuration data as observation_config.yaml
but in Python format for programmatic access.
"""

# Values to replace with NaN when loading data
FILL_VALUES = [99999997952.00, 1000000.00, 1000000000, 9999]

# Instrument weights
INSTRUMENT_WEIGHTS = {
    'atms': 1.0,
    'amsua': 1.0,
    'avhrr': 1.0,
    'ssmis': 1.0,
    'seviri': 1.0,
    'iasi': 1.0,
    'cris-fsr': 1.0,
    'ascat': 1.0,
    'surface_obs': 1.0,
    'snow_cover': 1.0,
    'radiosonde': 1.0,
    'sst': 1.0,
    'gps': 1.0,
    'surface_obs_t': 1.0,
    'surface_obs_uv': 1.0,
}

# Channel weights
CHANNEL_WEIGHTS = {
    'atms': [1.0] * 22,
    'amsua': [1.0] * 15,
    'avhrr': [1.0] * 3,
    'ssmis': [1.0] * 24,
    'seviri': [1.0] * 16,
    'iasi': [1.0] * 17,
    'cris-fsr': [1.0] * 15,
    'surface_obs': [1.0] * 6,  # one per predicted feature (6)
    'snow_cover': [1.0],  # snowDepth
    'radiosonde': [1.0] * 5,
}

# Channel numbers for specific instruments
ATMS_CHANNELS = list(range(1, 23))  # Channels 1-22
AMSUA_CHANNELS = list(range(1, 16))  # Channels 1-15
AVHRR_CHANNELS = [3, 4, 5]  # Channels 3-5
SSMIS_CHANNELS = list(range(1, 25))  # Channels 1-24
CRIS_FSR_CHANNELS = [
    28,
    95,
    132,
    158,
    400,
    496,
    626,
    678,
    748,
    874,
    1018,
    1133,
    1596,
    1635,
    2182]
IASI_CHANNELS = [
    89,
    148,
    259,
    350,
    414,
    1027,
    1271,
    1579,
    1710,
    2346,
    2701,
    3027,
    3322,
    5992,
    6182,
    6489,
    7584]

COMMON_RANGE = [150, 350]

from ocelot.obs_config_utils import apply_derived_instrument_dims  # noqa: E402

# Observation configuration
OBSERVATION_CONFIG = {
    'satellite': {
        'atms': {
            'sat_ids': [224, 225],
            'scan_angle_channels': 1,  # Number of scan angle dimensions
            'features': [f'bt_channel_{ch}' for ch in ATMS_CHANNELS],
            'metadata': ['sensorZenithAngle', 'solarZenithAngle', 'solarAzimuthAngle'],
            'dropna_cols': ['sensorZenithAngle', 'solarZenithAngle', 'solarAzimuthAngle'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },

        'amsua': {
            'sat_ids': [3, 5, 206, 209, 223],
            'scan_angle_channels': 1,  # Number of scan angle dimensions
            'features': [f'bt_channel_{ch}' for ch in AMSUA_CHANNELS],
            'metadata': ['sensorZenithAngle', 'solarZenithAngle', 'solarAzimuthAngle'],
            'dropna_cols': ['sensorZenithAngle', 'solarZenithAngle', 'solarAzimuthAngle'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },

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
        'diag_avhrr': {
            'sat_ids': ['metop-c', 'n18', 'metop-b'],
            'features': [f'observation_channel_{ch}' for ch in AVHRR_CHANNELS],
            'dropna_cols': ['sat_zenith_angle', 'sol_zenith_angle'],
            'metadata': ['sat_zenith_angle', 'sol_zenith_angle'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                f'observation_channel_{ch}': COMMON_RANGE for ch in AVHRR_CHANNELS
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
        'ascat': {
            'sat_ids': [3],
            'scan_angle_channels': 3,
            'features': ['backscatter_ch_1', 'backscatter_ch_2', 'backscatter_ch_3'],
            'metadata': [
                'radarIncidenceAngle_ch_1', 'radarIncidenceAngle_ch_2', 'radarIncidenceAngle_ch_3',
                'antennaBeamAzimuthAngle_ch_1', 'antennaBeamAzimuthAngle_ch_2', 'antennaBeamAzimuthAngle_ch_3'
            ],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_strict_flags': True,
            'qc_filters': {
                'backscatter_ch_1': {
                    'range': [-55.0, 3.0],
                },
                'backscatter_ch_2': {
                    'range': [-55.0, 5.0],
                },
                'backscatter_ch_3': {
                    'range': [-55.0, 3.2],
                }
            },
            'subsample': {
                'mode': 'random',
                'fraction': 0.05,
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
        'diag_ssmis': {
            'sat_ids': ['f17'],
            'scan_angle_channels': 2,
            'features': [f'observation_channel_{ch}' for ch in SSMIS_CHANNELS],
            'metadata': ['scan_position', 'sat_zenith_angle'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                f'observation_channel_{ch}': COMMON_RANGE for ch in SSMIS_CHANNELS
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
    },
    'conventional': {
        'surface_obs': {
            'source': 'zarr',
            'zarr_name': 'raw_surface_obs',
            'features': ['pressureMeanSeaLevel_bufr', 'airTemperature', 'dewPointTemperature', 'wind_u', 'wind_v'],
            'metadata': ['height_bufr'],
            'dropna_cols': ['height_bufr'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'qc_filters': {
                'pressureMeanSeaLevel_bufr': {
                    'range': [5.7, 7.5],
                },
                'airTemperature': {
                    'range': [-80, 60],
                    'qm_flag_col': 'airTemperatureQuality_event_0',
                    'keep': [1, 2, 4, 9]
                },
                'dewPointTemperature': {
                    'range': [-100, 40],
                    'qm_flag_col': 'dewPointTemperatureQuality_event_0',
                    'keep': [1, 2, 4, 9]
                },
                'height_bufr': {
                    'range': [-388, 9000]
                },
                'windSpeed': {
                    # global min/max across levels (0 always, max from summary)
                    'range': [0, 405],
                },
                'windDirection': {
                    # global min/max across levels (from summary)
                    'range': [0, 8.85],
                },

            },
            'qc_relations': {
                'dewpoint_le_temp': True,
                'max_temp_dewpoint_spread': 60,
                'rh_from_td_consistency_pct': 25,
                'pressure_vs_height': {
                    'enable': True,
                    'tolerance_hpa': 100,
                    'scale_height_m': 8000
                }
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1,
                'seed': 12345
            }
        },
        'diag_sst': {
            'source': 'zarr',
            'zarr_name': 'diag_sst',
            'features': ['observation'],
            'metadata': ['height'],
            'dropna_cols': [],
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
        'radiosonde': {
            'source': 'zarr',
            'zarr_name': 'raw_radiosonde',
            'features': ['geopotentialHeight_gp10', 'airTemperature', 'dewPointTemperature', 'wind_u', 'wind_v'],
            'metadata': ['airPressure'],
            'dropna_cols': ['airPressure'],
            'input_dim': 18,
            'target_dim': 5,
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'level_selection': {
                'filter_col': 'airPressure',
                'levels': [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
            },
            'qc_filters': {
                'airPressure': {
                    'range': [2, 7],
                    'qm_flag_col': 'airPressureQuality',
                    'keep': [1, 2]
                },
                'airTemperature': {
                    'range': [-273.15, 380.44],  # global min/max across levels
                    'range_by_level': {  # min, max per pressure level (from raw_radiosonde standard-levels summary)
                        1000: [-77.5, 72],
                        925: [-99.5, 52.8],
                        850: [-152.42, 37.4],
                        700: [-273.15, 322.28],
                        500: [-273.15, 311.3],
                        400: [-180.23, 188.42],
                        300: [-98.3, 229.37],
                        250: [-252.67, 106.49],
                        200: [-88.9, 270.33],
                        150: [-131.36, 317.7],
                        100: [-247.94, 55.2],
                        70: [-169.57, 169.56],
                        50: [-173.61, 156.47],
                        30: [-245.49, 328.02],
                        20: [-258.52, 356.32],
                        10: [-269.94, 380.44],
                    },
                    'qm_flag_col': 'airTemperatureQuality',
                    'keep': [1, 2, 3, ],
                    'keep_nan': 1
                },
                'height_bufr': {
                    'range': [-50, 6000],
                },
                'dewPointTemperature': {
                    'range': [-273.15, 380.44],  # global min/max across levels
                    'range_by_level': {  # min, max per pressure level (from raw_radiosonde standard-levels summary)
                        1000: [-77.5, 72],
                        925: [-99.5, 52.8],
                        850: [-152.42, 37.4],
                        700: [-273.15, 322.28],
                        500: [-273.15, 311.3],
                        400: [-180.23, 188.42],
                        300: [-98.3, 229.37],
                        250: [-252.67, 106.49],
                        200: [-88.9, 270.33],
                        150: [-131.36, 317.7],
                        100: [-247.94, 55.2],
                        70: [-169.57, 169.56],
                        50: [-173.61, 156.47],
                        30: [-245.49, 328.02],
                        20: [-258.52, 356.32],
                        10: [-269.94, 380.44],
                    },
                    'qm_flag_col': 'dewPointTemperatureQuality',
                    'keep': [2, 3, 9],
                    'keep_nan': 1
                },
                'windSpeed': {
                    # global min/max across levels (0 always, max from summary)
                    'range': [0, 75],
                    'qm_flag_col': 'windQuality',
                    'keep': [2]
                },
                'windDirection': {
                    # global min/max across levels (from summary)
                    'range': [0, 6.28],
                    'qm_flag_col': 'windQuality',
                    'keep': [2]
                },
                'geopotentialHeight_gp10': {
                    # global min/max across levels (from summary)
                    'range': [-9798, 1660060],
                    'range_by_level': {  # min, max per pressure level (from raw_radiosonde standard-levels summary)
                        1000: [-4782, 10721.2],
                        925: [-519, 16620.8],
                        850: [6223, 27097],
                        700: [-9798, 1.11284e+06],
                        500: [-9798, 679260],
                        400: [-9797.1, 93492],
                        300: [-9796.1, 121030],
                        250: [-9795.1, 142296],
                        200: [-9795.1, 165816],
                        150: [-9795.1, 514102],
                        100: [89954.2, 1.66006e+06],
                        70: [20215.4, 1.09798e+06],
                        50: [74473.1, 1.1105e+06],
                        30: [4215, 1.57114e+06],
                        20: [-3457.4, 1.54966e+06],
                        10: [-9793.1, 1.54104e+06],
                    }
                }
            },
            'subsample': {
                'mode': 'random',
                'fraction': 1.0,
                'seed': 12345
            }
        },
        'snow_cover': {
            'source': 'zarr',
            'zarr_name': 'snow_cover',
            'features': ['snowDepth'],
            'metadata': ['height', 'airTemperature'],
            'dropna_cols': ['height', 'airTemperature'],
            'qc_filters': {
                'snowDepth': {
                    'range': [0, 6.5]
                }
            },
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
            'subsample': {
                'mode': 'none',
                'fraction': 1.0,
                'seed': 12345
            }
        }
    }
}

apply_derived_instrument_dims(OBSERVATION_CONFIG)

# Feature statistics
FEATURE_STATS = {
    'surface_obs': {
        'latitude': [30.25, 28.51],
        'longitude': [-23.17, 84.75],
        'pressureMeanSeaLevel_bufr': [6.92317, 0.0163],
        'airTemperature': [14.47, 12.1],
        'dewPointTemperature': [8.55, 11.44],
        'relativeHumidity': [74.88, 19.12],
        'windSpeed': [3.67, 5.01],
        'windDirection': [167.85, 109.79],
        'height_bufr': [315.15, 508.54],
        'wind_u': [0.08, 4.17],
        'wind_v': [0.25, 3.37]
    },
    'radiosonde': {
        'airTemperature': [240.26, 28.63],
        'dewPointTemperature': [224.33, 34.44],
        'windDirection': [3.66, 1.66],
        'windSpeed': [14.91, 12.64],
        'latitude': [30.18, 28.81],
        'longitude': [15.98, 102.35],
        'height': [724.68, 11977.6],
        'wind_u': [10.00, 8.5],
        'wind_v': [0.25, 6.78],
        'height_bufr': [227, 415],
        'airPressure': [5.981, 0.746],
        # mean, std per pressure level (from raw_radiosonde standard-levels
        # summary); [mean, std]
        'mean_std_by_level': {
            'airTemperature': {
                1000: [6.11, 16.43],
                925: [3.43, 14.07],
                850: [0.88, 12.01],
                700: [-5.24, 11.44],
                500: [-17.36, 11.83],
                400: [-30.38, 11.1],
                300: [-42.75, 9.96],
                250: [-51.18, 7.92],
                200: [-56.03, 6.16],
                150: [-59.36, 6.25],
                100: [-63.75, 8.77],
                70: [-65.28, 8.56],
                50: [-62.73, 7.75],
                30: [-59.56, 8.55],
                20: [-56.52, 10.5],
                10: [-52.7, 12.13],
            },
            'dewPointTemperature': {
                1000: [1.05, 16.52],
                925: [-2.79, 14.58],
                850: [-8.74, 14.18],
                700: [-18.99, 14.98],
                500: [-31.91, 14.56],
                400: [-43.95, 12.87],
                300: [-55.51, 11.14],
                250: [-64.73, 9.21],
                200: [-72.54, 7.58],
                150: [-79.74, 6.08],
                100: [-85.04, 5.59],
                70: [-87.25, 5.93],
                50: [-87.76, 6.99],
                30: [-86.97, 7.6],
                20: [-85.53, 8.84],
                10: [-83.93, 10.69],
            },
            'windDirection': {
                1000: [3.07, 1.98],
                925: [3.36, 1.87],
                850: [3.64, 1.79],
                700: [3.93, 1.62],
                500: [4.04, 1.51],
                400: [4.08, 1.48],
                300: [4.11, 1.46],
                250: [4.18, 1.39],
                200: [4.28, 1.27],
                150: [4.38, 1.13],
                100: [4.39, 1.12],
                70: [4.35, 1.23],
                50: [4.24, 1.38],
                30: [4.01, 1.56],
                20: [3.78, 1.65],
                10: [3.88, 1.61],
            },
            'windSpeed': {
                1000: [5.49, 5.02],
                925: [8.65, 5.71],
                850: [9.38, 6.11],
                700: [12.08, 7.72],
                500: [16.94, 11.38],
                400: [22.25, 14.58],
                300: [26.95, 17.97],
                250: [29.77, 19.98],
                200: [30.16, 20.3],
                150: [27.59, 17.41],
                100: [23.18, 13.35],
                70: [17.54, 10.33],
                50: [15.34, 10.15],
                30: [15.74, 11.88],
                20: [19.17, 14.59],
                10: [28.05, 20.44],
            },
            'geopotentialHeight_gp10': {
                1000: [1847.35, 1316.27],
                925: [7598.48, 1780.43],
                850: [15788.3, 3089.86],
                700: [31142.6, 6125.68],
                500: [52206.2, 7800.98],
                400: [70964.4, 5567.3],
                300: [88049.7, 5558.61],
                250: [101925, 5251.54],
                200: [116103, 5936.1],
                150: [134180, 6947.12],
                100: [155944, 8026.67],
                70: [178443, 7352.32],
                50: [200840, 17248.7],
                30: [228083, 19896.5],
                20: [257392, 19359.6],
                10: [298357, 20747.7],
            },
        },
    },
    'snow_cover': {
        'snowDepth': [0.38, 0.68]
    },
    'atms': {
        'bt_channel_1': [211.00, 40.60],
        'bt_channel_2': [201.65, 42.08],
        'bt_channel_3': [239.09, 22.24],
        'bt_channel_4': [248.58, 17.55],
        'bt_channel_5': [252.58, 14.32],
        'bt_channel_6': [245.75, 11.47],
        'bt_channel_7': [231.98, 8.21],
        'bt_channel_8': [222.52, 6.76],
        'bt_channel_9': [216.10, 7.30],
        'bt_channel_10': [211.99, 9.49],
        'bt_channel_11': [215.01, 9.37],
        'bt_channel_12': [221.00, 10.17],
        'bt_channel_13': [229.26, 11.30],
        'bt_channel_14': [240.06, 11.97],
        'bt_channel_15': [250.28, 11.52],
        'bt_channel_16': [237.76, 28.61],
        'bt_channel_17': [258.29, 29.35],
        'bt_channel_18': [260.31, 22.15],
        'bt_channel_19': [258.26, 18.46],
        'bt_channel_20': [255.12, 15.34],
        'bt_channel_21': [250.24, 12.39],
        'bt_channel_22': [245.38, 10.41],
    },
    'amsua': {
        'bt_channel_1': [209.92, 40.44],
        'bt_channel_2': [201.16, 42.14],
        'bt_channel_3': [239.63, 20.68],
        'bt_channel_4': [252.44, 14.20],
        'bt_channel_5': [245.45, 11.31],
        'bt_channel_6': [231.10, 7.86],
        'bt_channel_7': [222.16, 6.51],
        'bt_channel_8': [216.24, 7.12],
        'bt_channel_9': [211.48, 9.39],
        'bt_channel_10': [214.51, 9.34],
        'bt_channel_11': [220.23, 10.18],
        'bt_channel_12': [228.57, 11.32],
        'bt_channel_13': [239.08, 11.94],
        'bt_channel_14': [249.39, 11.47],
        'bt_channel_15': [237.78, 28.23],
    },
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
    'diag_avhrr': {
        "observation_channel_3": (291.07, 10.59),
        "observation_channel_4": (289.19, 8.70),
        "observation_channel_5": (287.13, 8.18),
    },
    'ascat': {
        'backscatter_ch_1': [-17.84, 5.77],
        'backscatter_ch_2': [-14.84, 5.12],
        'backscatter_ch_3': [-17.81, 5.73]
    },
    'diag_ssmis': {
        **{f'observation_channel_{ch}': stats for ch, stats in zip(
            SSMIS_CHANNELS,
            [[241.56, 18.18], [253.06, 11.59], [243.67, 7.89], [228.35, 5.02], [214.87, 6.93],
             [212.03, 8.47], [216.7, 7.75], [259.45, 28.54], [
                 262.34, 17.43], [256.26, 12.29],
             [243.89, 8.39], [179.17, 56.61], [222.94, 34.67], [
                 238.08, 29.15], [191.91, 44.78],
             [232.2, 25.78], [254.83, 22.86], [234.82, 30.33], [
                 228.97, 7.64], [208.37, 10.73],
             [248.93, 7.13], [250.07, 9.89], [239.85, 10.08], [226.23, 8.88]]
        )}
    },
    'seviri': {
        'bt_all_ch_10': [99.96, 277.3],
        'bt_all_ch_11': [99.96, 258.43],
        'bt_all_ch_4': [99.95, 282.76],
        'bt_all_ch_5': [99.96, 237.52],
        'bt_all_ch_6': [99.96, 254.45],
        'bt_all_ch_7': [99.96, 276.94],
        'bt_all_ch_8': [99.96, 257.69],
        'bt_all_ch_9': [99.96, 278.99],
        'bt_clear_ch_10': [73.44, 288.78],
        'bt_clear_ch_11': [73.44, 265.8],
        'bt_clear_ch_4': [73.44, 290.5],
        'bt_clear_ch_5': [73.44, 240.42],
        'bt_clear_ch_6': [73.44, 259.62],
        'bt_clear_ch_7': [73.44, 287.69],
        'bt_clear_ch_8': [73.44, 264.72],
        'bt_clear_ch_9': [73.44, 290.49]
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
