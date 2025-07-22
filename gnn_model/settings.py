
setting_items = ["LOG_DIR",
                 "CHECKPOINT_DIR",
                 "DATA_DIR_CONUS",
                 "DATA_DIR_GLOBAL"]

# Added configuration from local settings
try:
    from local_settings import *
except ImportError:
    LOG_DIR = 'logs'
    CHECKPOINT_DIR = 'checkpoints'
    DATA_DIR_CONUS = '/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/'
    DATA_DIR_GLOBAL = '/scratch3/NCEPDEV/da/Azadeh.Gholoubi/data_v3/bigzarr'

# Validate the settings file
missing_settings = []
for item in setting_items:
    if item not in locals():
        missing_settings.append(item)

if missing_settings:
    missing_settings_str = "\n".join(missing_settings)
    raise RuntimeError(f"Missing settings in local_settings.py:\n{missing_settings_str}")
