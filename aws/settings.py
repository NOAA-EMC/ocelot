
setting_items = ["OS",
                 "REGION",
                 "KEY_NAME",
                 "SUBNET_ID",
                 "ADMIN_ROLE",
                 "HEAD_NODE_INSTANCE",
                 "HEAD_NODE_ROLE",
                 "COMPUTE_NODE_ROLE",
                 "IMPORT_PATH",
                 "EXPORT_PATH",
                 "KEY_FILE",
                 "ON_NODE_START_SCRIPT"]

# Added configuration from local settings
try:
    from local_settings import *
except ImportError:
    settings_str = "\n".join(setting_items)
    raise RuntimeError(f"local_settings.py not found. Please create local_settings.py with the "
                       f"following items:\n {settings_str}.")

# Validate the settings file
missing_settings = [item for item in setting_items if item not in locals()]
if missing_settings:
    missing_settings_str = "\n".join(missing_settings)
    raise RuntimeError(f"Missing settings in local_settings.py: {missing_settings_str}")
