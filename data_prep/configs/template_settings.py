import os

config_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]


# Path to the configuration yaml file for the BUFR tank
# (example: '/install_dir/spoc/tank/conf/hera_test_tank.yaml')
BUFR_TANK_YAML = os.path.join(config_dir, 'hera.yaml')

# Path to the root directory for the BUFR tank
# (example: '/data_dir/ops/prod/dcom')
TANK_PATH = ''

# The datetime format string for the subdirecotries in the tank
# (see docs for python datetime object)
DATETIME_DIR_FORMAT = 'gdas.%Y%m%d'

# Path to the directory that holds the BUFR mapping files
# (example: '/install_dir/src/spoc/tank/mapping')
MAPPING_FILE_DIR = os.path.realpath(os.path.join(config_dir, '..', 'mapping'))

# Path to the directory where the output files will be saved
OUTPUT_PATH = ''

# # The range of latitude and longitude for the continental US
# LAT_RANGE = (24, 51)
# LON_RANGE = (-115, -74)
