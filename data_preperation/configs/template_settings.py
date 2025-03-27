from pickle import LONG1

# Path to the configuration yaml file for the BUFR tank
# (example: '/install_dir/spoc/tank/conf/hera_test_tank.yaml')
BUFR_TANK_YAML = ''

# Path to the root directory for the BUFR tank
# (example: '/data_dir/ops/prod/dcom')
TANK_PATH = '' 

# The datetime format string for the subdirecotries in the tank 
# (see docs for python datetime object)
DATETIME_DIR_FORMAT = '%Y%m%d'

# Path to the directory that holds the BUFR mapping files
# (example: '/install_dir/src/spoc/tank/mapping')
MAPPING_FILE_DIR = ''

OUTPUT_DIR = ''

# The range of latitude and longitude for the continental US
LAT_RANGE = (24, 51)
LON_RANGE = (-115, -74)

