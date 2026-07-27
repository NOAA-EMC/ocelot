"""
This file contains shared constants used across the Ocelot project.
"""

# Normalized feature values above this threshold are clipped to 0 and marked invalid.
FEATURES_NORM_OUTLIER_THRESHOLD = 3.0

# Time encoding: fractional day / mean Julian year (matches legacy day-of-year normalization).
SECONDS_PER_DAY = 86400.0
YEAR_PROGRESS_DAYS = 366

# features_aux columns: sin_lat, cos_lat, sin_lon, cos_lon, year_progress_sin/cos, day_progress_sin/cos
FEATURES_AUX_DIM = 8

# Static per-gridpoint context (terrain height, land/sea mask) appended to the
# state/anal decoder target node features (see DARegionalGraphDataset._process_state_nodes).
STATIC_CONTEXT_DIM = 2
