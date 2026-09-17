from pathlib import Path
from functools import partial
from utils.cleaning import clean_drive_type, clean_price, clean_car_name, clean_emission_norm

# ==========================================
# System & Path Configurations
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_FILE = PROJECT_ROOT / "database" / "car_database.db"
PROCESS_RAW_DATA = True

# ==========================================
# Data Processing Paths
# ==========================================
# Mileage Paths
FEATURES_FILE_MILEAGE = (
    PROJECT_ROOT / "predictions" / "mileage" / "data" / "features.txt"
)
OUTPUT_FILE_MILEAGE = (
    PROJECT_ROOT / "predictions" / "mileage" / "data" / "raw" / "data.csv"
)
PROCESSED_FILE_MILEAGE = (
    PROJECT_ROOT / "predictions" / "mileage" / "data" / "processed" / "data.csv"
)
OHE_METADATA_FILE_MILEAGE = (
    PROJECT_ROOT / "predictions" / "mileage" / "model" / "ohe_metadata.json"
)

# Power Paths
FEATURES_FILE_POWER = (
    PROJECT_ROOT / "predictions" / "power" / "data" / "features.txt"
)
OUTPUT_FILE_POWER = (
    PROJECT_ROOT / "predictions" / "power" / "data" / "raw" / "data.csv"
)
PROCESSED_FILE_POWER = (
    PROJECT_ROOT / "predictions" / "power" / "data" / "processed" / "data.csv"
)
OHE_METADATA_FILE_POWER = (
    PROJECT_ROOT / "predictions" / "power" / "model" / "ohe_metadata.json"
)

# Price Paths
FEATURES_FILE_PRICE = (
    PROJECT_ROOT / "predictions" / "price" / "data" / "features.txt"
)
OUTPUT_FILE_PRICE = (
    PROJECT_ROOT / "predictions" / "price" / "data" / "raw" / "data.csv"
)
PROCESSED_FILE_PRICE = (
    PROJECT_ROOT / "predictions" / "price" / "data" / "processed" / "data.csv"
)
OHE_METADATA_FILE_PRICE = (
    PROJECT_ROOT / "predictions" / "price" / "model" / "ohe_metadata.json"
)

# ==========================================
# Preprocessing Rules & Maps
# ==========================================
FEATURE_REGEX_PATTERNS = {
    "Mileage": (r"(\d+\.?\d*)", float),
    "Engine": (r"(\d+)", float),
    "Kerb Weight": (r"(\d+)", float),
    "Power": (r"(\d+\.?\d*)", float),
    "Registration Year": (r"(\d{4})", float),
    "Transmission Type": {"Automatic": 1, "Manual": 0},
}

EMISSION_NORM_ORDINAL_MAP = {
    "BS I": 1,
    "BS II": 2,
    "BS III": 3,
    "Euro IV": 4,
    "BS IV": 4,
    "Euro V": 5,
    "BS VI": 6,
    "Euro VI": 6,
    "BS VI 2.0": 7,
    "ZEV": 8,
    "Unknown": 0,
}

OHE_FEATURES = ["Fuel"]

FUNC_CLEAN_DICT = {
    "Drive Type": clean_drive_type,
    "Price": clean_price,
    "Emission Norm Compliance": partial(clean_emission_norm, map_dict=EMISSION_NORM_ORDINAL_MAP),
    "car_name": (clean_car_name, ["brand", "model"]),
}

# ==========================================
# Model Training Settings: Mileage
# ==========================================
MILEAGE_MODEL_TYPE = "xgboost"
MILEAGE_RANDOM_STATE = 42
MILEAGE_CV_FOLDS = 5

MILEAGE_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# ==========================================
# Model Training Settings: Power
# ==========================================
POWER_MODEL_TYPE = "xgboost"
POWER_RANDOM_STATE = 42
POWER_CV_FOLDS = 5

POWER_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# ==========================================
# Model Training Settings: Price
# ==========================================
PRICE_MODEL_TYPE = "random_forest"
PRICE_RANDOM_STATE = 42
PRICE_CV_FOLDS = 5

PRICE_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 12, 16],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": [0.6, 0.75, 0.85],
}
