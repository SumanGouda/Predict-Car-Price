from pathlib import Path
from utils.cleaning import clean_drive_type, clean_engine_type
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data warehouse settings
### Mileage
db_file = PROJECT_ROOT / "database" / "car_database.db"
ohe_metadata_file_mileage = (
    PROJECT_ROOT / "predictions" / "mileage" / "model" / "ohe_metadata.json"
)
features_file_mileage = (
    PROJECT_ROOT / "predictions" / "mileage" / "data" / "features.txt"
)
output_file_mileage = (
    PROJECT_ROOT / "predictions" / "mileage" / "data" / "raw" / "data.csv"
)
processed_file_mileage = (
    PROJECT_ROOT / "predictions" / "mileage" / "data" / "processed" / "data.csv"
)

### Power
ohe_metadata_file_power = (
    PROJECT_ROOT / "predictions" / "power" / "model" / "ohe_metadata.json"
)
features_file_power = (
    PROJECT_ROOT / "predictions" / "power" / "data" / "features.txt"
)
output_file_power = (
    PROJECT_ROOT / "predictions" / "power" / "data" / "raw" / "data.csv"
)
processed_file_power = (
    PROJECT_ROOT / "predictions" / "power" / "data" / "processed" / "data.csv"
)

process_raw_data = True

FEATURE_REGEX_PATTERNS = {
    "Mileage": (r"(\d+\.?\d*)", float),
    "Engine": (r"(\d+)", float),
    "Kerb Weight": (r"(\d+)", float),
    "Power": (r"(\d+\.?\d*)", float),
    "Registration Year": (r"(\d{4})", float),
    "Transmission Type": {"Automatic": 1, "Manual": 0},
}
FUNC_CLEAN_DICT = {
    "Drive Type": clean_drive_type,
    "Engine Type": clean_engine_type,
}
OHE_FEATURES = ["Fuel", "Engine Type", "Drive Type"]

# Mileage Model Training Settings
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
