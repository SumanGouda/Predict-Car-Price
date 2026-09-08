from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Data warehouse settings
db_file                 = f"{PROJECT_ROOT}/database/car_database.db"
output_file_mileage     = f"{PROJECT_ROOT}/predictions/mileage/data/raw/data.csv"
ohe_metadata_file       = f"{PROJECT_ROOT}/predictions/mileage/model/ohe_metadata.json"
mileage_features_file   = f"{PROJECT_ROOT}/predictions/mileage/data/features.txt"
processed_file_mileage = f"{PROJECT_ROOT}/predictions/mileage/data/processed/data.csv"

process_raw_data        = True

FEATURE_REGEX_PATTERNS = {
    "Mileage": (r"(\d+\.?\d*)", float),
    "Engine": (r"(\d+)", float),
    "Kerb Weight": (r"(\d+)", float),
    "Power": (r"(\d+\.?\d*)", float),
    "Registration Year": (r"(\d{4})", float),
    "Transmission Type": {"Automatic": 1, "Manual": 0},
}

OHE_FEATURES = ["Fuel"]

