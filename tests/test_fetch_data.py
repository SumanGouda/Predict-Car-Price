import json
from pathlib import Path
import sqlite3
import sys
import pandas as pd
import pytest
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
 
from predictions.mileage.fetch_data import main, process


@pytest.fixture
def setup_mock_environment(tmp_path):
    """Creates a temporary SQLite DB, features.txt, raw data CSV, and directory structure."""
    # Setup mock SQLite database with two city tables
    db_path = tmp_path / "mock_car_database.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE Mumbai (
            mileage TEXT,
            engine TEXT,
            fuel TEXT,
            transmission_type TEXT
        )
    """
    )
    cursor.executemany(
        "INSERT INTO Mumbai VALUES (?, ?, ?, ?)",
        [
            ("18.5 kmpl", "1197 cc", "Petrol", "Manual"),
            ("15.0 kmpl", "1497 cc", "Diesel", "Automatic"),
        ],
    )

    cursor.execute(
        """
        CREATE TABLE Delhi (
            mileage TEXT,
            engine TEXT,
            fuel TEXT,
            transmission_type TEXT
        )
    """
    )
    cursor.executemany(
        "INSERT INTO Delhi VALUES (?, ?, ?, ?)",
        [
            ("20.0 kmpl", "999 cc", "CNG", "Manual"),
        ],
    )

    conn.commit()
    conn.close()

    # Setup mock features.txt
    features_path = tmp_path / "features.txt"
    features_path.write_text("mileage\nengine\nfuel\ntransmission_type\n")

    # Setup output file paths
    raw_output_csv = tmp_path / "data" / "raw" / "data.csv"
    processed_output_csv = (
        tmp_path / "data" / "processed" / "processed_data.csv"
    )
    metadata_json_path = tmp_path / "data" / "metadata" / "ohe_metadata.json"

    # Mock cleaning rules matching settings.py
    clean_dict = {
        "mileage": (r"(\d+\.?\d*)", float),
        "engine": (r"(\d+)", float),
        "transmission_type": {"Automatic": 1, "Manual": 0},
    }
    ohe_features = ["fuel"]

    return {
        "db_path": db_path,
        "features_path": features_path,
        "raw_output_csv": raw_output_csv,
        "processed_output_csv": processed_output_csv,
        "metadata_json_path": metadata_json_path,
        "clean_dict": clean_dict,
        "ohe_features": ohe_features,
    }


def test_main_data_extraction(setup_mock_environment):
    """Tests feature extraction from SQLite city tables into raw CSV format."""
    env = setup_mock_environment

    df_extracted = main(
        features_txt_path=str(env["features_path"]),
        db_path=str(env["db_path"]),
        output_csv_path=str(env["raw_output_csv"]),
    )

    # Validate output properties
    assert isinstance(df_extracted, pd.DataFrame)
    assert len(df_extracted) == 3
    assert "source_city" in df_extracted.columns
    assert env["raw_output_csv"].exists()


def test_process_data_cleaning_and_ohe(setup_mock_environment):
    """Tests regex extraction, dictionary encoding, OHE generation, and JSON metadata logging."""
    env = setup_mock_environment

    # Run extraction step first
    main(
        features_txt_path=str(env["features_path"]),
        db_path=str(env["db_path"]),
        output_csv_path=str(env["raw_output_csv"]),
    )

    # Run processing function
    process(
        csv_file=str(env["raw_output_csv"]),
        clean_dict=env["clean_dict"],
        ohe_features=env["ohe_features"],
        metadata_json=str(env["metadata_json_path"]),
        output_path=env["processed_output_csv"],
    )

    # 1. Verify Processed CSV File Output
    assert env["processed_output_csv"].exists()
    df_processed = pd.read_csv(env["processed_output_csv"])

    # Check numeric regex transformations
    assert df_processed["mileage"].dtype in ["float64", "float32"]
    assert df_processed["engine"].dtype in ["float64", "float32"]

    # Check label encoding for transmission_type
    assert set(df_processed["transmission_type"].unique()).issubset({0, 1})

    # Check OHE dummy column creation
    assert "fuel" not in df_processed.columns
    assert any(col.startswith("fuel_") for col in df_processed.columns)

    # 2. Verify JSON Metadata Artifact
    assert env["metadata_json_path"].exists()
    with open(env["metadata_json_path"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert "fuel" in metadata
    assert "all_categories" in metadata["fuel"]
    assert "encoded_categories" in metadata["fuel"]
    assert "dropped_category" in metadata["fuel"]
