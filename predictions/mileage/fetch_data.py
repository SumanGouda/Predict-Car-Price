import sys
import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.helper import get_and_validate_features
from utils.cleaning import clean_data
from config.settings import (
    db_file, output_file_mileage, mileage_features_file, ohe_metadata_file, processed_file_mileage,
    process_raw_data, FEATURE_REGEX_PATTERNS, OHE_FEATURES 
)

def main(features_txt_path: str, db_path: str, output_csv_path: str) -> pd.DataFrame:
    """Reads target features from features.txt, extracts them from all city tables

    in the SQLite database, combines them into a single dataset, and exports to
    CSV.
    """
    db_file         = Path(db_path)
    output_file     = Path(output_csv_path)
    features_file   = Path(features_txt_path)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)

    target_columns = get_and_validate_features(features_file, db_file)

    conn    = sqlite3.connect(db_file)
    cursor  = conn.cursor()
 
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        print("No tables found in the database.")
        conn.close()
        return pd.DataFrame()

    all_data = []
 
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}');")

        # Map lowercased column name -> actual database column name
        db_cols_map = {
            col[1].strip().lower(): col[1] for col in cursor.fetchall()
        }

        select_clauses = []
        for col in target_columns:
            col_lower = col.strip().lower()
            if col_lower in db_cols_map:
                actual_db_col = db_cols_map[col_lower]
                # Select real database column and alias it back to target column name
                select_clauses.append(f'"{actual_db_col}" AS "{col}"')
            else:
                select_clauses.append(f'NULL AS "{col}"')

        query = f"SELECT {', '.join(select_clauses)} FROM \"{table}\""

        try:
            df_table = pd.read_sql_query(query, conn)
            if not df_table.empty:
                df_table["source_city"] = table
                all_data.append(df_table)
        except Exception as e:
            print(f"Error querying table '{table}': {e}")

    conn.close()
    
    if not all_data:
        print("No data retrieved from any table.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.dropna(how="all", subset=target_columns, inplace=True)
    combined_df.drop_duplicates(subset=target_columns, inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    combined_df.to_csv(output_file, index=False)
    print(f"Loaded {len(target_columns)} features from '{features_file.name}'.")
    print(f"Successfully extracted {len(combined_df)} records across {len(tables)} city tables.")
    print(f"Warehouse dataset saved to: {output_file}")

    return combined_df

def process(csv_file: str, clean_dict: dict, ohe_features: list, metadata_json: str, output_path: str):
    csv_file = Path(csv_file)
    metadata_json = Path(metadata_json)
    output_path = Path(output_path)

    df = pd.read_csv(csv_file)
    df = clean_data(df, clean_dict, ohe_features, metadata_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main(mileage_features_file, db_file, output_file_mileage)

    if process_raw_data:
        process(output_file_mileage, FEATURE_REGEX_PATTERNS, OHE_FEATURES, ohe_metadata_file, processed_file_mileage)

