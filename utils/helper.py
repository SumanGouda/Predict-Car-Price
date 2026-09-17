import sqlite3
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve



def handle_missing_values(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Handles missing values in a specified DataFrame column based on the percentage

    of missing data.

    Rules:
    - 50% missing: Drop column.
    - 30% - 50% missing: Drop rows with missing values in column.
    - 10% - 30% missing: Fill missing values (mode for categorical, mean for numerical)
                         and create a missing indicator column (`{col}_was_missing`).
    - < 10% missing: Fill missing values (mode for categorical, median for numerical).
    """
    if col not in df.columns:
        print(f"Column '{col}' not found in DataFrame.")
        return df

    missing_count = df[col].isnull().sum()
    total_rows = len(df)

    if total_rows == 0:
        print("DataFrame is empty.")
        return df

    missing_pct = (missing_count / total_rows) * 100
    print(f"{col}: {missing_pct:.2f}% missing", end=" → ")

    # Drop column if > 50% missing
    if missing_pct > 50:
        df = df.drop(columns=[col])
        print("Dropped column (>50% missing)")

    # Drop rows if between 30% and 50% missing
    elif missing_pct > 30:
        df = df.dropna(subset=[col]).reset_index(drop=True)
        print("Dropped rows (30-50% missing)")

    # Fill + add indicator column if between 10% and 30% missing
    elif missing_pct > 10:
        indicator_col = f"{col}_was_missing"
        df[indicator_col] = df[col].isnull().astype(int)

        if df[col].dtype == "object" or isinstance(
            df[col].dtype, pd.CategoricalDtype
        ):
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            strategy = f"mode ('{fill_val}')"
        else:
            fill_val = df[col].mean()
            strategy = f"mean ({fill_val:.2f})"

        df[col] = df[col].fillna(fill_val)
        print(
            f"Filled with {strategy} + added indicator column '{indicator_col}' (10-30% missing)"
        )
    # Fill if < 10% missing
    elif missing_pct > 0:
        if df[col].dtype == "object" or isinstance(
            df[col].dtype, pd.CategoricalDtype
        ):
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            strategy = f"mode ('{fill_val}')"
        else:
            fill_val = df[col].median()
            strategy = f"median ({fill_val:.2f})"

        df[col] = df[col].fillna(fill_val)
        print(f"Filled with {strategy} (<10% missing)")
 
    else:
        print("No missing values")

    return df


def get_and_validate_features(features_file: Path, db_file: Path) -> list[str]:
    """Reads raw feature names from a text file and verifies their existence

    (case-insensitive) across tables in the SQLite database.

    Returns the original feature names from the file.
    """
    if not features_file.exists():
        raise FileNotFoundError(
            f"Features configuration file not found: {features_file}"
        )

    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_file}")

    # Read original feature names without lowercasing or replacing spaces
    with open(features_file, "r", encoding="utf-8") as f:
        target_columns = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    if not target_columns:
        raise ValueError(f"No valid features found in '{features_file.name}'.")

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE"
        " 'sqlite_%';"
    )
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        conn.close()
        raise ValueError(
            f"Database '{db_file.name}' does not contain any tables."
        )

    # Store DB column names normalized to lowercase for case-insensitive checking
    db_columns_lower = set()
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}');")
        for col in cursor.fetchall():
            db_columns_lower.add(col[1].strip().lower())

    conn.close()

    # Verify against normalized DB columns
    missing_columns = [
        col for col in target_columns if col.strip().lower() not in db_columns_lower
    ]

    if missing_columns:
        raise KeyError(
            f"Validation Failed: The following features from '{features_file.name}' "
            f"were not found in any database table: {missing_columns}"
        )

    print(
        f"Validation successful: All {len(target_columns)} features exist in the database."
    )
    return target_columns


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates R2, MAE, and RMSE regression evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2_score": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def generate_learning_curve_data(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = "r2",
) -> Dict[str, Any]:
    """Computes learning curve data across different training set batch sizes."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    return {
        "sizes": sizes.tolist(),
        "train_scores": np.mean(train_scores, axis=1).tolist(),
        "val_scores": np.mean(val_scores, axis=1).tolist(),
        "metric_name": scoring,
    }
