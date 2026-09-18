import json
from pathlib import Path 
import pandas as pd
import numpy as np
from typing import Any


def clean_drive_type(value: Any) -> str:
    """Standardizes drive type variants into canonical labels without encoding."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return "Unknown"

    cleaned_val = str(value).strip().upper().replace(" ", "")

    fwd_variants = {
        "FWD",
        "2WD",
        "4X2",
        "FRONTWHEELDRIVE",
        "TWOWHEELDRIVE",
        "TWOWHHEELDRIVE",
        "TWOWHHHEELDRIVE",
    }
    rwd_variants = {"RWD", "RWD(WITHMTT)"}
    awd_variants = {"AWD", "4WD", "4X4"}

    if cleaned_val in fwd_variants:
        return "FWD"
    if cleaned_val in rwd_variants:
        return "RWD"
    if cleaned_val in awd_variants:
        return "AWD"

    return "Unknown"

def clean_turbo_charger(value: Any) -> str:
    """Standardizes turbo charger variants into canonical labels without encoding."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return "Unknown"

    cleaned_val = str(value).strip().upper().replace(" ", "")

    no_variants = {
        "NO",
        "N",
        "NA",
        "N/A",
        "NONE",
        "FALSE",
        "0",
        "NOTAVAILABLE",
        "NOTURBO",
        "NATURALLYASPIRATED",
    }
    yes_variants = {
        "YES",
        "Y",
        "TRUE",
        "1",
        "TURBO",
        "TURBOCHARGED",
        "SINGLETURBO",
        "SINGLE",
    }
    twin_variants = {
        "TWIN",
        "TWINTURBO",
        "DUALTURBO",
        "DUAL",
        "BITURBO",
        "BI-TURBO",
        "TWIN-TURBO",
    }

    if cleaned_val in no_variants:
        return "No"
    if cleaned_val in yes_variants:
        return "Yes"
    if cleaned_val in twin_variants:
        return "Twin"

    return "Unknown"

def clean_emission_norm(value: Any, map_dict) -> int:
    """Normalizes emission norm variants and maps them to an ordinal strictness rank."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return map_dict["Unknown"]

    cleaned_val = str(value).strip().upper()

    if "ZEV" in cleaned_val:
        canonical = "ZEV"
    elif "6.0" in cleaned_val or "VI 2.0" in cleaned_val:
        canonical = "BS VI 2.0"
    elif "BS III" in cleaned_val or "BSIII" in cleaned_val or "BHARAT STAGE III" in cleaned_val:
        canonical = "BS III"
    elif "BS IV" in cleaned_val or "BSIV" in cleaned_val or "BHARAT STAGE IV" in cleaned_val:
        canonical = "BS IV"
    elif "BS VI" in cleaned_val or "BSVI" in cleaned_val or "BHARAT STAGE VI" in cleaned_val:
        canonical = "BS VI"
    elif "BS II" in cleaned_val or "BHARAT STAGE II" in cleaned_val:
        canonical = "BS II"
    elif "BS I" in cleaned_val or "BHARAT STAGE I" in cleaned_val:
        canonical = "BS I"
    elif "EURO VI" in cleaned_val or "EU 6" in cleaned_val:
        canonical = "Euro VI"
    elif "EURO V" in cleaned_val:
        canonical = "Euro V"
    elif "EURO IV" in cleaned_val:
        canonical = "Euro IV"
    else:
        canonical = "Unknown"

    return map_dict[canonical]

def clean_price(value: Any) -> float:
    """Converts price strings with Lakh/Crore/Thousand suffixes into a numeric value, rounded to 1 decimal place."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return np.nan

    cleaned_val = str(value).replace("₹", "").strip()

    if "Lakh" in cleaned_val:
        return round(float(cleaned_val.replace("Lakh", "").strip()) * 100000, 1)
    if "Crore" in cleaned_val:
        return round(float(cleaned_val.replace("Crore", "").strip()) * 10000000, 1)
    if "Thousand" in cleaned_val:
        return round(float(cleaned_val.replace("Thousand", "").strip()) * 1000, 1)

    return np.nan

def clean_ownership(value: Any, map_dict: dict) -> int:
    """Maps ownership label variants to an ordinal rank via map_dict."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return 0

    cleaned_val = str(value).strip().title()
    return map_dict.get(cleaned_val, 0)

def clean_car_name(value: Any) -> tuple[str, str]:
    """Splits a raw car name string into (brand, model) without encoding."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return "Unknown", "Unknown"

    parts = str(value).strip().split(" ", 1)
    brand = parts[0] if parts[0] else "Unknown"
    model = parts[1] if len(parts) > 1 and parts[1] else "Unknown"

    return brand, model

def apply_cleaning_pipeline(
    df: pd.DataFrame,
    regex_clean_dict: dict,
    func_clean_dict: dict,
    ohe_features: list,
    metadata_json_path: Path
) -> pd.DataFrame:
    """Cleans dataframe columns dynamically using regex extraction, dictionary
    mapping, custom cleaning functions, and One-Hot Encoding based on configuration settings.
    Any remaining object/category columns are left untouched (e.g. for XGBoost's
    native categorical handling) and reported at the end.
    """
    df = df.copy()
    ohe_metadata_registry = {}

    for col in list(df.columns):
        if col in regex_clean_dict:
            rule = regex_clean_dict[col]

            if isinstance(rule, tuple):
                pattern, dtype = rule
                extracted = df[col].astype(str).str.extract(pattern, expand=False)
                df[col] = pd.to_numeric(extracted, errors="coerce").astype(dtype)

            elif isinstance(rule, dict):
                df[col] = df[col].map(rule)

        if col in func_clean_dict:
            rule = func_clean_dict[col]

            if isinstance(rule, tuple):
                clean_func, new_cols = rule
                df[new_cols] = df[col].apply(clean_func).apply(pd.Series)
                df = df.drop(columns=[col])
            else:
                df[col] = df[col].apply(rule)

        if col in ohe_features:
            dummies, meta = _encode_column_ohe(df, column=col, drop_first=True)
            ohe_metadata_registry[col] = meta

            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)

    remaining_categorical = df.select_dtypes(include=["object", "category"]).columns
    if not remaining_categorical.empty:
        print(f"Columns left as string/object dtype (not encoded): {list(remaining_categorical)}")

    if metadata_json_path and ohe_metadata_registry:
        metadata_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(ohe_metadata_registry, f, indent=4)

        print(f"OHE metadata successfully saved to: {metadata_json_path}")

    return df

def _encode_column_ohe(
    df: pd.DataFrame, column: str, drop_first: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Generates one-hot encoded dummies and tracks category mappings and dropped reference level."""
    categories = sorted(df[column].dropna().unique().tolist())
    dummies = pd.get_dummies(
        df[column], prefix=column, drop_first=drop_first, dtype=int
    )

    dropped_feature = categories[0] if (drop_first and categories) else None

    metadata = {
        "original_column": column,
        "all_categories": categories,
        "encoded_columns": list(dummies.columns),
        "dropped_baseline_category": dropped_feature,
    }

    return dummies, metadata

    