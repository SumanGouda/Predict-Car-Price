import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd


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

def clean_engine_type(value: Any) -> str:
    """Standardizes engine family variants into canonical labels without encoding."""
    if pd.isna(value) or str(value).strip().lower() in ["nan", "none", ""]:
        return "Unknown"

    cleaned_val = str(value).strip()
    val_upper = cleaned_val.upper()

    if "KAPPA" in val_upper:
        return "Kappa"

    k_series_variants = [
        "K10B",
        "K10C",
        "K12M",
        "K12N",
        "K14B",
        "K15B",
        "K15C",
        "K SERIES",
        "K-SERIES",
        "ADVANCED K",
    ]
    if any(x in val_upper for x in k_series_variants):
        return "K Series"

    ivtec_variants = ["I-VTEC", "I VTEC", "IVTEC"]
    if any(x in val_upper for x in ivtec_variants):
        return "i-VTEC"

    idtec_variants = ["I-DTEC", "I DTEC"]
    if any(x in val_upper for x in idtec_variants):
        return "i-DTEC"

    if "REVOTRON" in val_upper:
        return "Revotron"

    if "REVOTORQ" in val_upper:
        return "Revotorq"

    if "DDIS" in val_upper:
        return "DDiS"

    if "IRDE2" in val_upper:
        return "IRDE2"

    if "TSI" in val_upper:
        return "TSI"

    tdi_variants = ["TDI", "TDCI"]
    if any(x in val_upper for x in tdi_variants):
        return "TDI"

    crdi_variants = ["CRDI", "CRDE"]
    if any(x in val_upper for x in crdi_variants):
        return "CRDi"

    if "MHAWK" in val_upper:
        return "mHawk"

    if "MSTALLION" in val_upper:
        return "mStallion"

    kryotec_variants = ["KRYOTEC", "KRYOJET"]
    if any(x in val_upper for x in kryotec_variants):
        return "Kryotec"

    if "SMARTSTREAM" in val_upper:
        return "SmartStream"

    if "F8D" in val_upper:
        return "F8D"

    vvt_variants = ["VVT", "VTVT", "VVTI", "TI-VCT"]
    if any(x in val_upper for x in vvt_variants):
        return "VVT"

    if "TWINPOWER" in val_upper:
        return "TwinPower"

    toyota_variants = ["D-4D", "2KD-FTV", "2-GD FTV"]
    if any(x in val_upper for x in toyota_variants):
        return "Toyota Diesel"

    if "PETROL" in val_upper:
        return "Petrol"

    if "DIESEL" in val_upper:
        return "Diesel"

    inline_variants = ["IN-LINE", "IN LINE", "INLINE"]
    if any(x in val_upper for x in inline_variants):
        return "In-Line"

    return "Other"

def clean_data(
    df: pd.DataFrame, cleaning_dict: dict, func_clean_dict:dict, ohe_features: list, metadata_json_path: Path
) -> pd.DataFrame:
    
    """Cleans dataframe columns dynamically using regex extraction, dictionary
    mapping, custom cleaning functions, and One-Hot Encoding based on configuration settings.
    Ensures final output contains strictly numeric dtypes (no object, category, or boolean).
    """
    df = df.copy()
    ohe_metadata_registry = {}

    for col in list(df.columns):
        if col in cleaning_dict:
            rule = cleaning_dict[col]

            if isinstance(rule, tuple):
                pattern, dtype = rule
                extracted = df[col].astype(str).str.extract(pattern, expand=False)
                df[col] = pd.to_numeric(extracted, errors="coerce").astype(dtype)

            elif isinstance(rule, dict):
                df[col] = df[col].map(rule)

        if col in func_clean_dict:
            clean_func = func_clean_dict[col]
            df[col] = df[col].apply(clean_func)

        if col in ohe_features:
            dummies, meta = _ohe_encoding(df, column=col, drop_first=True)
            ohe_metadata_registry[col] = meta

            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
 
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)
 
    remaining_categorical = df.select_dtypes(include=["object", "category"]).columns
    if not remaining_categorical.empty:
        df = pd.get_dummies(df, columns=remaining_categorical, drop_first=True, dtype=int)

    if metadata_json_path and ohe_metadata_registry:
        metadata_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(ohe_metadata_registry, f, indent=4)

        print(f"OHE metadata successfully saved to: {metadata_json_path}")

    return df

def _ohe_encoding(
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
