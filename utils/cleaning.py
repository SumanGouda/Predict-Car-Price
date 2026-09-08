import json
import pandas as pd
from pathlib import Path


def clean_data(
    df: pd.DataFrame, cleaning_dict: dict, ohe_features: list, metadata_json_path: Path
) -> pd.DataFrame:
    """Cleans dataframe columns dynamically using regex extraction, dictionary

    mapping, and One-Hot Encoding based on configuration settings.
    """
    df = df.copy()
    ohe_metadata_registry = {}

    for col in df.columns:
        if col in cleaning_dict:
            rule = cleaning_dict[col]
 
            if isinstance(rule, tuple):
                pattern, dtype = rule
                extracted = (
                    df[col].astype(str).str.extract(pattern, expand=False)
                )
                df[col] = pd.to_numeric(extracted, errors="coerce").astype(
                    dtype
                ) 

            elif isinstance(rule, dict):
                df[col] = df[col].map(rule)
 
        if col in ohe_features:
            dummies, meta = _ohe_encoding(df, column=col, drop_first=True)
            ohe_metadata_registry[col] = meta
 
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    if metadata_json_path and ohe_metadata_registry:
        metadata_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(ohe_metadata_registry, f, indent=4)

        print(f"OHE metadata successfully saved to: {metadata_json_path}")

    return df

def _ohe_encoding(df: pd.DataFrame, column: str, drop_first: bool = True) -> tuple[pd.DataFrame, dict]:
    """Applies One-Hot Encoding on a column and returns the dummy DataFrame

    along with categorical tracking metadata.
    """ 
    series = df[column].astype(str)

    dummies = pd.get_dummies(series, prefix=column, drop_first=drop_first)

    all_categories = sorted(series.unique().tolist())
    encoded_categories = list(dummies.columns)

    # Determine dropped base category
    if drop_first and len(all_categories) > 0:
        dropped_category = all_categories[0]
    else:
        dropped_category = None

    metadata = {
        "all_categories": all_categories,
        "encoded_categories": encoded_categories,
        "dropped_category": dropped_category,
    }

    return dummies, metadata
