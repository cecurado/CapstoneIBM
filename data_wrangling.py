
import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
RAW_FILE = os.path.join(RAW_DIR, "spacex_raw.csv")
CLEAN_FILE = os.path.join(RAW_DIR, "spacex_clean.csv")

def load_raw(path: str = RAW_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at {path}. Run data_collection_api.py first.")
    return pd.read_csv(path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features and the classification variable (Class) to match course goals.
    Here, we approximate 'landing success' using the 'cores' list items where available.
    In the IBM labs, Class==1 indicates a successful first-stage landing; 0 otherwise.
    """
    # Normalize some helpful columns
    df['date_utc'] = pd.to_datetime(df.get('date_utc'))
    df['year'] = df['date_utc'].dt.year

    # Outcome proxy: derive from cores' landing success (if present)
    # For v4 API, cores is a list of dicts with fields like landing_success, landing_type, etc.
    def landing_class(row):
        try:
            # cores.0.landing_success after json_normalize; fall back to parsing object
            if 'cores.0.landing_success' in row and not pd.isna(row['cores.0.landing_success']):
                return int(bool(row['cores.0.landing_success']))
            # Fallback for when cores is embedded as a JSON string/object
            cores = row.get('cores')
            if isinstance(cores, str) and cores.startswith('['):
                import json
                cores_obj = json.loads(cores)
            else:
                cores_obj = cores
            if isinstance(cores_obj, list) and len(cores_obj):
                ls = cores_obj[0].get('landing_success')
                if ls is None:
                    return 0
                return int(bool(ls))
            return 0
        except Exception:
            return 0

    df['Class'] = df.apply(landing_class, axis=1)

    # Basic engineered predictors often explored in the course:
    df['has_fairings'] = df.get('fairings.reused', pd.Series([np.nan]*len(df))).fillna(False).astype(int)
    df['reused_count'] = df.get('fairings.recovered', pd.Series([np.nan]*len(df))).fillna(False).astype(int)
    df['rocket_id'] = df.get('rocket')
    df['success'] = df.get('success').fillna(False).astype(int)

    # Categorical launch site (pad)
    site = df.get('launchpad', pd.Series([np.nan]*len(df))).fillna("unknown")
    df['launch_site'] = site

    # Destination/orbit proxy: use 'payloads' existence count
    payloads_col = df.get('payloads')
    if payloads_col is not None:
        df['payload_count'] = payloads_col.apply(lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else (len(x) if isinstance(x, list) else 0))
    else:
        df['payload_count'] = 0

    # Keep a compact set of columns for ML
    keep_cols = ['year', 'has_fairings', 'reused_count', 'success', 'launch_site', 'payload_count', 'Class']
    clean = df[keep_cols].copy()

    # Encode launch_site as category
    clean['launch_site'] = clean['launch_site'].astype('category')
    for cat in clean['launch_site'].cat.categories:
        clean[f'launch_site__{cat}'] = (clean['launch_site'] == cat).astype(int)
    clean.drop(columns=['launch_site'], inplace=True)

    return clean

def save_clean(df: pd.DataFrame, path: str = CLEAN_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    print("🧼 Loading raw data...")
    raw = load_raw(RAW_FILE)
    print(f"   Raw shape: {raw.shape}")
    print("🧪 Engineering features...")
    clean = engineer_features(raw)
    print(f"   Clean shape: {clean.shape}")
    save_clean(clean, CLEAN_FILE)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print(f"💾 Saved cleaned data to {CLEAN_FILE}")

if __name__ == "__main__":
    main()
