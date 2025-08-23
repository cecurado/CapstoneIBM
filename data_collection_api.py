
import os
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_FILE = os.path.join(RAW_DIR, "spacex_raw.csv")

SPACEX_API = "https://api.spacexdata.com/v4/launches"

def fetch_spacex_launches() -> pd.DataFrame:
    """Fetch launches from SpaceX public API (v4) and normalize into a DataFrame."""
    resp = requests.get(SPACEX_API, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.json_normalize(data)
    return df

def save_raw(df: pd.DataFrame, path: str = RAW_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    print("📡 Fetching SpaceX launches from public API...")
    df = fetch_spacex_launches()
    print(f"   Retrieved {len(df)} launches.")
    save_raw(df, RAW_FILE)
    print(f"💾 Saved raw data to {RAW_FILE}")

if __name__ == "__main__":
    main()
