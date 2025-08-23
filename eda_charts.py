
import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
CLEAN_FILE = os.path.join(DATA_DIR, "spacex_clean.csv")

def load_clean(path: str = CLEAN_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean data not found at {path}. Run data_wrangling.py first.")
    return pd.read_csv(path)

def plot_and_save(df: pd.DataFrame, col: str, fname: str):
    plt.figure()
    df[col].value_counts(dropna=False).sort_index().plot(kind="bar")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out = os.path.join(ARTIFACTS_DIR, fname)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Saved {out}")

def main():
    print("📈 Loading cleaned data for EDA...")
    df = load_clean(CLEAN_FILE)

    # Simple charts: class distribution and year histogram
    plot_and_save(df, "Class", "class_distribution.png")

    plt.figure()
    df["year"].dropna().astype(int).plot(kind="hist", bins=20)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out = os.path.join(ARTIFACTS_DIR, "launch_year_hist.png")
    plt.title("Launch Year Histogram")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Saved {out}")

if __name__ == "__main__":
    main()
