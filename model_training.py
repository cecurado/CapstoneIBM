
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
CLEAN_FILE = os.path.join(DATA_DIR, "spacex_clean.csv")
BEST_MODEL_FILE = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
METRICS_FILE = os.path.join(ARTIFACTS_DIR, "metrics.txt")

def load_clean(path: str = CLEAN_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean data not found at {path}. Run data_wrangling.py first.")
    return pd.read_csv(path)

def get_xy(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler(with_mean=False)  # some columns are already dummy
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "svm_linear": SVC(kernel="linear", probability=True),
        "svm_rbf": SVC(kernel="rbf", probability=True),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "dtree": DecisionTreeClassifier(random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "report": classification_report(y_test, y_pred, digits=4)
        }
    return results, scaler

def persist_best(results, scaler):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    joblib.dump({"model": best_model, "scaler": scaler}, BEST_MODEL_FILE)

    with open(METRICS_FILE, "w") as f:
        for name, res in results.items():
            f.write(f"== {name} ==\n")
            f.write(f"accuracy: {res['accuracy']:.4f}\n")
            f.write(res["report"] + "\n\n")
        f.write(f"BEST: {best_name}\n")

    return best_name

def main():
    print("🤖 Loading cleaned data...")
    df = load_clean(CLEAN_FILE)
    X, y = get_xy(df)
    print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
    results, scaler = train_models(X, y)
    best = persist_best(results, scaler)
    print(f"🏆 Best model saved to {BEST_MODEL_FILE} ({best})")
    print(f"📝 Metrics saved to {METRICS_FILE}")

if __name__ == "__main__":
    main()
