"""
train_model.py
==============
Trains and compares 5 regression models to estimate Instagram post reach
(impressions) from engagement signals, then saves the best one (Ridge) as a
serialized pipeline the Django app loads at runtime.

Run:
    python train_model.py

Output:
    analytics/models/ridge_pipeline.pkl   <- scaler + Ridge, ready to serve
    analytics/models/feature_order.json   <- feature order the app must follow

Notes on data leakage (important):
    The raw dataset has columns that break impressions down by source
    (From Home / From Hashtags / From Explore) which sum to the target, and an
    earlier 'engagement_rate = total_engagement / Impressions' feature that
    divides by the target. BOTH leak the answer and are deliberately excluded
    here, which is why the honest R2 (~0.91) is lower than a leaky ~0.95.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = "Instagram_Data.csv"
MODEL_DIR = os.path.join("analytics", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ridge_pipeline.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_order.json")

# The 12 leak-free features the model is trained on. The Django app builds its
# feature row in THIS EXACT ORDER, so keep them in sync.
FEATURES = [
    "Likes", "Saves", "Comments", "Shares", "Profile Visits", "Follows",
    "saves_per_like", "shares_per_like", "follow_rate", "total_engagement",
    "hashtag_count", "caption_len",
]


def load_and_engineer(path: str) -> pd.DataFrame:
    """Load the CSV and build the leak-free engineered features."""
    # latin1 because the captions contain emoji / special characters
    df = pd.read_csv(path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]

    # Derived behavioral features (+1 guards against divide-by-zero)
    df["total_engagement"] = df["Likes"] + df["Saves"] + df["Comments"] + df["Shares"]
    df["saves_per_like"] = df["Saves"] / (df["Likes"] + 1)
    df["shares_per_like"] = df["Shares"] / (df["Likes"] + 1)
    df["follow_rate"] = df["Follows"] / (df["Profile Visits"] + 1)
    df["hashtag_count"] = df["Hashtags"].astype(str).str.count("#")
    df["caption_len"] = df["Caption"].astype(str).str.len()

    return df


def compare_models(X, y):
    """Train all 5 models, print a comparison table, return best name by CV R2."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1.0),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
    }

    print(f"\n{'Model':<18}{'TestR2':>9}{'CV_R2':>8}{'MAE':>9}{'RMSE':>9}")
    print("-" * 53)

    best = None
    X_all_s = scaler.transform(X)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        r2 = r2_score(y_test, pred)
        cv_r2 = cross_val_score(model, X_all_s, y, cv=cv, scoring="r2").mean()
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        print(f"{name:<18}{r2:>9.3f}{cv_r2:>8.3f}{mae:>9.0f}{rmse:>9.0f}")
        if best is None or cv_r2 > best[1]:
            best = (name, cv_r2)

    print("-" * 53)
    print(f"Best by CV R2: {best[0]} ({best[1]:.3f})")
    print(
        "\nNote: Ridge is shipped even if GradientBoosting edges it on CV — on "
        "~119 rows that gap is within noise, and Ridge is more stable and "
        "interpretable."
    )
    return X_test, y_test, scaler


def report_ridge(X, y):
    """Print an honest, detailed report for the shipped Ridge model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_train, y_train)

    p_train, p_test = pipe.predict(X_train), pipe.predict(X_test)
    mape = np.mean(np.abs((y_test - p_test) / y_test)) * 100

    # Baseline: predict the mean every time
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    base_rmse = np.sqrt(mean_squared_error(y_test, dummy.predict(X_test)))
    ridge_rmse = np.sqrt(mean_squared_error(y_test, p_test))

    print("\n── Ridge detailed report ──────────────────────────────")
    print(f"Impressions: mean={y.mean():.0f}  median={y.median():.0f}  "
          f"range={y.min()}-{y.max()}")
    print(f"Train R2 = {r2_score(y_train, p_train):.3f}")
    print(f"Test  R2 = {r2_score(y_test, p_test):.3f}  "
          f"(overfit gap = {r2_score(y_train, p_train) - r2_score(y_test, p_test):+.3f})")
    print(f"Test MAE = {mean_absolute_error(y_test, p_test):.0f}")
    print(f"Test RMSE= {ridge_rmse:.0f}")
    print(f"Test MAPE= {mape:.1f}%")
    print(f"vs mean baseline RMSE {base_rmse:.0f} -> "
          f"{(1 - ridge_rmse / base_rmse) * 100:.0f}% lower error")


def train_and_save(X, y):
    """Fit the final Ridge pipeline on ALL data and serialize it."""
    # Fit on .values (numpy) so the served model carries no feature-name metadata
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X.values, y.values)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURES, f, indent=2)

    print(f"\nSaved model    -> {MODEL_PATH}")
    print(f"Saved features -> {FEATURES_PATH}")
    print(f"Trained on {len(X)} posts using {len(FEATURES)} leak-free features.")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run this from the project root."
        )

    df = load_and_engineer(DATA_PATH)
    X = df[FEATURES].fillna(0)
    y = df["Impressions"]

    compare_models(X, y)
    report_ridge(X, y)
    train_and_save(X, y)
    print("\nDone. The Django app will load the saved pipeline at startup.")


if __name__ == "__main__":
    main()
