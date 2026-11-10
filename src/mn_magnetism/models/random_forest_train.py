

# src/mn_magnetism/models/random_foresttrain.py

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


# -------------------------
# PATHS
# -------------------------

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data/processed/features.csv"
OUTPUT_DIR = ROOT / "artifacts/random_forest"


# -------------------------
# CONFIGURATION
# -------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "magnetic_order"

# Columns that must not be used as input features
EXCLUDE_COLUMNS = [
    "reduced_formula",
    "source_material_id",
    "data_id",
    TARGET,
]


# -------------------------
# TRAINING
# -------------------------

def train_random_forest():

    # 1. Load processed dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Separate features and targets
    X = df.drop(columns=EXCLUDE_COLUMNS, errors="ignore")
    y = df[TARGET]

    print(f"Total materials: {len(df)}")
    print("\nClass distribution:")
    print(y.value_counts())

    # 3. Identify numeric and categorical features
    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns found.")

    # 4. Preprocessing
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )),
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_cols,
        ))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            )),
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 6. Random forest pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    # 7. Train
    print("\nTraining random forest...")
    model.fit(X_train, y_train)

    # 8. Predict
    y_pred = model.predict(X_test)

    # Save test-set predictions for each material
    predictions_df = pd.DataFrame({
        "reduced_formula": df.loc[X_test.index, "reduced_formula"],
        "data_id": df.loc[X_test.index, "data_id"],
        "true": y_test,
        "predicted": y_pred,
    })

    predictions_df["correct"] = (
        predictions_df["true"] == predictions_df["predicted"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False
    )

    # 9. Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0,
    ))

    labels = model.named_steps["classifier"].classes_

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    print("\nConfusion matrix:")
    print(pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ))

    # 10. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        OUTPUT_DIR / "model.joblib"
    )

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
        "labels": labels.tolist(),
        "features": X.columns.tolist(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 11. Feature importance
    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        model.named_steps["classifier"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    print("\nTop 10 features:")
    print(importance_df.head(10))

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return model, metrics, importance_df


if __name__ == "__main__":
    train_random_forest()
