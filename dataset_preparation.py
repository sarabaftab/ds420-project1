"""Shared data preparation for the phishing URL classification project.

Investigation and training notebooks should import this file so every model
uses the same cleaned rows, the same split, and the same preprocessing.

    from dataset_preparation import prepare_datasets

    data = prepare_datasets()
    pipeline = data["pipeline"]
    X_train_prepared = pipeline.fit_transform(data["X_train"])
    X_dev_prepared = pipeline.transform(data["X_dev"])
    X_test_prepared = pipeline.transform(data["X_test"])

The returned pipeline is not fitted yet. Fit it on the training set only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "Dataset.csv"
TARGET_COLUMN = "label"
TEXT_COLUMNS = ["url", "dom", "tld"]

# 60% train, 20% dev, 20% test.
# dev_size is the share of the train+dev block, not the share of all rows.
TEST_SIZE = 0.20
DEV_SIZE = 0.25
RANDOM_STATE = 42


def load_raw_data(path=DATA_PATH):
    """Load the original URL dataset."""
    return pd.read_csv(path)


def remove_duplicate_urls(df):
    """Remove repeated observations before the split.

    Duplicate rows in this file are repeated URLs. Removing them first keeps
    the same URL from landing in both training and evaluation.
    """
    return df.drop_duplicates().reset_index(drop=True)


def split_features_and_target(df):
    """Separate predictors from the label.

    The label is numeric, so it has to leave the feature table before
    preprocessing. Otherwise the pipeline would scale it as a predictor.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_train_dev_test(
    X,
    y,
    test_size=TEST_SIZE,
    dev_size=DEV_SIZE,
    random_state=RANDOM_STATE,
):
    """Stratified train/dev/test split.

    Dev is for model comparison and grid search. Test stays unused until the
    final evaluation in the conclusion notebook.
    """
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_dev,
        y_train_dev,
        test_size=dev_size,
        stratify=y_train_dev,
        random_state=random_state,
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def build_preparation_pipeline():
    """Build the preprocessing pipeline.

    Numeric URL features are imputed and standardized. Raw text columns are
    dropped inside the pipeline. The scaler is fitted only when the caller
    fits this pipeline on the training set.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include="number"),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("prepare", column_transformer),
        ]
    )


def prepare_datasets(path=DATA_PATH, random_state=RANDOM_STATE):
    """Return the shared splits and an unfitted preparation pipeline."""
    cleaned = remove_duplicate_urls(load_raw_data(path))
    X, y = split_features_and_target(cleaned)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(
        X,
        y,
        random_state=random_state,
    )
    return {
        "X_train": X_train,
        "X_dev": X_dev,
        "X_test": X_test,
        "y_train": y_train,
        "y_dev": y_dev,
        "y_test": y_test,
        "pipeline": build_preparation_pipeline(),
    }


def _phishing_percent(y):
    return 100 * float(np.mean(y))


def main():
    raw = load_raw_data()
    cleaned = remove_duplicate_urls(raw)
    X, y = split_features_and_target(cleaned)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y)

    print("Raw rows:", len(raw))
    print("Rows after removing duplicates:", len(cleaned))
    print("Duplicate rows removed:", len(raw) - len(cleaned))
    print("Predictors before the pipeline:", X.shape[1])
    print()

    split_rows = {
        "train": len(y_train),
        "dev": len(y_dev),
        "test": len(y_test),
    }
    for name, labels in (("train", y_train), ("dev", y_dev), ("test", y_test)):
        print(
            f"{name:5} rows: {split_rows[name]:6}   "
            f"phishing: {_phishing_percent(labels):5.2f}%"
        )
    print()

    pipeline = build_preparation_pipeline()
    X_train_prepared = pipeline.fit_transform(X_train)
    scaler_means = pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["scaler"].mean_.copy()

    X_dev_prepared = pipeline.transform(X_dev)
    X_test_prepared = pipeline.transform(X_test)
    scaler_means_after = pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["scaler"].mean_

    feature_names = list(pipeline.get_feature_names_out())
    train_means = X_train_prepared.mean(axis=0)
    train_stds = X_train_prepared.std(axis=0)

    print("Pipeline steps:", list(pipeline.named_steps))
    print("Prepared features:", len(feature_names))
    print("Train prepared shape:", X_train_prepared.shape)
    print("Dev prepared shape:", X_dev_prepared.shape)
    print("Test prepared shape:", X_test_prepared.shape)
    print("Train scaled mean (largest absolute):", round(float(np.max(np.abs(train_means))), 6))
    print("Train scaled std (min, max):", round(float(train_stds.min()), 4), round(float(train_stds.max()), 4))
    print()

    train_urls = set(X_train["url"])
    dev_urls = set(X_dev["url"])
    test_urls = set(X_test["url"])
    checks = {
        "no shared URLs across splits": train_urls.isdisjoint(dev_urls)
        and train_urls.isdisjoint(test_urls)
        and dev_urls.isdisjoint(test_urls),
        "text columns removed": all(column not in feature_names for column in TEXT_COLUMNS),
        "22 numeric predictors": len(feature_names) == 22,
        "no missing prepared values": not np.isnan(X_train_prepared).any()
        and not np.isnan(X_dev_prepared).any()
        and not np.isnan(X_test_prepared).any(),
        "training features are standardized": np.allclose(train_means, 0, atol=1e-6)
        and np.allclose(train_stds, 1, atol=1e-6),
        "scaler stayed fit on training data": np.allclose(scaler_means, scaler_means_after),
        "split sizes sum to the cleaned data": sum(split_rows.values()) == len(cleaned),
    }
    for name, passed in checks.items():
        print(("PASS" if passed else "FAIL"), name)

    if not all(checks.values()):
        raise SystemExit(1)

    print()
    print(
        "Preparation is ready. The three splits keep about 14.4% phishing, "
        "the raw text columns are out of the modeling data, and the scaler "
        "was fit on training rows only."
    )


if __name__ == "__main__":
    main()
