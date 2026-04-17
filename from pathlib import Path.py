from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = SCRIPT_DIR / "traindataset.json"
TEST_FILE = SCRIPT_DIR / "testdataset.json"

TARGET_COLUMN = "Label"
DROP_COLUMNS = ["@timestamp", "AttackCategory"]
NUMERIC_PARSE_THRESHOLD = 0.8


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return pd.read_json(path, lines=True)


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned = cleaned[cleaned[TARGET_COLUMN].isin([0, 1])].copy()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned


def infer_column_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in features.columns:
        series = features[column]

        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        parsed = pd.to_numeric(series, errors="coerce")
        parse_rate = parsed.notna().mean()

        if parse_rate >= NUMERIC_PARSE_THRESHOLD:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str]]:
    train_df = clean_target(train_df)
    test_df = clean_target(test_df)

    drop_in_train = [column for column in DROP_COLUMNS if column in train_df.columns]
    drop_in_test = [column for column in DROP_COLUMNS if column in test_df.columns]

    train_df = train_df.drop(columns=drop_in_train, errors="ignore")
    test_df = test_df.drop(columns=drop_in_test, errors="ignore")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    numeric_columns, categorical_columns = infer_column_types(X_train)

    for column in numeric_columns:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        X_test[column] = pd.to_numeric(X_test[column], errors="coerce")

    for column in categorical_columns:
        X_train[column] = X_train[column].astype("string").str.strip()
        X_test[column] = X_test[column].astype("string").str.strip()

    return X_train, y_train, X_test, y_test, numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def main() -> None:
    train_df = load_dataset(TRAIN_FILE)
    test_df = load_dataset(TEST_FILE)

    X_train, y_train, X_test, y_test, numeric_columns, categorical_columns = prepare_features(
        train_df, test_df
    )

    print(f"Training rows used: {len(X_train)}")
    print(f"Test rows used: {len(X_test)}")
    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")
    print()

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    }

    results: list[dict[str, float | str]] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        results.append(
            {
                "Model": name,
                "Train Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
            }
        )
        fitted_pipelines[name] = pipeline

        print(name)
        print(f"  Train accuracy: {train_accuracy:.4f}")
        print(f"  Test accuracy:  {test_accuracy:.4f}")
        print()

    results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_model_name]

    print("Final ranking:")
    print(results_df.to_string(index=False))
    print()
    print(f"Best model: {best_model_name}")
    print()

    best_predictions = best_pipeline.predict(X_test)
    print("Confusion matrix for best model:")
    print(confusion_matrix(y_test, best_predictions))
    print()
    print("Classification report for best model:")
    print(classification_report(y_test, best_predictions, digits=4))


if __name__ == "__main__":
    main()
