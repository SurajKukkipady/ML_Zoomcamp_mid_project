"""Shared preprocessing and estimator configuration for title-type prediction."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


FEATURE_COLUMNS = [
    "description",
    "listed_in",
    "country",
    "rating",
    "release_year",
]
TARGET_COLUMN = "type"


def prepare_data(data):
    """Keep the modeled fields and normalize missing or malformed values."""
    data = data.copy()
    data = data.dropna(subset=[TARGET_COLUMN])
    data = data[data[TARGET_COLUMN].isin(["Movie", "TV Show"])]

    for column in ("description", "listed_in", "country"):
        data[column] = data[column].fillna("").astype(str)

    data["rating"] = data["rating"].fillna("Unknown").astype(str)
    data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")

    return data[FEATURE_COLUMNS], data[TARGET_COLUMN]


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "description",
                TfidfVectorizer(
                    max_features=30000,
                    min_df=2,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
                "description",
            ),
            (
                "genres",
                TfidfVectorizer(
                    max_features=10000,
                    min_df=2,
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b[\w&-]+\b",
                ),
                "listed_in",
            ),
            (
                "countries",
                TfidfVectorizer(
                    max_features=5000,
                    min_df=2,
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b[\w&-]+\b",
                ),
                "country",
            ),
            (
                "rating",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["rating"],
            ),
            (
                "release_year",
                SimpleImputer(strategy="median"),
                ["release_year"],
            ),
        ],
        remainder="drop",
    )


def make_pipeline(classifier):
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", classifier),
        ]
    )


def make_search(cv=5, n_jobs=-1):
    """Compare and tune three model families using macro F1 for selection."""
    baseline = LogisticRegression(max_iter=2000, class_weight="balanced")
    return GridSearchCV(
        estimator=make_pipeline(baseline),
        param_grid=[
            {
                "classifier": [baseline],
                "classifier__C": [0.1, 1.0, 5.0],
            },
            {
                "classifier": [LinearSVC(class_weight="balanced")],
                "classifier__C": [0.1, 1.0, 5.0],
            },
            {
                "classifier": [
                    DecisionTreeClassifier(
                        class_weight="balanced", random_state=42
                    )
                ],
                "classifier__max_depth": [5, 10, 20],
                "classifier__min_samples_leaf": [1, 5],
            },
        ],
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        refit="macro_f1",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
    )
