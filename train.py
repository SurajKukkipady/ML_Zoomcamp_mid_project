"""Train, evaluate, and save the final Netflix title-type classifier."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

from model import prepare_data, make_search


def top_feature_importance(estimator, limit=15):
    """Return the strongest weights or importances for the selected estimator."""
    classifier = estimator.named_steps["classifier"]
    names = estimator.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]
        classes = classifier.classes_
        results = []
        for label, order in (
            (classes[0], coefficients.argsort()[:limit]),
            (classes[1], coefficients.argsort()[-limit:][::-1]),
        ):
            results.extend(
                {
                    "class": label,
                    "feature": names[index],
                    "importance": float(coefficients[index]),
                    "importance_type": "signed_coefficient",
                }
                for index in order
            )
        return results

    importances = classifier.feature_importances_
    order = importances.argsort()[-limit:][::-1]
    return [
        {
            "class": "all",
            "feature": names[index],
            "importance": float(importances[index]),
            "importance_type": "tree_feature_importance",
        }
        for index in order
        if importances[index] > 0
    ]


def train(data_path, model_path, cv=5, n_jobs=-1):
    data = pd.read_csv(data_path)
    features, target = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    search = make_search(cv=cv, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)

    print(f"Best cross-validation macro F1: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(
        "Holdout balanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )
    print(f"Holdout macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, model_path)
    print(f"Saved model to {model_path}")

    results = pd.DataFrame(search.cv_results_)
    results["model"] = results["param_classifier"].map(
        lambda classifier: classifier.__class__.__name__
    )
    summary = (
        results.sort_values("mean_test_macro_f1", ascending=False)
        .groupby("model", as_index=False)
        .first()
        [["model", "mean_test_macro_f1", "mean_test_balanced_accuracy", "params"]]
        .rename(
            columns={
                "mean_test_macro_f1": "cv_macro_f1",
                "mean_test_balanced_accuracy": "cv_balanced_accuracy",
            }
        )
        .sort_values("cv_macro_f1", ascending=False)
    )
    summary_path = model_path.parent / "model_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved cross-validation model comparison to {summary_path}")
    print(summary.to_string(index=False))

    importance = top_feature_importance(search.best_estimator_)
    if importance:
        importance_path = model_path.parent / "feature_importance.csv"
        pd.DataFrame(importance).to_csv(importance_path, index=False)
        print(f"Saved feature importance to {importance_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("netflix_titles.csv"),
        help="Path to the Netflix titles CSV",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/netflix_type.joblib"),
        help="Where to save the trained model",
    )
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Parallel jobs for model selection"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model, cv=args.cv, n_jobs=args.n_jobs)
