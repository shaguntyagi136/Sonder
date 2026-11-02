from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.preprocessing.preprocessor import build_tabular_dataset
from app.preprocessing.wesad import (
    available_subject_ids,
    prepare_binary_wesad_windows,
)


ARCHIVE = Path("data/archive (8).zip")


def main() -> None:
    subjects = available_subject_ids(ARCHIVE)

    results = []

    for test_subject in subjects:
        train_subjects = tuple(
            subject for subject in subjects
            if subject != test_subject
        )

        train_windows, _ = prepare_binary_wesad_windows(
            ARCHIVE,
            train_subjects,
        )

        test_windows, _ = prepare_binary_wesad_windows(
            ARCHIVE,
            (test_subject,),
        )

        train_dataset = build_tabular_dataset(train_windows)
        test_dataset = build_tabular_dataset(test_windows)

        X_train = train_dataset.X
        y_train = [
            1 if label == "STRESS" else 0
            for label in train_dataset.y
        ]

        X_test = test_dataset.X
        y_test = [
            1 if label == "STRESS" else 0
            for label in test_dataset.y
        ]

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        result = {
            "subject": test_subject,
            "rows": len(y_test),
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": (
                roc_auc_score(y_test, probabilities)
                if len(set(y_test)) == 2
                else None
            ),
        }

        results.append(result)

        print(
            f"{test_subject}: "
            f"rows={result['rows']} "
            f"accuracy={result['accuracy']:.4f} "
            f"precision={result['precision']:.4f} "
            f"recall={result['recall']:.4f} "
            f"f1={result['f1']:.4f} "
            f"roc_auc={result['roc_auc']}"
        )

    print("\n=== LOSO SUMMARY ===")

    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ):
        values = [
            result[metric]
            for result in results
            if result[metric] is not None
        ]

        print(
            f"{metric.upper():10s}: "
            f"{sum(values) / len(values):.4f}"
        )


if __name__ == "__main__":
    main()