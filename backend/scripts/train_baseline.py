from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from app.preprocessing.preprocessor import build_tabular_dataset
from app.preprocessing.wesad import (
    available_subject_ids,
    prepare_binary_wesad_windows,
)


ARCHIVE = Path("data/archive (8).zip")

# Deterministic subject-independent split.
TEST_SUBJECTS = ("S13", "S14", "S15", "S16")


def main() -> None:
    subjects = available_subject_ids(ARCHIVE)
    train_subjects = tuple(
        subject for subject in subjects
        if subject not in TEST_SUBJECTS
    )

    print("All subjects  :", subjects)
    print("Train subjects:", train_subjects)
    print("Test subjects :", TEST_SUBJECTS)

    # Convert archive data into raw SensorReading windows first.
    train_windows, train_summary = prepare_binary_wesad_windows(
        ARCHIVE,
        train_subjects,
    )

    test_windows, test_summary = prepare_binary_wesad_windows(
        ARCHIVE,
        TEST_SUBJECTS,
    )

    # Convert raw windows into ML-ready tabular data.
    train_dataset = build_tabular_dataset(train_windows)
    test_dataset = build_tabular_dataset(test_windows)

    print("\nTrain rows:", len(train_dataset.rows))
    print("Test rows :", len(test_dataset.rows))

    print(
        "Train stress:",
        sum(row.stress_label == "STRESS" for row in train_dataset.rows),
    )
    print(
        "Train non-stress:",
        sum(row.stress_label == "NON_STRESS" for row in train_dataset.rows),
    )

    print(
        "Test stress:",
        sum(row.stress_label == "STRESS" for row in test_dataset.rows),
    )
    print(
        "Test non-stress:",
        sum(row.stress_label == "NON_STRESS" for row in test_dataset.rows),
    )

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

    print("\n=== RANDOM FOREST BASELINE ===")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["NON_STRESS", "STRESS"],
            digits=4,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    if len(set(y_test)) == 2:
        print("\nROC-AUC:", roc_auc_score(y_test, probabilities))


if __name__ == "__main__":
    main()