from collections import Counter
from app.preprocessing.wesad import (
    available_subject_ids,
    load_wesad_participant,
    prepare_binary_wesad_windows,
)

archive = "data/archive (8).zip"
counts = Counter()
total = 0

subjects = available_subject_ids(archive)

print("Subjects:", subjects)
print()

for subject_id in subjects:
    participant = load_wesad_participant(archive, subject_id)
    windows = prepare_binary_wesad_windows(participant)

    total += len(windows)
    counts.update(w.stress_label for w in windows)

    print(f"{subject_id}: {len(windows)} labeled windows")

print()
print("TOTAL:", total)
print("STRESS:", counts["STRESS"])
print("NON_STRESS:", counts["NON_STRESS"])
