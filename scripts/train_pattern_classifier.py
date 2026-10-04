"""
Train the HOG + SVM pattern classifier.

    python scripts/train_pattern_classifier.py data/raw/NISTDB4

Folder names must contain 'arc', 'loop' or 'whorl'.

NOTE on evaluation: NIST SD4 contains two impressions (f/s) of every
finger. A random split can put the same finger in train and test and
inflate accuracy. If your file names let you recover the finger id,
pass --group-regex to split by finger; otherwise treat the reported
hold-out number as optimistic.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processing.classifier import MODEL_PATH, hog_features  # noqa: E402


def label_of(path):
    text = str(path).lower()
    for label in ("arc", "loop", "whorl"):
        if label in text:
            return label
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--group-regex", default=None,
                    help="regex with one group extracting the finger id from the file name")
    ap.add_argument("--out", default=MODEL_PATH)
    args = ap.parse_args()

    files = sorted(
        p for p in Path(args.root).rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    )
    files = [(p, label_of(p)) for p in files]
    files = [(p, y) for p, y in files if y]

    X = np.vstack([hog_features(cv2.imread(str(p), 0)) for p, _ in files])
    y = np.array([l for _, l in files])

    if args.group_regex:
        groups = np.array([re.search(args.group_regex, p.name).group(1) for p, _ in files])
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, groups))
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                                  random_state=42, stratify=y)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=5.0, gamma="scale", probability=True,
                    class_weight="balanced", random_state=42)),
    ])
    model.fit(X[tr], y[tr])
    print(classification_report(y[te], model.predict(X[te])))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.fit(X, y)     # final model on all data
    joblib.dump(model, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
