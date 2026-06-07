"""
bm25_rf_standardize.py  —  BM25 features + Random Forest classifier
=====================================================================
Same logic as classifier_standardize.py (TF-IDF + Random Forest),
but replaces TF-IDF feature extraction with BM25 token-based features.

Install:
  pip install rank-bm25 pandas scikit-learn openpyxl

Usage:
  python bm25_rf_standardize.py
  python bm25_rf_standardize.py good.xlsx nogood.xlsx output.xlsx
"""

import re, sys
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

CONFIDENCE_THRESHOLD = 0.60
VALIDATION_SIZE      = 0.20
TOP_N                = 30    # top-N BM25 scores used as features

# ── Color tokens ──────────────────────────────────────────────────────────────
COLOR_TOKENS = [
    "GRN/YEL-ST","GRN/YEL-RI","WHT/GRY-ST","GN/YW-ST",
    "ORG/WHT","ORG/BLK","RED/WHT","RED/BLK","WHT/BLU","BLU/WHT",
    "WHT/BLK","WHT/RED","WHT/GRN","WHT/GRY","WHT/BRN",
    "YEL/GRN","YEL/WHT","YEL/RED","GRN/YEL","PINK/WHT",
    "LT-BLU","LT-GRN","DK-BLU","DK-GRN","D-BLU","D-GRN",
    "BEIGE","SLATE","DARK","PINK",
    "BLK","RED","WHT","BLU","GRN","YEL","ORG","BRN","GRY","PUR","PNK","VIO",
]
COLOR_PAT = "|".join(re.escape(c) for c in COLOR_TOKENS)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
def tokenize(text):
    text = str(text).upper().strip()
    text = (text.replace("\uff0c",",").replace("\uff08","(")
                .replace("\uff09",")").replace("\u3000"," "))
    tokens = re.split(r"[,\s#/()\-]+", text)
    return [t for t in tokens if t]

# ── BM25 feature extraction ───────────────────────────────────────────────────
def extract_bm25_features(descriptions, bm25_model, n=TOP_N):
    """
    For each description, get its BM25 scores against all Good documents.
    Use the top-N scores as the feature vector for Random Forest.
    This replaces TF-IDF character n-grams with BM25 token scores.
    """
    features = []
    for i, desc in enumerate(descriptions):
        query = tokenize(desc)
        scores = bm25_model.get_scores(query)
        top_scores = np.sort(scores)[::-1][:n]
        if len(top_scores) < n:
            top_scores = np.pad(top_scores, (0, n - len(top_scores)))
        features.append(top_scores)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(descriptions)}...", end="\r")
    return np.array(features)

# ── Hand-crafted binary features ─────────────────────────────────────────────
def hand_crafted_features(descriptions):
    features = []
    for d in descriptions:
        d = str(d)
        features.append([
            int(bool(re.match(r"^CBL\.WIRE", d))),
            int(bool(re.search(r"WIRE:", d))),
            int(",," in d),
            int(bool(re.search(r"\)(" + COLOR_PAT + r")([,\s]|$)", d))),
            int(bool(re.match(r"^\s+", d))),
            int(bool(re.search(r"\s+,", d))),
            int(bool(re.search(r"[^\x00-\x7F]", d))),
            d.count(","),
            len(d),
        ])
    return np.array(features, dtype=float)

# ── Error injection ───────────────────────────────────────────────────────────
def inject_error(desc, error_type):
    d = desc.strip()
    if error_type == "clean":
        return d
    elif error_type == "dot_separator":
        return re.sub(r"^CBL,WIRE", "CBL.WIRE", d)
    elif error_type == "colon_separator":
        return re.sub(r"WIRE,", "WIRE:", d, count=1)
    elif error_type == "double_comma":
        idx = d.find(",")
        return d[:idx] + ",," + d[idx+1:] if idx != -1 else d
    elif error_type == "missing_comma_before_color":
        return re.sub(r",(" + COLOR_PAT + r")([,\s]|$)",
                      lambda m: m.group(1) + m.group(2), d)
    elif error_type == "leading_space":
        return "  " + d
    elif error_type == "space_before_comma":
        idx = d.find(",")
        return d[:idx] + " ," + d[idx+1:] if idx != -1 else d
    return d

def generate_synthetic_data(good_descs):
    error_types = [
        "clean", "dot_separator", "colon_separator", "double_comma",
        "missing_comma_before_color", "leading_space", "space_before_comma",
    ]
    descriptions, labels = [], []
    for desc in good_descs:
        for et in error_types:
            if et == "missing_comma_before_color":
                if not re.search(r",(" + COLOR_PAT + r")[,\s]", desc):
                    continue
            descriptions.append(inject_error(desc, et))
            labels.append(et)
    return descriptions, labels

# ── Correction ────────────────────────────────────────────────────────────────
def apply_correction(desc, predicted_error):
    d = str(desc).strip()
    original = d
    if predicted_error == "clean":
        return d, "none"
    elif predicted_error == "dot_separator":
        d = re.sub(r"^CBL\.WIRE", "CBL,WIRE", d)
    elif predicted_error == "colon_separator":
        d = re.sub(r"WIRE:", "WIRE,", d)
    elif predicted_error == "double_comma":
        d = re.sub(r",{2,}", ",", d)
    elif predicted_error == "missing_comma_before_color":
        d = re.sub(r"\)(" + COLOR_PAT + r")([,\s]|$)",
                   lambda m: ")," + m.group(1) + m.group(2), d)
    elif predicted_error == "leading_space":
        d = d.lstrip()
    elif predicted_error == "space_before_comma":
        d = re.sub(r"\s+,", ",", d)
    return d, (predicted_error if d != original else "none")

# ── Main ──────────────────────────────────────────────────────────────────────
def main(good_path, nogood_path, output_path):

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading data...")
    good_df   = pd.read_excel(good_path,   sheet_name="Raw")
    nogood_df = pd.read_excel(nogood_path, sheet_name="Raw")
    good_df.columns   = good_df.columns.str.strip()
    nogood_df.columns = nogood_df.columns.str.strip()

    good_descs   = good_df["Full Item Description"].astype(str).tolist()
    nogood_descs = nogood_df["Full Item Description"].astype(str).tolist()
    print(f"  Good   : {len(good_descs)}")
    print(f"  No Good: {len(nogood_descs)}")

    # ── Build BM25 index ──────────────────────────────────────────────────────
    print("\nBuilding BM25 index on Good descriptions...")
    tokenized_good = [tokenize(d) for d in good_descs]
    bm25 = BM25Okapi(tokenized_good)
    print(f"  Index built with {len(tokenized_good)} documents")

    # ── Synthetic training data ───────────────────────────────────────────────
    print("\nGenerating synthetic training data...")
    all_descs, all_labels = generate_synthetic_data(good_descs)
    print(f"  Total: {len(all_descs)} examples")

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        all_descs, all_labels,
        test_size=VALIDATION_SIZE,
        random_state=42,
        stratify=all_labels
    )
    print(f"  Train: {len(X_train_raw)}  |  Validation: {len(X_val_raw)}")

    # ── Extract BM25 features ─────────────────────────────────────────────────
    print("\nExtracting BM25 features for training set...")
    X_bm25_train = extract_bm25_features(X_train_raw, bm25)
    print("Extracting BM25 features for validation set...")
    X_bm25_val   = extract_bm25_features(X_val_raw,   bm25)

    X_train = np.hstack([X_bm25_train, hand_crafted_features(X_train_raw)])
    X_val   = np.hstack([X_bm25_val,   hand_crafted_features(X_val_raw)])

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    print(f"\n  Feature matrix: {X_train.shape}")
    print(f"  Classes: {list(le.classes_)}")

    # ── Train Random Forest ───────────────────────────────────────────────────
    print("\nTraining Random Forest (200 trees)...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        min_samples_leaf=2, n_jobs=-1, random_state=42,
    )
    clf.fit(X_train, y_train_enc)

    train_acc = clf.score(X_train, y_train_enc)
    val_acc   = clf.score(X_val,   y_val_enc)
    gap       = train_acc - val_acc

    print(f"\n  Training accuracy  : {train_acc*100:.2f}%")
    print(f"  Validation accuracy: {val_acc*100:.2f}%")
    print(f"  Overfit gap        : {gap*100:.2f}%  {'(acceptable)' if gap < 0.05 else '(WARNING)'}")

    y_val_pred = clf.predict(X_val)
    print("\n  Per-class report:")
    print(classification_report(y_val_enc, y_val_pred,
                                target_names=le.classes_, digits=3))

    # ── Predict on No_Good ────────────────────────────────────────────────────
    print("Predicting on No_Good descriptions...")
    X_bm25_test = extract_bm25_features(nogood_descs, bm25)
    X_test      = np.hstack([X_bm25_test, hand_crafted_features(nogood_descs)])

    predicted_labels  = le.inverse_transform(clf.predict(X_test))
    confidence_scores = clf.predict_proba(X_test).max(axis=1)

    good_set = set(d.strip() for d in good_descs)
    results, counts = [], {"AUTO": 0, "MANUAL REVIEW NEEDED": 0}

    for i, original in enumerate(nogood_descs):
        part_no    = nogood_df["Part Number"].iloc[i]
        pred_error = predicted_labels[i]
        confidence = float(confidence_scores[i])

        if original.strip() in good_set:
            final, correction, status = original.strip(), "none", "AUTO"
            confidence, pred_error = 1.0, "clean"
        elif confidence >= CONFIDENCE_THRESHOLD:
            final, correction = apply_correction(original, pred_error)
            status = "AUTO"
        else:
            final, correction, status = "", "", "MANUAL REVIEW NEEDED"

        counts[status] += 1
        results.append({
            "No_Good Part#"           : part_no,
            "Original Description"    : original,
            "Standardized Description": final,
            "Status"                  : status,
            "Predicted Error Type"    : pred_error,
            "Confidence"              : round(confidence, 4),
            "Correction Applied"      : correction,
        })

    # ── Write Excel ───────────────────────────────────────────────────────────
    print("\nWriting Excel...")
    out_df = pd.DataFrame(results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Results")
        val_report_dict = classification_report(
            y_val_enc, y_val_pred, target_names=le.classes_,
            digits=3, output_dict=True
        )
        metrics_rows = []
        for label, metrics in val_report_dict.items():
            if isinstance(metrics, dict):
                metrics_rows.append({
                    "Class": label,
                    "Precision": round(metrics["precision"], 4),
                    "Recall":    round(metrics["recall"], 4),
                    "F1-Score":  round(metrics["f1-score"], 4),
                    "Support":   int(metrics["support"]),
                })
        pd.DataFrame(metrics_rows).to_excel(
            writer, index=False, sheet_name="Validation Metrics")

    wb = load_workbook(output_path)
    for sn in ["Results", "Validation Metrics"]:
        ws = wb[sn]
        hf = PatternFill("solid", fgColor="1F4E79")
        for cell in ws[1]:
            cell.fill = hf
            cell.font = Font(bold=True, color="FFFFFF", size=11)
            cell.alignment = Alignment(horizontal="center", vertical="center")

    ws = wb["Results"]
    green = PatternFill("solid", fgColor="E2EFDA")
    red   = PatternFill("solid", fgColor="F4CCCC")
    for row in ws.iter_rows(min_row=2):
        fill = green if row[3].value == "AUTO" else red
        for cell in row: cell.fill = fill
    for idx, w in enumerate([15, 65, 65, 22, 26, 12, 26], 1):
        ws.column_dimensions[get_column_letter(idx)].width = w
    ws.freeze_panes = "A2"
    wb.save(output_path)

    total  = len(out_df)
    auto   = counts["AUTO"]
    manual = counts["MANUAL REVIEW NEEDED"]
    print(f"\n{'='*50}")
    print(f"  Total processed      : {total}")
    print(f"  Auto standardized    : {auto}  ({auto/total*100:.1f}%)")
    print(f"  Manual review        : {manual}  ({manual/total*100:.1f}%)")
    print(f"\n  Model (BM25 + Random Forest):")
    print(f"    Training accuracy  : {train_acc*100:.2f}%")
    print(f"    Validation accuracy: {val_acc*100:.2f}%")
    print(f"\n  Output: {output_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    good_path   = sys.argv[1] if len(sys.argv) > 1 else "data/1.Good.xlsx"
    nogood_path = sys.argv[2] if len(sys.argv) > 2 else "data/2.No_Good.xlsx"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "outputs/BM25_RF_Standardized_Output.xlsx"
    main(good_path, nogood_path, output_path)