# Item Description Standardization
### Machine Learning and Big Data Analytics Project

Automatically standardizes wire and cable item descriptions using TF-IDF similarity matching trained on known-correct reference data.

---

## Problem

Over 30 years of procurement, 5,078 item descriptions accumulated formatting inconsistencies — missing commas, inconsistent punctuation, full-width characters, and non-standard spacing. These issues make inventory search and comparison unreliable.

**Example:**
```
Before:  CBL,WIRE,MIL-W-16878E/1,24#(19/36TA)BLK,OD1.14,ALPHA#1854/19-2
After:   CBL,WIRE,MIL-W-16878E/1,24#(19/36TA),BLK,OD1.14,ALPHA#1854/19-2
```

---

## Approach

A two-step ML pipeline:

1. **TF-IDF Nearest-Neighbor Matching** — vectorizes 4,511 correct (Good) descriptions using character n-grams, then finds the closest match for each No_Good entry using cosine similarity
2. **Post-Correction Rules** — applies formatting fixes to the matched output to handle errors present even in the Good dataset

Descriptions with similarity score below **0.95** are left blank and flagged for manual review rather than producing a potentially incorrect output.

---

## Results

| Metric | Count | % |
|--------|-------|---|
| Total processed | 5,078 | 100% |
| Auto-standardized (score ≥ 0.95) | 4,519 | 89.0% |
| Flagged for manual review | 559 | 11.0% |

---

## Repository Structure

```
├── ml_standardize.py          # Main ML script
├── 1_Good.xlsx                # Training data (correct descriptions)
├── 2_No_Good.xlsx             # Input data (to be standardized)
├── ML_Standardized_Output.xlsx  # Output results
├── Project_Report.docx        # Full project report
└── README.md
```

---

## Requirements

```
Python 3.8+
pandas
scikit-learn
openpyxl
```

Install dependencies:
```bash
pip install pandas scikit-learn openpyxl
```

---

## Usage

**Default** (uses the files in the repo):
```bash
python ml_standardize.py
```

**Custom paths:**
```bash
python ml_standardize.py good.xlsx nogood.xlsx output.xlsx
```

---

## Output File Columns

| Column | Description |
|--------|-------------|
| No_Good Part# | Original part number |
| Original Description | Raw input |
| Standardized Description | Final output (blank if manual review needed) |
| Status | `AUTO` or `MANUAL REVIEW NEEDED` |
| Similarity Score | Cosine similarity to best Good match (0–1) |
| Matched Good Part# | Reference entry used from Good dataset |
| ML Match (reference) | Raw ML output before post-correction |
| Post-corrections | List of formatting fixes applied |

**Row colors:** 🟢 Green = auto-standardized, 🔴 Red = manual review needed

---

## How It Works

### Step 1: TF-IDF Model

```python
vectorizer = TfidfVectorizer(
    analyzer="char_wb",   # character n-grams
    ngram_range=(2, 4),   # captures patterns like ",BLK" or "OD1."
    max_features=8000,
)
good_matrix = vectorizer.fit_transform(good_descriptions)   # train on Good data
```

For each No_Good entry, cosine similarity is computed against all Good vectors. The highest-scoring match is selected as the standardized reference.

### Step 2: Post-Correction Rules

Fixes applied to ML output:
- Full-width characters (`，（）`) → ASCII
- Missing comma before color: `)BLK` → `),BLK`
- Missing comma before/after OD value
- Extra whitespace removal

### Similarity Threshold

| Score | Action |
|-------|--------|
| ≥ 0.95 | Accept match, apply post-correction, output result |
| < 0.95 | Leave blank, flag as MANUAL REVIEW NEEDED |
