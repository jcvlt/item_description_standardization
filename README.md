# Item Description Standardization
### Machine Learning and Big Data Analytics — Group Project

Automatically standardizes wire and cable item descriptions using multiple ML approaches trained on known-correct reference data.

---

## Problem

Over 30 years of procurement, 5,078 item descriptions accumulated formatting inconsistencies — missing commas, inconsistent punctuation, full-width characters, and non-standard spacing. These make inventory search and comparison unreliable.

**Example:**
```
Before:  CBL,WIRE,MIL-W-16878E/1,24#(19/36TA)BLK,OD1.14,ALPHA#1854/19-2
After:   CBL,WIRE,MIL-W-16878E/1,24#(19/36TA),BLK,OD1.14,ALPHA#1854/19-2
```

---

## Data

| File | Rows | Description |
|------|------|-------------|
| `data/1.Good.xlsx` | 4,511 | Correctly formatted descriptions (training/reference set) |
| `data/2.No_Good.xlsx` | 5,078 | Non-conforming descriptions to be standardized |

**Key discovery:** The Good dataset itself contains ~354 descriptions with formatting errors, which required a post-correction step in the search-based methods.

**Standard description format:**
```
CBL,WIRE,[Standard],[Wire Gauge#(Strands/Diameter+Material)],[Color],[OD],[Vendor Part#]
```

---

## Methods

We implemented and compared 5 approaches across two categories:

### Category 1: Search-based (Nearest Neighbor)
For each No_Good description, find the most similar Good description and use it as the standardized output.

| Script | Method | How similarity is measured |
|--------|--------|--------------------------|
| `ml_standardize.py` | TF-IDF | Character n-gram vectors + cosine similarity |
| `fuzzy_standardize.py` | Fuzzy Matching | Edit distance (difflib, no install needed) |
| `bm25_standardize.py` | BM25 | Token frequency ranking |

All three use a similarity threshold — descriptions below the threshold are left blank and flagged for manual review.

### Category 2: ML Classifier (Error Detection + Correction)
Train a Random Forest classifier to identify the error type in each description, then apply a targeted correction.

| Script | Method | Features used |
|--------|--------|--------------|
| `classifier_standardize.py` | TF-IDF + Random Forest | Character n-gram TF-IDF vectors |
| `bm25_rf_standardize.py` | BM25 + Random Forest | BM25 token scores |

**Training pipeline:**
1. Generate synthetic training data by injecting 7 known error types into Good descriptions (~30,000 examples)
2. 80/20 train/validation split (stratified)
3. Train Random Forest (200 trees)
4. Predict error type for each No_Good description → apply targeted fix

**Error classes:** `clean`, `dot_separator`, `colon_separator`, `double_comma`, `missing_comma_before_color`, `leading_space`, `space_before_comma`

---

## Results

| Method | Auto-standardized | Manual Review | Val. Accuracy | Real ML? |
|--------|:-----------------:|:-------------:|:-------------:|:--------:|
| TF-IDF | 4,519 (89.0%) | 559 (11.0%) | — | Partial |
| Fuzzy Matching | 4,532 (89.2%) | 546 (10.8%) | — | Partial |
| BM25 | 4,759 (93.7%) | 319 (6.3%) | — | Partial |
| TF-IDF + Random Forest | 4,528 (89.2%) | 550 (10.8%) | 97.31% | ✅ Yes |
| **BM25 + Random Forest** | **4,738 (93.3%)** | **340 (6.7%)** | **98.34%** | ✅ **Yes** |

**Best overall:** BM25 + Random Forest — highest AUTO rate among true ML methods and highest validation accuracy.

**Key finding:** BM25 token-level features outperform TF-IDF character n-gram features when used with Random Forest, because tokens like `BLK`, `OD1.3`, `22#` carry more meaning than character fragments like `BL`, `D1.`, `2#`.

---

## Project Outputs

### Output 1 — Before/After Results
All output Excel files contain a `Results` sheet with:
- Original Description (before)
- Standardized Description (after)
- Status: `AUTO` or `MANUAL REVIEW NEEDED`
- Confidence/similarity score
- Correction details

🟢 Green rows = auto-standardized &nbsp;·&nbsp; 🔴 Red rows = manual review needed

### Output 2 — AI-Learned Standard Description Rules
Only `BM25_RF_Standardized_Output.xlsx` includes Output 2, as it is the best-performing model. It contains a second sheet **"AI-Learned Standard Rules"** with:

| Section | Content |
|---------|---------|
| Section 1 | Standard description format — field order, structure, full example |
| Section 2 | Formatting rules — wrong vs. correct examples |
| Section 3 | Valid color codes learned from Good data |
| Section 4 | Top wire standards learned from Good data |
| Section 5 | 7 ML-learned error classes with F1-scores |

---

## Installation

```bash
pip install pandas scikit-learn openpyxl rank-bm25
```

`fuzzy_standardize.py` uses only Python built-ins — no extra packages needed.

---

## Usage

```bash
python ml_standardize.py
python fuzzy_standardize.py
python bm25_standardize.py
python classifier_standardize.py
python bm25_rf_standardize.py
```

Custom paths:
```bash
python <script>.py good.xlsx nogood.xlsx output.xlsx
```

Data files are expected in the `data/` folder and outputs are saved to `outputs/` automatically.

---

## Output File Columns

### Search-based methods (TF-IDF, Fuzzy, BM25)
| Column | Description |
|--------|-------------|
| No_Good Part# | Original part number |
| Original Description | Raw input |
| Standardized Description | Final output (blank if manual review needed) |
| Status | `AUTO` or `MANUAL REVIEW NEEDED` |
| Similarity Score | Match confidence (0–1) |
| Matched Good Part# | Reference entry used |
| Match (reference) | Raw match before post-correction |
| Post-corrections | Formatting fixes applied |

### Classifier methods (TF-IDF + RF, BM25 + RF)
| Column | Description |
|--------|-------------|
| No_Good Part# | Original part number |
| Original Description | Raw input |
| Standardized Description | Final output (blank if manual review needed) |
| Status | `AUTO` or `MANUAL REVIEW NEEDED` |
| Predicted Error Type | Error class predicted by Random Forest |
| Confidence | Classifier confidence (0–1) |
| Correction Applied | Fix that was applied |

---

## Repository Structure

```
project/
├── data/
│   ├── 1.Good.xlsx                          # Training/reference data
│   └── 2.No_Good.xlsx                       # Input data to standardize
├── outputs/
│   ├── ML_Standardized_Output.xlsx          # TF-IDF results
│   ├── Fuzzy_Standardized_Output.xlsx       # Fuzzy Matching results
│   ├── BM25_Standardized_Output.xlsx        # BM25 results
│   ├── Classifier_Standardized_Output.xlsx  # TF-IDF + Random Forest results
│   └── BM25_RF_Standardized_Output.xlsx     # BM25 + Random Forest results (includes Output 2)
├── ml_standardize.py                        # Method 1: TF-IDF
├── fuzzy_standardize.py                     # Method 2: Fuzzy Matching
├── bm25_standardize.py                      # Method 3: BM25
├── classifier_standardize.py                # Method 4: TF-IDF + Random Forest
├── bm25_rf_standardize.py                   # Method 5: BM25 + Random Forest
└── README.md
```