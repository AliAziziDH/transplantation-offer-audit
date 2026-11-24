# Leakage & Benchmark Audit: Transplantation Offer Prediction

## 1. Executive Summary

The headline claim of **99.3% accuracy** for predicting `"Hospital with Offers"` with a `max_depth=5` Decision Tree is **not defensible**.

1. **The original model script (`Transplantation_Model_01.py`) never computed accuracy** — it pointed to a hardcoded Windows path (`test.xlsx` not present in the repository), lacked categorical encoding, and only printed `export_text(...)`.
2. **Every path toward ~99.3% on the private repository data collapses under scrutiny:**
   - **Target-target alias leakage (`Entire Name` ↔ `Hospital`):** 100% 1-to-1 duplicate of the target column. Even **with** this direct leak, `max_depth=5` achieves only **88.72%** (`ordinal`) or **71.05%** (`one-hot`) on `ki_offers.csv`; reaching **99.27% test / 99.30% train** requires increasing depth to `max_depth=10` purely to memorize the hospital name-to-code lookup table.
   - **Base-rate illusion on `Offer Accepted`:** When predicting binary `Offer Accepted` (`99.4030%` `No` vs. `0.5970%` `Yes`), a `max_depth=5` tree on 35 clean pre-decision features scores **99.3497%** raw test accuracy — **worse than always predicting `No` (99.3705%)**, with `Yes` recall of only **17.22%** (`0.00%` on numeric-only features).
   - **Post-decision leakage:** `Primary Refusal Code` and `Prime Descrip` are populated on `59,612 / 59,612` refused offers and `NaN` on `358 / 358` accepted offers, giving **100.00%** accuracy in 2 leaves.
   - **Donor-level train/test leakage:** Random 60/40 row splitting puts **218 / 242 donors (90.1%)** in both train and test. Removing `Entire Name` and switching to donor-disjoint `GroupShuffleSplit` drops `Hospital` accuracy from **61.36%** to **21.11%** (and to **28.90%** on clean pre-offer clinical features, below the **30.85%** majority baseline).
   - **Structural non-identifiability on `"Hospital with Offers"` (`KI Outcome`):** 65 of 67 non-ID columns are constant per donor, yielding only `1,003` unique feature rows out of `8,348` and an **exact Bayes-optimal theoretical ceiling of `12.01%`** (`1,003 / 8,348`).

---

## 2. Defects in the Original Script (`Transplantation_Model_01.py`)

Inspection of the original script in the private baseline repository established four structural defects:

- **Hardcoded external file path:** Loaded a local Windows OneDrive Excel path (`test.xlsx`) absent from version control.
- **No preprocessing / categorical handling:** Passed raw string columns directly to `DecisionTreeClassifier(max_depth=5, random_state=42)` (which fails in `scikit-learn` without encoding).
- **Row-level random split across repeated donors:** Used `train_test_split(x, y, test_size=0.4, random_state=42)` without grouping by donor ID.
- **No evaluation metrics computed:** Only printed `export_text(clf, ...)` without `predict`, `accuracy_score`, confusion matrix, balanced accuracy, or per-class F1.

---

## 3. Private Data Audited Offline (Not Committed to Git)

All real-data measurements below were computed offline on the two private UNOS/OPTN CSV extracts (excluded from this repository via `.gitignore`):

| Dataset (Excluded from Git) | Rows ($N$) | Columns | Unique Donors | Target Column | Classes | Majority Class Baseline (Full) |
|---|---:|---:|---:|---|---:|---:|
| `ki_offers.csv` | 59,970 | 102 | 242 (avg 247.81 rows/donor) | `Hospital` *(equivalent of `Hospital with Offers`)* | 33 | `TXHS-TX1`: 17,785 (**29.66%**) |
| `ki_offers.csv` | 59,970 | 102 | 242 | `Offer Accepted` *(binary offer outcome)* | 2 | `No`: 59,612 (**99.40%**) / `Yes`: 358 (**0.60%**) |
| `KI Outcome Download_data 10.1.20-9.30.23.csv` | 8,348 | 68 | 992 (avg 8.42 rows/donor) | `Hospital with Offers` *(literal column name)* | 35 | `TXHI-TX1`: 586 (**7.02%**) |

Class imbalance is severe across both files: in `ki_offers.csv`, the tail classes `TXCF-TX1` (27), `TXUC-TX1` (23), and `TXDC-TX1` (19) represent `< 0.05%` each; in `KI Outcome`, `TXDL-TX1` (23), `TXLP-TX1` (13), and `MSUM-TX1` (10) represent `0.12%–0.28%` each.

---

## 4. Catalogue of Data Leaks & Structural Artifacts

### 4.1 Direct Target Alias (`Entire Name` ↔ `Hospital`)
In `ki_offers.csv` ($N = 59{,}970$), `Entire Name` is the full hospital name corresponding 1-to-1 with `Hospital`:
- `nunique(Hospital) = 33`, `nunique(Entire Name) = 33`, unique `(Hospital, Entire Name)` pairs = **33**.
- Single-column Bayes purity = **100.0000%**.
- A `max_depth=5` tree with ordinal encoding allocates 15 leaves (`Entire Name` feature importance = **0.9959**) and gets **88.72%** test accuracy; `max_depth=5` is structurally capped at $\le 32$ leaves, so it cannot separate all 33 hospitals. Raising depth to `max_depth=10` (30 leaves) gives **99.27% test / 99.30% train**, and `max_depth=None` (33 leaves) gives **100.00%** — purely by memorizing the name-to-code bijection.

### 4.2 Post-Decision & Batch-Timestamp Columns
- **`Offer Accepted` post-decision leaks (`ki_offers.csv`):**
  - `Primary Refusal Code` and `Prime Descrip` are **non-null on `59,612 / 59,612` `No` rows and `NaN` on `358 / 358` `Yes` rows**. A depth-1 split on `Prime Descrip` (`2` leaves) achieves **100.0000%** test accuracy.
  - `Match Sequence` appearing in `Accept Sequence(s)` is `False` for `59,612 / 59,612` `No` rows and `True` for `354 / 358` `Yes` rows.
- **`Hospital` post-offer / batch-timestamp leaks (`ki_offers.csv`):**
  - `Unnamed: 0` is the row index `0..59,969` (sorted by donor/hospital/sequence; **100.00%** single-column purity).
  - `Final Response Dt` has **98.9161%** single-column Bayes purity across 2,705 unique timestamps; `initial_response_dt` has **86.3365%** across 1,453 unique timestamps.
  - `Allocation Dist` and `Lc Dist` are **100% identical** (`68.7110%` single-column purity), and `(Donor ID, Allocation Dist)` uniquely identifies a single `Hospital` in **77.4510%** of pairs.
  - Post-transplant recipient columns (`Recipient One/Two ...`, `Tx1 Dist`, `Tx2 Dist`) encode donor-level outcomes recorded after acceptance.

### 4.3 Donor-Level Train/Test Contamination
Because each donor has an average of **247.81** rows in `ki_offers.csv` (and **8.42** rows in `KI Outcome`), ungrouped `train_test_split(..., test_size=0.4, random_state=42)` leaks donor identities and donor-specific distance/outcome profiles across splits:
- `ki_offers.csv` random split ($N_{\text{train}} = 35{,}982$, $N_{\text{test}} = 23{,}988$): **218 / 242 donors (90.08%)** appear in both train and test.
- `KI Outcome` random split ($N_{\text{train}} = 5{,}008$, $N_{\text{test}} = 3{,}340$): **823 / 992 donors (82.96%)** appear in both train and test.
- Removing `Entire Name` on `ki_offers.csv` (99 remaining columns) yields **61.36%** under random row split, but collapses to **21.11%** under donor-disjoint `GroupShuffleSplit` ($N_{\text{train}} = 39{,}222$ / 145 donors; $N_{\text{test}} = 20{,}748$ / 97 donors; 0 shared donors).

### 4.4 Structural Non-Identifiability in `KI Outcome` (`Hospital with Offers`)
In `KI Outcome Download_data 10.1.20-9.30.23.csv` ($N = 8{,}348$, 992 donors), **65 of 67 non-ID columns are 100% invariant within each `Donor Id`**; only `Hospital with Offers` (937 donors) and `Match Date` (11 donors, max 2 dates) vary within a donor. Consequently:
- Only **1,003 unique feature vectors** exist across **8,348 rows** (each donor's features are duplicated across an average of 8.42 hospitals).
- **Exact Bayes-optimal theoretical accuracy ceiling = `1,003 / 8,348 = 12.0149%` (12.01%).** No model of any architecture can exceed 12.01% without memorizing row order or target labels.

---

## 5. Base-Rate Illusion (`Offer Accepted`: 99.35% vs. 99.37% Baseline)

On `Offer Accepted` in `ki_offers.csv` ($N = 59{,}970$):
- Full dataset: `No = 59,612` (**99.4030%**), `Yes = 358` (**0.5970%**).
- Random 40% test split ($N_{\text{test}} = 23{,}988$, $N_{\text{train}} = 35{,}982$): `No = 23,837` (**99.3705%**), `Yes = 151` (**0.6295%**).

Evaluating `DecisionTreeClassifier(max_depth=5, random_state=42)` on **35 clean pre-decision features** (`30` donor/candidate clinical features + `Hospital`, `Allocation Dist`, `Match Classification`, `Match Sequence`, `response_post_clamp`):
- **Random split (ordinal encoding):** Test accuracy = **99.3497%** — **0.02 percentage points worse than predicting `No` for every offer (99.3705%)**, with `Yes` precision = **45.61%**, recall = **17.22%**, F1 = **0.2500** (`57` predicted vs. `151` actual `Yes`).
- **Random split (numeric-only, 11 features):** Test accuracy = **99.3497%**, `Yes` precision = **0.00%**, recall = **0.00%**, F1 = **0.0000** (`5` predicted `Yes`, 0 true positives).
- **Donor-grouped split (`GroupShuffleSplit`, $N_{\text{train}} = 39{,}222$ / 145 donors, $N_{\text{test}} = 20{,}748$ / 97 donors, baseline `99.3156%`):** Test accuracy drops to **94.9007%**, `Yes` precision = **2.09%**, recall = **14.08%**, F1 = **0.0364** (`956` predicted vs. `142` actual `Yes`).

Reporting `99.35%` raw accuracy here is pure **base-rate illusion**.

---

## 6. Defensible Offline Benchmark Summary

All models below use `DecisionTreeClassifier(max_depth=5, random_state=42)` (unless marked as depth diagnostic) with `test_size=0.4, random_state=42`, and preprocessing fit strictly on the training fold.

| Dataset & Target | Split ($N_{\text{train}} / N_{\text{test}}$, Shared Donors) | Feature Set & Encoding | Test Baseline | Train Acc | Test Acc | Balanced Acc | Macro F1 / Minority Metric |
|---|---|---|---:|---:|---:|---:|---|
| `ki_offers` → `Hospital` (33 cls) | Random (35,982 / 23,988, 218/242) | All 100 cols minus `Unnamed: 0` (**with `Entire Name` leak**), ordinal | 29.83% | 88.39% | **88.72%** | 45.45% | Macro F1 = 0.4131 |
| `ki_offers` → `Hospital` (33 cls) | Random (35,982 / 23,988, 218/242) | Same (**with `Entire Name` leak**), one-hot (6,191 cols) | 29.83% | 70.63% | **71.05%** | 18.18% | Macro F1 = 0.1590 |
| `ki_offers` → `Hospital` *(diag `depth=10`)* | Random (35,982 / 23,988, 218/242) | Same (**with `Entire Name` leak**), ordinal | 29.83% | 99.30% | **99.27%** | 90.91% | Macro F1 = 0.8969 (memorizes alias) |
| `ki_offers` → `Hospital` (33 cls) | Random (35,982 / 23,988, 218/242) | 99 cols (no `Entire Name`, leaks remain), ordinal | 29.83% | 61.24% | **61.36%** | 23.42% | Macro F1 = 0.2330 |
| `ki_offers` → `Hospital` (33 cls) | `GroupShuffleSplit(Donor ID)` (39,222 / 20,748, **0**) | 99 cols (no `Entire Name`), ordinal | 30.85% | 60.63% | **21.11%** | 11.79% | Macro F1 = 0.1018 |
| `ki_offers` → `Hospital` (33 cls) | Random (35,982 / 23,988, 218/242) | 30 clean pre-offer clinical cols, ordinal | 29.83% | 33.02% | **32.71%** | 7.55% | Macro F1 = 0.0517 |
| `ki_offers` → `Hospital` (33 cls) | `GroupShuffleSplit(Donor ID)` (39,222 / 20,748, **0**) | 30 clean pre-offer clinical cols, ordinal | 30.85% | 33.17% | **28.90%** | 3.77% | Macro F1 = 0.0259 (below baseline) |
| `KI Outcome` → `Hospital with Offers` (35 cls) | Random (5,008 / 3,340, 823/992) | All 67 cols, ordinal (Bayes cap = **12.01%**) | 7.60% | 10.14% | **6.50%** | 3.91% | Macro F1 = 0.0169 (numeric: 7.07%) |
| `KI Outcome` → `Hospital with Offers` (35 cls) | `GroupShuffleSplit(Donor Id)` (5,102 / 3,246, **0**) | All 67 cols, ordinal (Bayes cap = **12.01%**) | 7.02% | 9.02% | **8.01%** | 4.19% | Macro F1 = 0.0260 |
| `ki_offers` → `Offer Accepted` (2 cls) | Random (35,982 / 23,988, 218/242) | All 100 cols (**with refusal leak**), ordinal | 99.37% | 100.00% | **100.00%** | 100.00% | `Yes` F1 = 1.0000 (2 leaves) |
| `ki_offers` → `Offer Accepted` (2 cls) | Random (35,982 / 23,988, 218/242) | 35 pre-decision cols, ordinal | 99.37% | 99.47% | **99.35%** | 58.54% | `Yes` P=45.6%, R=17.2%, F1=0.2500 |
| `ki_offers` → `Offer Accepted` (2 cls) | `GroupShuffleSplit(Donor ID)` (39,222 / 20,748, **0**) | 35 pre-decision cols, ordinal | 99.32% | 99.53% | **94.90%** | 54.77% | `Yes` P=2.1%, R=14.1%, F1=0.0364 |

---

## 7. Note on Synthetic Sample Data vs. Offline Private Audit Telemetry

Private `.csv`/`.xlsx` medical records are excluded via `.gitignore` and never stored in this repository. Running `python3 src/scripts/eval_transplantation.py` in a fresh clone defaults to the **100% synthetic** datasets in `sample_data/` (`synthetic_ki_offers.csv`, `synthetic_ki_outcomes.csv`) for pipeline verification and prints `[SYNTHETIC SAMPLE DATA]` in its header. Those synthetic rehearsal outputs verify code execution only and do not replace the offline private-data audit measurements reported in Sections 3–6.
