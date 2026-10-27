# Mortgage Approval & Denial Prediction - Fair Lending 1B
---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
|  David Balogun   | @BALOGUN-DAVID | Challenge Advisor            |
|  Hrushikesh Shetty | @HRUSHIKESH070902 | Coach
|  Michelle Kelly  | @MICHELLE-KELLY    |  Break Through Tech Fellow |
|  Akhar Min Hein  | @AHKARMIN  | Break Through Tech Fellow                 |
|  Rudra Patel     | @RUDRA1729   | Break Through Tech Fellow  |
|  Thang Nguyen    | @Quixowo    |  Break Through Tech Fellow         |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

1. Clone the repository and move into it.
2. Install dependencies: `pip install -r requirements.txt` (the notebooks also need `jupyter`).
3. Download the NY HMDA 2015 data from [Kaggle](https://www.kaggle.com/datasets/jboysen/ny-home-mortgage) and save it as `data/ny_hmda_2015.csv`. The `data/` folder is git-ignored, so every generated file stays local.
4. From the project root, run the whole pipeline: `python main.py`. It runs, in order:
   - `scripts/duplicate_handler.py`: drops exact duplicate rows and the redundant `*_name` columns
   - `scripts/constant_column_handler.py`: writes a copy without constant columns (not used by later steps)
   - `scripts/feature_engineering.py`: builds the target, applies the scope filters, creates the train/test features
   - `scripts/logistic_regression_trainer.py`: trains and saves the logistic regression model
   - `scripts/decision_tree_trainer.py`: trains and saves the decision tree model, then prints its test metrics
5. Notebooks in `notebooks/` explain each stage: `Data Understanding`, `Duplicates and Outliers` (read-only analysis), `Feature_Engineering` (a step-by-step version of the script, which checks itself against the script's saved output) and `Model_Analysis`.

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**Dataset.** Home Mortgage Disclosure Act (HMDA) loan applications for New York in 2015: 439,654 rows and 78 columns, one row per application. Most categorical fields appear twice, as a numeric code and a text `*_name` column. The data has no credit score, debt-to-income ratio or loan-to-value, which limits what any model can explain.

**Cleaning and preprocessing decisions** (all in `scripts/feature_engineering.py`):

| Decision | What we do | Why |
|---|---|---|
| Target | `is_approved` = 1 for originated or approved-not-accepted, 0 for denied or preapproval denied. Withdrawn, incomplete and purchased loans are removed. | Only the first four are lending decisions by the reporting institution |
| Scope | Keep one-to-four family homes (`property_type` 1, 2) and remove applicants with race code 7 ("not applicable") | Multifamily loans (4,677 rows) have no income and different underwriting. Race 7 (2,833 more rows) marks applicants who aren't natural persons, such as companies. We study individual borrowers. |
| Leakage | Drop `denial_reason_*`, `rate_spread`, `purchaser_type`, `hoepa_status` | Only known after the decision; `hoepa_status` is only ever set on loans that were made |
| Geography | Use `county_code` (one-hot) instead of `msamd`; drop `hud_median_family_income` and the tract number; keep `tract_to_msamd_income` | Counties sit inside metro areas, and the HUD income takes one value per metro area, so it only repeated the geography column and made the model's coefficients unstable |
| Lender | `respondent_id` is target-encoded (smoothed approval rate, computed out-of-fold on the training split) plus a frequency column | 800+ lenders with very different approval rates; one-hot would be mostly empty |
| Outliers | Income, loan amount and loan-to-income are capped at the training 1st / 99th percentile, then logged | Top-coded values like 9999 and 99999; cutoffs come from the training split only so nothing leaks |
| Missing income | Median-imputed with an `income_missing` flag | About 5% of rows |
| Race | Each race gets its own 0/1 column (applicant and co-applicant), so people who list several races are counted in each, plus "not provided" and "2+ races" flags | Only the first of five race codes was being used. About 0.4% of applicants list more than one race. |

The final split is 80/20 and stratified: 251,540 training rows and 62,885 test rows, 75% approved. Every fitted step (caps, encoders, imputers, scaler) learns from the training rows only.

**Protected attributes.** Race, sex and ethnicity are kept out of the default model's inputs and saved in `data/features/protected_*.csv` for the fairness audit. The script also writes a second feature set that includes them, so a model trained with them can be compared against one trained without. `minority_population` is still a model input and may act as a proxy for race; we plan to check this.

---

## 🧠 **Model Development**

**Baseline model.** Logistic regression (`scripts/logistic_regression_trainer.py`) with `class_weight='balanced'` and a 5-fold grid search over `C` (0.1 to 100, L2, `lbfgs`), scored on F1. The input is `data/features/X_train_prep.csv`, with no protected attributes. Decision tree and gradient boosting models are still to do (`scripts/decision_tree_trainer.py` is an empty placeholder).

---

## 📈 **Results & Key Findings**

Current baseline, evaluated on the held-out test split (`notebooks/Model_Analysis.ipynb`; the grid search selected `C=1`):

| Metric | Value |
|---|---|
| AUC-ROC | 0.792 |
| F1 (approved class) | 0.809 |
| Accuracy | 0.734 |

AUC-ROC is just under the 0.80 success target. A quick check with an untuned `C=1` model trained with protected attributes gave about 0.795 AUC, a gain of only about 0.003, but that comparison and the fairness metrics (demographic parity, equalized odds) have not been done properly yet.

---

## 🚀 **Next Steps**

- Run the bias audit on the protected files, including the multi-race groups and the "not provided" groups, and decide how to treat race code 6.
- Compare the models trained with and without protected attributes, and test whether race can be predicted from the remaining features (proxy check, for example `minority_population`).
- Train decision tree and gradient boosting models.
- Tune for AUC-ROC (the success metric) instead of F1, and widen the search beyond the current grid.
- Clean up the remaining pipeline pieces: the unused constant-column output and the blanket warning suppression in the scripts.

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
