"""
Prediction results of a trained model on each protected group of the test split.

Works with any binary classifier trained on is_approved: sklearn and XGBoost models, Keras models, or anything with
a predict method. Pass the X_test the model was trained on (X_test_prep.csv or X_test_prep_with_protected.csv).

From a notebook in notebooks/:
  import sys; sys.path.append('../scripts')
  from fairness_eval import results_by_group

Example, from the project root:
  X_test = pd.read_csv('data/features/X_test_prep.csv', index_col='row_id')
  y_test = pd.read_csv('data/features/y_test.csv', index_col='row_id')['is_approved']
  protected = pd.read_csv('data/features/protected_test.csv', index_col='row_id')
  results_by_group(model, X_test, y_test, protected['applicant_race_1'])
  results_by_group(model, X_test, y_test, protected[['applicant_race_1', 'applicant_sex']])  # race x sex
"""

############## Imports ###############
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score

############## Settings ###############
# HMDA codes -> labels, checked against the *_name columns of data/ny_hmda_2015.csv
RACE_LABELS = {1: 'American Indian or Alaska Native', 2: 'Asian', 3: 'Black or African American',
               4: 'Native Hawaiian or Other Pacific Islander', 5: 'White', 6: 'Information not provided',
               7: 'Not applicable', 8: 'No co-applicant'}
ETHNICITY_LABELS = {1: 'Hispanic or Latino', 2: 'Not Hispanic or Latino', 3: 'Information not provided',
                    4: 'Not applicable', 5: 'No co-applicant'}
SEX_LABELS = {1: 'Male', 2: 'Female', 3: 'Information not provided', 4: 'Not applicable', 5: 'No co-applicant'}
HMDA_LABELS = {'applicant_race_1': RACE_LABELS, 'co_applicant_race_1': RACE_LABELS,
               'applicant_ethnicity': ETHNICITY_LABELS, 'co_applicant_ethnicity': ETHNICITY_LABELS,
               'applicant_sex': SEX_LABELS, 'co_applicant_sex': SEX_LABELS}

############## Handlers ###############
def results_by_group(model, X_test: pd.DataFrame, y_test: pd.Series, protected,
                     threshold: float = 0.5) -> pd.DataFrame:
  """One row per protected group, plus an 'overall' row first, with the model's prediction results on it.

  protected is a column of data/features/protected_test.csv, or several columns for intersections, with the same
  index as X_test. HMDA codes are shown as labels. Approved (1) is the positive class.
  """
  if not (protected.index.equals(X_test.index) and y_test.index.equals(X_test.index)):
    raise ValueError('X_test, y_test and protected must have the same index (row_id)')

  groups = protected.to_frame() if isinstance(protected, pd.Series) else protected
  groups = groups.astype(object).replace({col: HMDA_LABELS[col] for col in groups.columns if col in HMDA_LABELS})

  scores = _approval_probability(model, X_test)
  rows = pd.DataFrame({'actual': y_test.to_numpy() == 1, 'predicted': scores >= threshold, 'score': scores})

  records = [{**{col: 'overall' for col in groups.columns}, **_summarize(rows)}]
  for key, part in rows.groupby([groups[col].to_numpy() for col in groups.columns], dropna=False):
    records.append({**dict(zip(groups.columns, key)), **_summarize(part)})
  table = pd.DataFrame(records)
  return pd.concat([table.iloc[:1], table.iloc[1:].sort_values('n', ascending=False)], ignore_index=True)

def _approval_probability(model, X: pd.DataFrame) -> np.ndarray:
  """Probability of approval for each row. A model without probabilities gives its 0/1 predictions instead."""
  if hasattr(model, 'feature_names_in_'):
    X = X[list(model.feature_names_in_)]  # e.g. a model fit on the committed 97 columns, given the 99-column file
  if hasattr(model, 'predict_proba'):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]
  if 'keras' in type(model).__module__:
    return model.predict(X.to_numpy(dtype='float32'), verbose=0).ravel()  # one sigmoid output
  return np.asarray(model.predict(X), dtype=float).ravel()  # AUC is then computed on 0/1 predictions

def _summarize(rows: pd.DataFrame) -> dict:
  actual, predicted = rows['actual'], rows['predicted']
  return {
    'n': len(rows),
    'n_denied': int((~actual).sum()),
    'approval_rate': actual.mean(),
    'predicted_approval_rate': predicted.mean(),
    'accuracy': (actual == predicted).mean(),
    'tpr': predicted[actual].mean(),    # share of approved applicants predicted approved
    'fpr': predicted[~actual].mean(),   # share of denied applicants predicted approved
    'f1_denied': f1_score(~actual, ~predicted, zero_division=0),
    'auc': roc_auc_score(actual, rows['score']) if actual.nunique() == 2 else np.nan,
  }
