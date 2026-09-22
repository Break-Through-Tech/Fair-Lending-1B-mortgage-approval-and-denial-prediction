"""
Build model-ready train/test features from the deduplicated, winsorized HMDA data.
Script version of notebooks/Feature_Engineering.ipynb, which explains each step.
"""

############## Imports #################
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

############## Data Files #############
INPUT_PATH = os.path.join(os.getcwd(), "data", "ny_hmda_2015_dedup_outliers.csv")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "features")

############## Settings ###############
USE_PROTECTED_AS_FEATURES = True
RANDOM_STATE = 42
TARGET = 'is_approved'

############## Column Groups ##########
LEAKAGE_COLS = ['denial_reason_1', 'denial_reason_2', 'denial_reason_3',
                'rate_spread', 'purchaser_type', 'action_taken']
CONSTANT_COLS = ['as_of_year', 'state_code', 'application_date_indicator']
MOSTLY_EMPTY_COLS = (['edit_status']
                     + [f'applicant_race_{i}' for i in range(2, 6)]
                     + [f'co_applicant_race_{i}' for i in range(2, 6)])
ID_LIKE_COLS = ['sequence_number', 'respondent_id', 'census_tract_number']
REDUNDANT_COLS = ['county_code']

LOG_SOURCE_COLS = ['applicant_income_000s', 'loan_amount_000s', 'loan_to_income']
LOG_COLS = [f'log_{c}' for c in LOG_SOURCE_COLS]

CATEGORICAL_COLS = ['agency_code', 'loan_type', 'property_type', 'loan_purpose',
                    'owner_occupancy', 'preapproval', 'lien_status', 'hoepa_status', 'msamd']
PROTECTED_COLS = ['applicant_sex', 'applicant_ethnicity', 'applicant_race_1',
                  'co_applicant_sex', 'co_applicant_ethnicity', 'co_applicant_race_1']
NUMERIC_COLS = LOG_COLS + ['hud_median_family_income', 'tract_to_msamd_income',
                           'minority_population', 'population',
                           'number_of_owner_occupied_units', 'number_of_1_to_4_family_units']
BINARY_COLS = ['income_missing', 'has_co_applicant']

############## Handlers ###############
def build_target(df: pd.DataFrame) -> pd.DataFrame:
  """Approved (1): action_taken 1, 2. Denied (0): 3, 7. Withdrawn (4), incomplete (5) and purchased (6) are removed."""
  df = df[~df['action_taken'].isin([4, 5, 6])].copy()
  df[TARGET] = df['action_taken'].isin([1, 2]).astype(int)
  return df.drop(columns=['action_taken'])

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
  text_cols = [c for c in df.columns if '_name' in c or c.endswith('_abbr')]
  drop_groups = {
    'leakage': LEAKAGE_COLS,
    'constant': CONSTANT_COLS,
    'mostly empty': MOSTLY_EMPTY_COLS,
    'ID-like': ID_LIKE_COLS,
    'redundant': REDUNDANT_COLS,
    'text labels': text_cols,
  }
  for group, cols in drop_groups.items():
    present = [c for c in cols if c in df.columns]
    print(f'Dropping {group} ({len(present)}): {present}')
    df = df.drop(columns=present)
  print('Shape after drops:', df.shape)
  return df

def add_row_features(frame: pd.DataFrame) -> pd.DataFrame:
  out = frame.copy()
  income = out['applicant_income_000s']
  out['income_missing'] = income.isna().astype(int)
  out['has_co_applicant'] = (out['co_applicant_sex'] != 5).astype(int)
  out['loan_to_income'] = out['loan_amount_000s'] / income.replace(0, np.nan)
  for src, dst in zip(LOG_SOURCE_COLS, LOG_COLS):
    out[dst] = np.log1p(out[src])
  return out

class FeaturePreprocessor:
  """Impute, one-hot encode and scale the model inputs, learning every step from the training split."""

  def __init__(self, cat_cols: list, numeric_cols: list, binary_cols: list) -> None:
    self.cat_cols = cat_cols
    self.numeric_cols = numeric_cols
    self.binary_cols = binary_cols
    self.cat_imputer = SimpleImputer(strategy='constant', fill_value=-1)
    self.num_imputer = SimpleImputer(strategy='median')
    self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int)
    self.scaler = StandardScaler()

  def fit(self, X: pd.DataFrame):
    self.cat_imputer.fit(X[self.cat_cols])
    self.num_imputer.fit(X[self.numeric_cols])
    self.encoder.fit(self._impute_cat(X))
    self.scaler.fit(self._impute_num(X))
    return self

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    cat = pd.DataFrame(self.encoder.transform(self._impute_cat(X)),
                       columns=self.encoder.get_feature_names_out(), index=X.index)
    num = pd.DataFrame(self.scaler.transform(self._impute_num(X)),
                       columns=self.numeric_cols, index=X.index)
    return pd.concat([cat, num, X[self.binary_cols]], axis=1)

  def _impute_cat(self, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(self.cat_imputer.transform(X[self.cat_cols]),
                        columns=self.cat_cols, index=X.index).astype(int)

  def _impute_num(self, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(self.num_imputer.transform(X[self.numeric_cols]),
                        columns=self.numeric_cols, index=X.index)

############## Entry Point #############
if __name__ == "__main__":
  df = pd.read_csv(INPUT_PATH, low_memory=False)
  print('Input shape:', df.shape)

  df = build_target(df)
  print('Shape with target:', df.shape, f'approval rate {df[TARGET].mean():.3f}')

  df = drop_unused_columns(df)
  df = add_row_features(df)

  X = df.drop(columns=[TARGET])
  y = df[TARGET]
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
  print('Train:', X_train.shape, f'approval rate {y_train.mean():.3f}')
  print('Test: ', X_test.shape, f'approval rate {y_test.mean():.3f}')

  cat_cols = CATEGORICAL_COLS + (PROTECTED_COLS if USE_PROTECTED_AS_FEATURES else [])
  missing = X_train[cat_cols + NUMERIC_COLS].isna().sum()
  print('Missing values in the training split:')
  print(missing[missing > 0].to_string())

  preprocessor = FeaturePreprocessor(cat_cols, NUMERIC_COLS, BINARY_COLS).fit(X_train)
  X_train_prep = preprocessor.transform(X_train)
  X_test_prep = preprocessor.transform(X_test)

  assert not X_train_prep.isna().any().any(), 'NaNs left in X_train_prep'
  assert not X_test_prep.isna().any().any(), 'NaNs left in X_test_prep'
  assert list(X_train_prep.columns) == list(X_test_prep.columns)
  print('X_train_prep:', X_train_prep.shape)
  print('X_test_prep: ', X_test_prep.shape)

  # row_id is the row's position in the input CSV, so every file lines up with the others
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  outputs = {
    'X_train_prep': X_train_prep,
    'X_test_prep': X_test_prep,
    'y_train': y_train,
    'y_test': y_test,
    'protected_train': X_train[PROTECTED_COLS],
    'protected_test': X_test[PROTECTED_COLS],
  }
  for name, data in outputs.items():
    data.to_csv(os.path.join(OUTPUT_DIR, f'{name}.csv'), index_label='row_id')

  # Saved as a plain dict of the fitted sklearn objects and column lists, so loading it doesn't need this script
  joblib.dump(vars(preprocessor), os.path.join(OUTPUT_DIR, 'preprocessors.joblib'))
  print(f'Saved {", ".join(outputs)} and preprocessors → {OUTPUT_DIR}')
