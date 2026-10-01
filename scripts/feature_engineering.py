"""
Build model-ready train/test features from the deduplicated HMDA data.
Script version of notebooks/Feature_Engineering.ipynb, which explains each step.
"""

############## Imports #################
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

############## Data Files #############
INPUT_PATH = os.path.join(os.getcwd(), "data", "ny_hmda_2015_dedup.csv")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "features")

############## Settings ###############
RANDOM_STATE = 42
TARGET = 'is_approved'
WINSOR_LIMITS = (1, 99)   # percentiles, learned from the training split
LENDER_SMOOTHING = 20     # pseudo-applications of overall approval rate blended into each lender's rate
LENDER_FOLDS = 5

############## Scope ##################
# Only one-to-four family homes (property_type 1, 2) and applicants who are people (race 7 = "not applicable", e.g. a company)
KEEP_PROPERTY_TYPES = [1, 2]
NOT_APPLICABLE_RACE = 7

############## Column Groups ##########
LEAKAGE_COLS = ['denial_reason_1', 'denial_reason_2', 'denial_reason_3',
                'rate_spread', 'purchaser_type', 'action_taken',
                'hoepa_status']  # only high-cost loans that were made can be HOEPA, so it leaks the outcome
CONSTANT_COLS = ['as_of_year', 'state_code', 'application_date_indicator',
                 'property_type']  # constant after the scope filter
MOSTLY_EMPTY_COLS = (['edit_status']
                     + [f'applicant_race_{i}' for i in range(2, 6)]
                     + [f'co_applicant_race_{i}' for i in range(2, 6)])
ID_LIKE_COLS = ['sequence_number', 'census_tract_number']
REDUNDANT_COLS = ['msamd',                      # county sits inside an MSA and is finer
                  'hud_median_family_income']   # one value per MSA/county, so it only repeats that column

LENDER = 'respondent_id'
LENDER_COLS = ['lender_approval_rate', 'log_lender_applications']

LOG_SOURCE_COLS = ['applicant_income_000s', 'loan_amount_000s', 'loan_to_income']
CATEGORICAL_COLS = ['agency_code', 'loan_type', 'loan_purpose', 'owner_occupancy',
                    'preapproval', 'lien_status', 'county_code']
NUMERIC_COLS = ['tract_to_msamd_income', 'minority_population', 'population',
                'number_of_owner_occupied_units', 'number_of_1_to_4_family_units']
BINARY_COLS = ['income_missing', 'has_co_applicant']

# Race is stored as up to five codes per person, so each race gets its own 0/1 column
RACE_NAMES = {1: 'aian', 2: 'asian', 3: 'black', 4: 'nhpi', 5: 'white'}
RACE_PREFIXES = ['applicant', 'co_applicant']
RACE_COLS = [f'{p}_race_{n}' for p in RACE_PREFIXES for n in RACE_NAMES.values()]
RACE_FLAG_COLS = [f'{p}_{f}' for p in RACE_PREFIXES for f in ('race_not_provided', 'multiracial')]
RACE_INDICATOR_COLS = RACE_COLS + RACE_FLAG_COLS

# Held out for the fairness audit. The with-protected variant also trains on sex, ethnicity and the race columns.
PROTECTED_CAT_COLS = ['applicant_sex', 'applicant_ethnicity', 'co_applicant_sex', 'co_applicant_ethnicity']
PROTECTED_COLS = PROTECTED_CAT_COLS + ['applicant_race_1', 'co_applicant_race_1'] + RACE_INDICATOR_COLS

############## Handlers ###############
def build_target(df: pd.DataFrame) -> pd.DataFrame:
  """Approved (1): action_taken 1, 2. Denied (0): 3, 7. Withdrawn (4), incomplete (5) and purchased (6) are removed."""
  df = df[~df['action_taken'].isin([4, 5, 6])].copy()
  df[TARGET] = df['action_taken'].isin([1, 2]).astype(int)
  return df.drop(columns=['action_taken'])

def filter_scope(df: pd.DataFrame) -> pd.DataFrame:
  """Keep one-to-four family homes with an applicant who is a person."""
  before = len(df)
  df = df[df['property_type'].isin(KEEP_PROPERTY_TYPES)]
  print(f'Rows removed for property_type not in {KEEP_PROPERTY_TYPES}: {before - len(df)}')
  before = len(df)
  df = df[df['applicant_race_1'] != NOT_APPLICABLE_RACE]
  print(f'Rows removed for applicant_race_1 == {NOT_APPLICABLE_RACE}: {before - len(df)}')
  return df.copy()

def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
  """One 0/1 column per race, set if that race appears in any of race_1..race_5. Must run before those columns are dropped."""
  out = df.copy()
  for prefix in RACE_PREFIXES:
    listed = out[[f'{prefix}_race_{i}' for i in range(1, 6)]]
    for code, name in RACE_NAMES.items():
      out[f'{prefix}_race_{name}'] = (listed == code).any(axis=1).astype(int)
    out[f'{prefix}_race_not_provided'] = (out[f'{prefix}_race_1'] == 6).astype(int)
    out[f'{prefix}_multiracial'] = (out[[f'{prefix}_race_{n}' for n in RACE_NAMES.values()]].sum(axis=1) >= 2).astype(int)
  return out

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
  """Row-level features that learn nothing from the data. Clipping and logs happen in FeaturePreprocessor."""
  out = frame.copy()
  income = out['applicant_income_000s']
  out['income_missing'] = income.isna().astype(int)
  out['has_co_applicant'] = (out['co_applicant_sex'] != 5).astype(int)
  out['loan_to_income'] = out['loan_amount_000s'] / income.replace(0, np.nan)
  return out

class FeaturePreprocessor:
  """Winsorize, log, impute, encode and scale the model inputs, learning every step from the training split.

  The lender is target-encoded as a smoothed approval rate. The training rows get an out-of-fold rate (computed
  without their own outcome); test rows get the rate from the whole training split.
  """

  def __init__(self, cat_cols: list, log_source_cols: list, numeric_cols: list, binary_cols: list) -> None:
    self.cat_cols = cat_cols
    self.log_source_cols = log_source_cols
    self.log_cols = [f'log_{c}' for c in log_source_cols]
    self.numeric_cols = numeric_cols
    self.binary_cols = binary_cols
    self.num_names = self.log_cols + numeric_cols + LENDER_COLS
    self.bounds = {}
    self.lender_stats = None
    self.lender_mean = None
    self.cat_imputer = SimpleImputer(strategy='constant', fill_value=-1)
    self.num_imputer = SimpleImputer(strategy='median')
    self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int)
    self.scaler = StandardScaler()

  def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Fit on the training split and return its features."""
    self.bounds = {c: tuple(np.nanpercentile(X[c].to_numpy(dtype=float), WINSOR_LIMITS))
                   for c in self.log_source_cols}
    self.lender_mean = y.mean()
    self.lender_stats = y.groupby(X[LENDER]).agg(['sum', 'count'])
    lender_rate = self._out_of_fold_lender_rate(X[LENDER], y)

    self.cat_imputer.fit(X[self.cat_cols])
    self.encoder.fit(self._impute_cat(X))
    self.num_imputer.fit(self._numeric(X, lender_rate))
    self.scaler.fit(self.num_imputer.transform(self._numeric(X, lender_rate)))
    return self._assemble(X, lender_rate)

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    """Features for new rows, such as the test split."""
    return self._assemble(X, self._lender_rate(X[LENDER], self.lender_stats, self.lender_mean))

  def _assemble(self, X: pd.DataFrame, lender_rate: pd.Series) -> pd.DataFrame:
    cat = pd.DataFrame(self.encoder.transform(self._impute_cat(X)),
                       columns=self.encoder.get_feature_names_out(), index=X.index)
    num = pd.DataFrame(self.scaler.transform(self.num_imputer.transform(self._numeric(X, lender_rate))),
                       columns=self.num_names, index=X.index)
    return pd.concat([cat, num, X[self.binary_cols]], axis=1)

  def _impute_cat(self, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(self.cat_imputer.transform(X[self.cat_cols]),
                        columns=self.cat_cols, index=X.index).astype(int)

  def _numeric(self, X: pd.DataFrame, lender_rate: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for src, dst in zip(self.log_source_cols, self.log_cols):
      lo, hi = self.bounds[src]
      out[dst] = np.log1p(X[src].clip(lower=lo, upper=hi))
    for col in self.numeric_cols:
      out[col] = X[col]
    out['lender_approval_rate'] = lender_rate
    out['log_lender_applications'] = np.log1p(X[LENDER].map(self.lender_stats['count']).fillna(0))
    return out

  @staticmethod
  def _lender_rate(lenders: pd.Series, stats: pd.DataFrame, overall_mean: float) -> pd.Series:
    """Approval rate shrunk toward the overall rate. A lender not in stats gets the overall rate."""
    approved = lenders.map(stats['sum']).fillna(0)
    total = lenders.map(stats['count']).fillna(0)
    return (approved + LENDER_SMOOTHING * overall_mean) / (total + LENDER_SMOOTHING)

  def _out_of_fold_lender_rate(self, lenders: pd.Series, y: pd.Series) -> pd.Series:
    rate = np.empty(len(lenders))
    for fit_idx, held_idx in KFold(LENDER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(lenders):
      stats = y.iloc[fit_idx].groupby(lenders.iloc[fit_idx]).agg(['sum', 'count'])
      rate[held_idx] = self._lender_rate(lenders.iloc[held_idx], stats, y.iloc[fit_idx].mean()).to_numpy()
    return pd.Series(rate, index=lenders.index)

############## Entry Point #############
if __name__ == "__main__":
  df = pd.read_csv(INPUT_PATH, low_memory=False)
  print('Input shape:', df.shape)

  df = build_target(df)
  print('Shape with target:', df.shape, f'approval rate {df[TARGET].mean():.3f}')

  df = filter_scope(df)
  print('Shape after scope filter:', df.shape, f'approval rate {df[TARGET].mean():.3f}')

  df = add_race_features(df)
  df = drop_unused_columns(df)
  df = add_row_features(df)

  X = df.drop(columns=[TARGET])
  y = df[TARGET]
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
  print('Train:', X_train.shape, f'approval rate {y_train.mean():.3f}')
  print('Test: ', X_test.shape, f'approval rate {y_test.mean():.3f}')

  # Two feature sets from the same split: without protected attributes (the default model) and with them (for comparison)
  variants = {
    '': (CATEGORICAL_COLS, BINARY_COLS),
    '_with_protected': (CATEGORICAL_COLS + PROTECTED_CAT_COLS, BINARY_COLS + RACE_INDICATOR_COLS),
  }

  # row_id is the row's position in the input CSV, so every file lines up with the others
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  outputs = {'y_train': y_train, 'y_test': y_test,
             'protected_train': X_train[PROTECTED_COLS], 'protected_test': X_test[PROTECTED_COLS]}

  for suffix, (cat_cols, binary_cols) in variants.items():
    missing = X_train[cat_cols + LOG_SOURCE_COLS + NUMERIC_COLS].isna().sum()
    print(f'Missing values in the training split{suffix}:')
    print(missing[missing > 0].to_string())

    preprocessor = FeaturePreprocessor(cat_cols, LOG_SOURCE_COLS, NUMERIC_COLS, binary_cols)
    X_train_prep = preprocessor.fit_transform(X_train, y_train)
    X_test_prep = preprocessor.transform(X_test)

    assert not X_train_prep.isna().any().any(), 'NaNs left in X_train_prep'
    assert not X_test_prep.isna().any().any(), 'NaNs left in X_test_prep'
    assert list(X_train_prep.columns) == list(X_test_prep.columns)
    print(f'X_train_prep{suffix}:', X_train_prep.shape)
    print(f'X_test_prep{suffix}: ', X_test_prep.shape)

    outputs[f'X_train_prep{suffix}'] = X_train_prep
    outputs[f'X_test_prep{suffix}'] = X_test_prep
    # Saved as a plain dict of the fitted sklearn objects and column lists, so loading it doesn't need this script
    joblib.dump(vars(preprocessor), os.path.join(OUTPUT_DIR, f'preprocessors{suffix}.joblib'))

  for name, data in outputs.items():
    data.to_csv(os.path.join(OUTPUT_DIR, f'{name}.csv'), index_label='row_id')
  print(f'Saved {", ".join(outputs)} and preprocessors → {OUTPUT_DIR}')
