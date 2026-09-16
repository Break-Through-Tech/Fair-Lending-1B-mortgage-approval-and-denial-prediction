"""
Winsorize continuous loan features at configured percentile limits.
"""

############## Imports #################
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin

############## Data Files #############
INPUT_PATH = os.path.join(os.getcwd(), "data", "ny_hmda_2015_dedup.csv")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "ny_hmda_2015_dedup_outliers.csv")

############## Handlers ###############
class OutlierHandler(BaseEstimator, TransformerMixin):
  """Winsorize selected numeric columns using percentile clipping."""

  def __init__(
    self,
    columns=None,
    limits=(0.01, 0.99),
  ) -> None:
    self.columns = columns or ['applicant_income_000s', 'loan_amount_000s']
    self.limits = limits
    self.bounds_ = {}

  def fit(self, X, y=None):
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    low_q, high_q = self.limits[0] * 100, self.limits[1] * 100
    self.bounds_ = {}
    for col in self.columns:
      values = df[col].to_numpy(dtype=float)
      lo, hi = np.nanpercentile(values, [low_q, high_q])
      self.bounds_[col] = (lo, hi)
    return self

  def transform(self, X):
    df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
    for col, (lo, hi) in self.bounds_.items():
      before = df[col].copy()
      df[col] = df[col].clip(lower=lo, upper=hi)
      n_capped = (before.notna() & (before != df[col])).sum()
      print(f'{col}: capped {n_capped} values to [{lo:.1f}, {hi:.1f}]')
    return df

  def report_iqr(self, X) -> None:
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    for col in self.columns:
      s = df[col].dropna()
      q1, q3 = s.quantile(0.25), s.quantile(0.75)
      iqr = q3 - q1
      lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
      n_out = ((s < lo) | (s > hi)).sum()
      print(f'{col}: {n_out} outside IQR [{lo:.1f}, {hi:.1f}] ({100 * n_out / len(s):.1f}%)')

############## Entry Point #############
if __name__ == "__main__":
  df = pd.read_csv(INPUT_PATH, low_memory=False)
  print('Input shape:', df.shape)

  handler = OutlierHandler()
  handler.report_iqr(df)
  df = handler.fit_transform(df)

  df.to_csv(OUTPUT_PATH, index=False)
  print(f'Saved {df.shape[0]} rows × {df.shape[1]} cols → {OUTPUT_PATH}')
