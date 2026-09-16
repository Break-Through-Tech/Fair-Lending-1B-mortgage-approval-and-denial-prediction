"""
Drop columns that contain only a single repeated value (no variation).
"""

############## Imports #################
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

############## Data Files #############
INPUT_PATH = os.path.join(os.getcwd(), "data", "ny_hmda_2015_dedup_outliers.csv")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "ny_hmda_2015_no_constant.csv")

############## Handlers ###############
class ConstantColumnHandler:
  """Remove columns with only one unique value (including all-missing)."""

  def __init__(self, df: pd.DataFrame) -> None:
    self.df = df.copy()

  def find_constant_columns(self) -> list:
    return [c for c in self.df.columns if self.df[c].nunique(dropna=False) <= 1]

  def drop_constant_columns(self) -> None:
    constant_cols = self.find_constant_columns()
    print(f'Constant columns ({len(constant_cols)}):')
    for col in constant_cols:
      print(f'  {col}: {self.df[col].iloc[0]}')
    self.df = self.df.drop(columns=constant_cols)
    print('Shape after dropping constant columns:', self.df.shape)

  def transform(self) -> pd.DataFrame:
    self.drop_constant_columns()
    return self.df

############## Entry Point #############
if __name__ == "__main__":
  df = pd.read_csv(INPUT_PATH, low_memory=False)
  print('Input shape:', df.shape)

  df = ConstantColumnHandler(df).transform()
  df.to_csv(OUTPUT_PATH, index=False)
  print(f'Saved {df.shape[0]} rows × {df.shape[1]} cols → {OUTPUT_PATH}')
