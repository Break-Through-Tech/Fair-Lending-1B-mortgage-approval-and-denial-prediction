"""
Drop exact duplicate rows and redundant HMDA *_name columns.
"""

############## Imports #################
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

############## Data Files #############
INPUT_PATH = os.path.join(os.getcwd(), "data", "ny_hmda_2015.csv")
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "ny_hmda_2015_dedup.csv")

############## Handlers ###############
class DuplicateHandler:
  """Remove exact row duplicates and code/name column redundancy."""

  def __init__(self, df: pd.DataFrame) -> None:
    self.df = df.copy()

  def drop_exact_duplicates(self) -> None:
    before = len(self.df)
    self.df = self.df.drop_duplicates()
    print(f'Exact duplicate rows removed: {before - len(self.df)}')

  def drop_name_columns(self) -> None:
    name_cols = [c for c in self.df.columns if '_name' in c]
    print(f'Redundant name columns ({len(name_cols)}):')
    print(name_cols)
    self.df = self.df.drop(columns=name_cols)
    print('Shape after dropping name columns:', self.df.shape)

  def transform(self) -> pd.DataFrame:
    self.drop_exact_duplicates()
    self.drop_name_columns()
    return self.df

############## Entry Point #############
if __name__ == "__main__":
  df = pd.read_csv(INPUT_PATH, low_memory=False)
  print('Input shape:', df.shape)

  df = DuplicateHandler(df).transform()
  df.to_csv(OUTPUT_PATH, index=False)
  print(f'Saved {df.shape[0]} rows × {df.shape[1]} cols → {OUTPUT_PATH}')
