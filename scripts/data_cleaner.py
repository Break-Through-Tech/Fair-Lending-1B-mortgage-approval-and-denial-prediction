"""
This python script modifies the original data to prepare for model training.
"""

############## Imports #################
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

############## Data Files #############
mortgage_filename = os.path.join(os.getcwd(), "data", "ny_hmda_2015.csv")

############## Handlers ###############
class MissingValueHandler:

  def __init__(self) -> None:
    self.df = df.copy()

  def analyze(self) -> None:
    pass



############## Entry Point #############
if __name__ == "__main__":
  df = pd.read_csv(mortgage_filename, header=0)

  # Reorganize Target Column `action_taken`
  df = df[~df['action_taken'].isin([4,5])]
  df['is_approved'] = df['action_taken'].isin([1, 3, 6]).astype(int)
  df.drop(columns=['action_taken', 'action_taken_name'], inplace=True)

  df.to_csv('data/updated_ny_hmda_2015.csv', index=False)