"""
Train a logistic regression model.
"""

############## Imports ###############
import os
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

############## Data Files #############
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "models", "logistic_regression_model.joblib")

############### Load Training Data ###############
X_train = pd.read_csv('data/features/X_train_prep.csv', index_col='row_id')
y_train = pd.read_csv('data/features/y_train.csv', index_col='row_id')['is_approved']

############### Train Model ##############
# Define model. Handle class imbalance with 'balanced'
model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')

# Define hyperparameter grid
param_grid = {
  'C': [0.1, 1, 10, 100], 
  'penalty': ['l2'],       
  'solver': ['lbfgs']      
}

# Perform grid search
grid_search = GridSearchCV(
  estimator=model,
  param_grid=param_grid,
  cv=5,         # 5-fold cross-validation
  scoring='f1', # Use F1 score for imbalanced datasets
  n_jobs=-1,    # Fast training with parallel processing
)

print("Searching for optimal hyperparameters...")
grid_search.fit(X_train, y_train)

optimal_model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")

# Save model
joblib.dump(optimal_model, OUTPUT_PATH)
print(f"Logistic Regression Model saved -> {OUTPUT_PATH}")

