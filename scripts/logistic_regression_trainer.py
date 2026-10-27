"""
Train a logistic regression model.
"""

############## Imports ###############
import os
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
  accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

############## Data Files #############
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "models", "logistic_regression_model.joblib")

############### Load Training Data ###############
X_train = pd.read_csv('data/features/X_train_prep.csv', index_col='row_id')
y_train = pd.read_csv('data/features/y_train.csv', index_col='row_id')['is_approved']
X_test = pd.read_csv('data/features/X_test_prep.csv', index_col='row_id')
y_test = pd.read_csv('data/features/y_test.csv', index_col='row_id')['is_approved']

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

print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

optimal_model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")

# Save model
joblib.dump(optimal_model, OUTPUT_PATH)
print(f"Logistic Regression Model saved -> {OUTPUT_PATH}")

############### Evaluate Model ##############
y_pred = optimal_model.predict(X_test)
y_pred_proba = optimal_model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression Evaluation Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nConfusion matrix (rows = actual, cols = predicted; denied, approved):")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['denied', 'approved']))

#### Feature Importance Metrics
# Find Absolute Importance scores
coefficients = optimal_model.coef_[0]
feature_names = [name for name in X_test.columns]
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Absolute_Importance': np.abs(coefficients)
})
# Sort features by importance and grab only the top 15 rows
top_15_df = importance_df.sort_values(by='Absolute_Importance', ascending=False).reset_index(drop=True)
print("Top 15 feature importances:")
print(top_15_df.head(15).to_string(index=False))
