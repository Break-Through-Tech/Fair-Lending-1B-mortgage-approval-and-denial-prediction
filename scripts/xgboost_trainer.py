"""
Train a gradient boosting model with XGBoost and report test metrics.
"""

############## Imports ###############
import os
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
  accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

############## Data Files #############
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "models", "xgboost_model.joblib")

############### Load Data ###############
X_train = pd.read_csv('data/features/X_train_prep.csv', index_col='row_id')
y_train = pd.read_csv('data/features/y_train.csv', index_col='row_id')['is_approved']
X_test = pd.read_csv('data/features/X_test_prep.csv', index_col='row_id')
y_test = pd.read_csv('data/features/y_test.csv', index_col='row_id')['is_approved']

############### Train Model ##############
# Handle class imbalance. Approved is the majority class, so weight it down.
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
  n_estimators=200,
  scale_pos_weight=scale_pos_weight,
  tree_method='hist',
  random_state=42,
  n_jobs=1,
)

# Define hyperparameter grid. Learning rate and depth are the settings to tune.
param_grid = {
  'learning_rate': [0.05, 0.1, 0.3],
  'max_depth': [3, 6, 10],
}

# Perform grid search
grid_search = GridSearchCV(
  estimator=model,
  param_grid=param_grid,
  cv=5,        
  scoring='f1', 
  n_jobs=-1,
)

print("Searching for optimal hyperparameters...")
grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)
print("\nAll settings, best first:")
print(
  results[['param_learning_rate', 'param_max_depth', 'mean_test_score', 'std_test_score']]
  .sort_values('mean_test_score', ascending=False)
  .to_string(index=False)
)

optimal_model = grid_search.best_estimator_
print(f"\nBest hyperparameters: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")

# Save model
joblib.dump(optimal_model, OUTPUT_PATH)
print(f"XGBoost Model saved -> {OUTPUT_PATH}")

############### Evaluate Model ##############
y_pred = optimal_model.predict(X_test)
y_pred_proba = optimal_model.predict_proba(X_test)[:, 1]

print("\n--- XGBoost Evaluation Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['denied', 'approved']))

importances = pd.Series(optimal_model.feature_importances_, index=X_train.columns)
print("Top 15 feature importances:")
print(importances.sort_values(ascending=False).head(15).to_string())
