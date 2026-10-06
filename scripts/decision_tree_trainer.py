"""
Train a decision tree model and report test metrics.
"""

############## Imports ###############
import os
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
  accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

############## Data Files #############
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "models", "decision_tree_model.joblib")

############### Load Data ###############
X_train = pd.read_csv('data/features/X_train_prep.csv', index_col='row_id')
y_train = pd.read_csv('data/features/y_train.csv', index_col='row_id')['is_approved']
X_test = pd.read_csv('data/features/X_test_prep.csv', index_col='row_id')
y_test = pd.read_csv('data/features/y_test.csv', index_col='row_id')['is_approved']

############### Train Model ##############
model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Define hyperparameter grid. Depth and leaf size limit overfitting.
param_grid = {
  'criterion': ['gini'],
  'max_depth': [5, 10, 15, 20],
  'min_samples_leaf': [50, 200, 1000],
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
print(f"Best CV F1: {grid_search.best_score_:.4f}")

# Save model
joblib.dump(optimal_model, OUTPUT_PATH)
print(f"Decision Tree Model saved -> {OUTPUT_PATH}")

############### Evaluate Model ##############
y_pred = optimal_model.predict(X_test)
y_pred_proba = optimal_model.predict_proba(X_test)[:, 1]

print("\n--- Decision Tree Evaluation Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nConfusion matrix (rows = actual, cols = predicted; denied, approved):")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['denied', 'approved']))

importances = pd.Series(optimal_model.feature_importances_, index=X_train.columns)
print("Top 15 feature importances:")
print(importances.sort_values(ascending=False).head(15).to_string())
