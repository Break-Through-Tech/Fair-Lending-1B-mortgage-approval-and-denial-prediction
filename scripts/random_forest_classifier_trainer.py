"""
Train a Random Forest Classifier model.
"""

############## Imports ###############
import os
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
  accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

############## Data Files #############
BALANCED_MODEL_PATH = os.path.join(os.getcwd(), "data", "models", "random_forest_balanced.joblib")
BALANCED_PROTECTED_MODEL_PATH = os.path.join(os.getcwd(), "data", "models", "random_forest_balanced_protected.joblib")
IMBALANCED_MODEL_PATH = os.path.join(os.getcwd(), "data", "models", "random_forest_classifier_imbalanced.joblib")

############### Load Training Data ###############
X_train = pd.read_csv('data/features/X_train_prep.csv', index_col='row_id')
X_train_protected = pd.read_csv('data/features/X_train_prep_with_protected.csv', index_col='row_id')
y_train = pd.read_csv('data/features/y_train.csv', index_col='row_id')['is_approved']
X_test = pd.read_csv('data/features/X_test_prep.csv', index_col='row_id')
X_test_protected = pd.read_csv('data/features/X_test_prep_with_protected.csv', index_col='row_id')
y_test = pd.read_csv('data/features/y_test.csv', index_col='row_id')['is_approved']

############### Define Hyperparameter Grid ###############
param_grid = {
  'n_estimators': [50, 100],
  'max_depth': [15, 25],
  'min_samples_leaf': [2],
  'max_features': ['sqrt']
}

# Define balanced and imbalanced model parameters
model_specs = [
  ('balanced', BALANCED_MODEL_PATH, "Balanced Model"),
  ('balanced', BALANCED_PROTECTED_MODEL_PATH, "Balanced Protected Model"),
  (None, IMBALANCED_MODEL_PATH, "Imbalanced Model"),
]

############### Train Model ##############
for class_weight_setting, model_path, model_label in model_specs:
  # Define model.
  model = RandomForestClassifier(
    random_state=42,
    class_weight=class_weight_setting,
    n_jobs=-1
  )

  grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1   # Summary Notifications
  )

  print(f"Starting Grid Search: {model_label} ...")
  grid_search.fit(X_train, y_train)

  optimal_model = grid_search.best_estimator_
  print(f"Best hyperparameters for {model_label}: {grid_search.best_params_}")
    
  # Save the optimal model
  joblib.dump(optimal_model, model_path)
  print(f"Successfully saved model -> {model_path}")

  ############### Evaluate Model ##############
  y_pred = optimal_model.predict(X_test)
  y_pred_proba = optimal_model.predict_proba(X_test)[:, 1]

  print(f"\n--- {model_label} Evaluation Metrics ---")
  print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
  print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
  print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

  print("\nConfusion matrix (rows = actual, cols = predicted; denied, approved):")
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred, target_names=['denied', 'approved']))

  importances = pd.Series(optimal_model.feature_importances_, index=X_train.columns)
  print("\nTop 15 feature importances:")
  print(importances.sort_values(ascending=False).head(15).to_string())
  print("\n")
