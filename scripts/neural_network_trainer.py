"""
Train a Keras neural network, searching for hyperparameters in two stages:
  1. every config on a stratified subsample, with few folds, to rank them cheaply
  2. the best few configs on all training rows, with more folds, to pick the winner

Local experiment file. It is listed in .git/info/exclude, so it is never pushed.
Set NN_SMOKE=1 for a tiny run that checks the whole script in about a minute.
"""

############## Imports ###############
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')  # Keras 3 runs on the installed torch; set before importing keras

import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import keras
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

############## Data Files #############
FEATURE_DIR = os.path.join(os.getcwd(), "data", "features")
MODEL_DIR = os.path.join(os.getcwd(), "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "neural_network_model.keras")
RESULTS_PATH = os.path.join(MODEL_DIR, "neural_network_search.csv")

############## Settings ###############
VARIANT = ''            # '' for the default feature set, '_with_protected' for the comparison one
RANDOM_STATE = 42
BATCH_SIZE = 1024
MAX_EPOCHS = 40
PATIENCE = 4            # epochs without a better validation loss before stopping
SEARCH_SAMPLE = 0.25    # share of the training rows stage 1 searches on
SEARCH_FOLDS = 3
FINAL_FOLDS = 5
FINALISTS = 3           # configs carried from stage 1 into stage 2
THRESHOLD = 0.5         # probability cutoff for the F1 score

############## Grid ##################
GRID = [{'units': units, 'dropout': dropout, 'lr': lr}
        for units in [(64, 32), (128, 64), (256, 128)]
        for dropout in [0.2, 0.4]
        for lr in [1e-3, 3e-4]]

SMOKE = os.environ.get('NN_SMOKE') == '1'
if SMOKE:
  GRID = GRID[:2]
  MAX_EPOCHS, SEARCH_SAMPLE, SEARCH_FOLDS, FINAL_FOLDS, FINALISTS = 2, 0.05, 2, 2, 1

############## Handlers ##############
def build_model(n_features: int, config: dict) -> keras.Model:
  """A plain feed-forward net: dense + dropout per hidden layer, one sigmoid output."""
  keras.utils.set_random_seed(RANDOM_STATE)
  layers = [keras.layers.Input(shape=(n_features,))]
  for units in config['units']:
    layers.append(keras.layers.Dense(units, activation='relu'))
    layers.append(keras.layers.Dropout(config['dropout']))
  layers.append(keras.layers.Dense(1, activation='sigmoid'))

  model = keras.Sequential(layers)
  model.compile(optimizer=keras.optimizers.Adam(config['lr']), loss='binary_crossentropy')
  return model

def fit_once(X_fit, y_fit, X_val, y_val, config: dict, class_weight: dict):
  """Train on X_fit, stopping early on the validation loss, and score on X_val."""
  model = build_model(X_fit.shape[1], config)
  model.fit(X_fit, y_fit,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
            verbose=0)

  proba = model.predict(X_val, batch_size=4096, verbose=0).ravel()
  scores = {
    'f1': f1_score(y_val, proba >= THRESHOLD),
    'roc_auc': roc_auc_score(y_val, proba),
    'pr_auc': average_precision_score(y_val, proba),
    'epochs_run': len(model.history.history['loss']),
  }
  return model, scores

def cross_validate(X, y, config: dict, folds: int, class_weight: dict) -> dict:
  """Mean score over stratified folds, so every fold keeps the overall approval rate."""
  splitter = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE)
  fold_scores = [fit_once(X[fit_idx], y[fit_idx], X[val_idx], y[val_idx], config, class_weight)[1]
                 for fit_idx, val_idx in splitter.split(X, y)]
  return {name: float(np.mean([s[name] for s in fold_scores])) for name in fold_scores[0]}

def run_stage(stage: int, X, y, configs: list, folds: int, class_weight: dict) -> list:
  print(f'\nStage {stage}: {len(configs)} configs x {folds} folds on {len(y):,} rows')
  rows = []
  for i, config in enumerate(configs, 1):
    start = time.time()
    scores = cross_validate(X, y, config, folds, class_weight)
    rows.append({'stage': stage, **config, **scores, 'seconds': round(time.time() - start, 1)})
    print(f'  [{i}/{len(configs)}] units {config["units"]} dropout {config["dropout"]} lr {config["lr"]:g}'
          f' -> f1 {scores["f1"]:.4f}  roc_auc {scores["roc_auc"]:.4f}  pr_auc {scores["pr_auc"]:.4f}'
          f'  ({time.time() - start:.0f}s)')
  return rows

############## Entry Point #############
if __name__ == "__main__":
  X_df = pd.read_csv(os.path.join(FEATURE_DIR, f'X_train_prep{VARIANT}.csv'), index_col='row_id')
  y_series = pd.read_csv(os.path.join(FEATURE_DIR, 'y_train.csv'), index_col='row_id')['is_approved']
  X = X_df.to_numpy(dtype='float32')
  y = y_series.to_numpy()
  print(f'Backend: {keras.backend.backend()}  rows: {len(y):,}  features: {X.shape[1]}'
        f'  approval rate: {y.mean():.3f}')

  # Same idea as class_weight='balanced': the smaller class (denied) counts for more
  class_weight = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * (y == 1).sum())}
  print(f'Class weights: {{0: {class_weight[0]:.3f}, 1: {class_weight[1]:.3f}}}')

  X_sample, _, y_sample, _ = train_test_split(X, y, train_size=SEARCH_SAMPLE,
                                              stratify=y, random_state=RANDOM_STATE)
  results = run_stage(1, X_sample, y_sample, GRID, SEARCH_FOLDS, class_weight)

  finalists = [{key: row[key] for key in ('units', 'dropout', 'lr')}
               for row in sorted(results, key=lambda r: -r['f1'])[:FINALISTS]]
  results += run_stage(2, X, y, finalists, FINAL_FOLDS, class_weight)

  best = max((r for r in results if r['stage'] == 2), key=lambda r: r['f1'])
  best_config = {key: best[key] for key in ('units', 'dropout', 'lr')}
  print(f'\nBest config: {best_config}')
  print(f'  cross-validated f1 {best["f1"]:.4f}  roc_auc {best["roc_auc"]:.4f}  pr_auc {best["pr_auc"]:.4f}'
        f'  ({best["epochs_run"]:.0f} epochs on average)')

  # Final model: all training rows except a slice held back for early stopping
  X_fit, X_stop, y_fit, y_stop = train_test_split(X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE)
  model, scores = fit_once(X_fit, y_fit, X_stop, y_stop, best_config, class_weight)
  print(f'Final model on the early-stopping slice: f1 {scores["f1"]:.4f}  roc_auc {scores["roc_auc"]:.4f}'
        f'  ({scores["epochs_run"]} epochs)')

  os.makedirs(MODEL_DIR, exist_ok=True)
  model.save(MODEL_PATH)
  pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
  print(f'\nSaved model -> {MODEL_PATH}')
  print(f'Saved search results -> {RESULTS_PATH}')
