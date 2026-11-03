"""
Tests for scripts/fairness_eval.py on small synthetic data. Run from the project root: python -m pytest tests
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from fairness_eval import results_by_group

############## Helpers ###############
class ScoreColumnModel:
  """sklearn-style model whose approval probability is the 'score' column of X."""

  def __init__(self, classes=(0, 1)):
    self.classes_ = np.array(classes)

  def predict_proba(self, X):
    approve = X['score'].to_numpy()
    return np.column_stack([approve, 1 - approve] if self.classes_[0] == 1 else [1 - approve, approve])

class KerasStyleModel:
  """Like a Keras binary classifier: predict returns an (n, 1) array of approval probabilities."""
  __module__ = 'keras.stub'

  def predict(self, X, verbose=0):
    return X[:, :1]

class PredictOnlyModel:
  """Like an sklearn classifier without predict_proba, e.g. LinearSVC."""

  def predict(self, X):
    return (X['score'] >= 0.5).astype(int).to_numpy()

def make_data():
  """Group A: 1000 rows, 80% predicted approved, TPR 1.0, FPR 0.6. Group B: 500 rows, 60%, TPR 1.0, FPR 0.2."""
  y = np.array([1] * 500 + [0] * 500 + [1] * 250 + [0] * 250)
  scores = np.array([0.9] * 800 + [0.1] * 200 + [0.9] * 300 + [0.1] * 200)
  index = pd.Index(np.arange(1500) + 1000, name='row_id')
  return (pd.DataFrame({'score': scores}, index=index),
          pd.Series(y, index=index, name='is_approved'),
          pd.Series(['A'] * 1000 + ['B'] * 500, index=index, name='group'))

def row(table: pd.DataFrame, group: str) -> pd.Series:
  return table[table['group'] == group].iloc[0]

############## Tests ##################
def test_results_per_group():
  table = results_by_group(ScoreColumnModel(), *make_data())

  assert table['group'].tolist() == ['overall', 'A', 'B']
  a, b = row(table, 'A'), row(table, 'B')
  assert (a['n'], a['n_denied'], b['n'], b['n_denied']) == (1000, 500, 500, 250)
  assert a['predicted_approval_rate'] == pytest.approx(0.8)
  assert b['predicted_approval_rate'] == pytest.approx(0.6)
  assert (a['tpr'], a['fpr']) == pytest.approx((1.0, 0.6))
  assert (b['tpr'], b['fpr']) == pytest.approx((1.0, 0.2))
  assert row(table, 'overall')['predicted_approval_rate'] == pytest.approx(1100 / 1500)

def test_threshold_is_applied():
  table = results_by_group(ScoreColumnModel(), *make_data(), threshold=0.95)
  assert row(table, 'overall')['predicted_approval_rate'] == 0

def test_misaligned_index_raises():
  X, y, protected = make_data()
  with pytest.raises(ValueError, match='same index'):
    results_by_group(ScoreColumnModel(), X, y, protected.reset_index(drop=True))

def test_single_class_group_has_nan_auc():
  X, y, protected = make_data()
  y[protected == 'B'] = 1
  b = row(results_by_group(ScoreColumnModel(), X, y, protected), 'B')
  assert np.isnan(b['auc'])
  assert np.isnan(b['fpr'])

def test_reversed_classes_score_approval():
  X, y, protected = make_data()
  pd.testing.assert_frame_equal(results_by_group(ScoreColumnModel(), X, y, protected),
                                results_by_group(ScoreColumnModel(classes=(1, 0)), X, y, protected))

@pytest.mark.parametrize('model', [KerasStyleModel(), PredictOnlyModel()])
def test_models_without_predict_proba(model):
  X, y, protected = make_data()
  table = results_by_group(model, X, y, protected)
  assert row(table, 'A')['predicted_approval_rate'] == pytest.approx(0.8)
  assert row(table, 'B')['predicted_approval_rate'] == pytest.approx(0.6)

def test_only_the_columns_the_model_was_fit_on_are_used():
  X, y, protected = make_data()
  model = ScoreColumnModel()
  model.feature_names_in_ = np.array(['score'])
  table = results_by_group(model, X.assign(property_type_1=0.0), y, protected)
  assert row(table, 'A')['predicted_approval_rate'] == pytest.approx(0.8)

def test_intersections_and_hmda_labels():
  X, y, _ = make_data()
  protected = pd.DataFrame({'applicant_race_1': [5] * 1000 + [3] * 500,
                            'applicant_sex': [1, 2] * 750}, index=X.index)
  table = results_by_group(ScoreColumnModel(), X, y, protected)

  pairs = set(zip(table['applicant_race_1'], table['applicant_sex'])) - {('overall', 'overall')}
  assert pairs == {('White', 'Male'), ('White', 'Female'),
                   ('Black or African American', 'Male'), ('Black or African American', 'Female')}
  assert table['n'].iloc[1:].sum() == 1500
