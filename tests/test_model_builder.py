# ================================================================
#  tests/test_model_builder.py
# ================================================================
#  Unit and integration tests for the ModelBuilder class.
#
#  - Uses pytest for test structure and execution.
#  - Leverages fixtures for reusable test data.
#  - Covers initialization, evaluation, selection, training,
#    and persistence logic.
#  - Ensures correct error handling for invalid operations.
#  - Verifies end-to-end workflow for both classification and regression.
#
#  Fixtures:
#   - classification_data: Provides a sample classification dataset.
#   - regression_data: Provides a sample regression dataset.
#
#  Tested Components:
#   - Initialization for both classification and regression tasks.
#   - Custom model injection.
#   - Model evaluation via cross-validation.
#   - Model selection logic.
#   - Model training and persistence (save/load).
#   - Error handling for invalid operations.
#
#  Author: Flavio Aguirre
#  Date: 2025-08-28
# ================================================================

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# --- Module to be tested ---
from src.model_builder import ModelBuilder, ModelBuilderError   # type: ignore

# ================================================================
#  Fixtures: Reusable Test Data
# ================================================================

@pytest.fixture(scope="module")
def classification_data():
    """
    Provides a sample classification dataset.

    Returns
    -------
    tuple (pd.DataFrame, pd.Series)
        Features and target for classification.
    """
    X, y = make_classification(
        n_samples=100, n_features=10, random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)

@pytest.fixture(scope="module")
def regression_data():
    """
    Provides a sample regression dataset.

    Returns
    -------
    tuple (pd.DataFrame, pd.Series)
        Features and target for regression.
    """
    X, y = make_regression(
        n_samples=100, n_features=10, random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)

# ================================================================
#  Test Cases
# ================================================================

def test_initialization_classification():
    """
    Tests if the ModelBuilder initializes correctly for classification tasks.

    Asserts:
        - The task attribute is set to 'classification'.
        - Default classification models are present in the models dictionary.
        - The random forest model is an instance of RandomForestClassifier.
    """
    builder = ModelBuilder(task="classification", random_state=42)
    assert builder.task == "classification"
    assert "logistic_regression" in builder.models
    assert "xgboost" in builder.models
    assert isinstance(builder.models["random_forest"], RandomForestClassifier)

def test_initialization_regression():
    """
    Tests if the ModelBuilder initializes correctly for regression tasks.

    Asserts:
        - The task attribute is set to 'regression'.
        - Default regression models are present in the models dictionary.
        - The linear regression model is an instance of LinearRegression.
    """
    builder = ModelBuilder(task="regression", random_state=42)
    assert builder.task == "regression"
    assert "linear_regression" in builder.models
    assert "xgboost" in builder.models
    assert isinstance(builder.models["linear_regression"], LinearRegression)

def test_initialization_invalid_task():
    """
    Tests if the ModelBuilder raises an error for an unsupported task.

    Asserts:
        - ModelBuilderError is raised with the correct message.
    """
    with pytest.raises(ModelBuilderError, match="Task must be 'classification' or 'regression'"):
        ModelBuilder(task="clustering")

def test_custom_models_injection():
    """
    Tests if custom models can be correctly passed during initialization.

    Asserts:
        - The models dictionary contains only the custom model.
        - The custom model has the correct parameters.
    """
    custom_models = {
        "my_rf": RandomForestClassifier(n_estimators=50, random_state=42)
    }
    builder = ModelBuilder(task="classification", custom_models=custom_models)
    assert list(builder.models.keys()) == ["my_rf"]
    assert builder.models["my_rf"].n_estimators == 50

def test_evaluate_models_classification(classification_data):
    """
    Tests the cross-validation evaluation for classification models.

    Asserts:
        - The results are returned as a dictionary.
        - The 'xgboost' model is present in the results.
        - Standard classification metrics are present in the results.
    """
    X_train, y_train = classification_data
    builder = ModelBuilder(task="classification")
    results = builder.evaluate_models(X_train, y_train)

    assert isinstance(results, dict)
    assert "xgboost" in results
    assert "test_f1" in results["xgboost"]
    assert "test_roc_auc" in results["xgboost"]

def test_evaluate_models_regression(regression_data):
    """
    Tests the cross-validation evaluation for regression models.

    Asserts:
        - The results are returned as a dictionary.
        - The 'linear_regression' model is present in the results.
        - Standard regression metrics are present in the results.
    """
    X_train, y_train = regression_data
    builder = ModelBuilder(task="regression")
    results = builder.evaluate_models(X_train, y_train)

    assert isinstance(results, dict)
    assert "linear_regression" in results
    assert "test_r2" in results["linear_regression"]

def test_select_best_model_without_evaluation():
    """
    Tests that selecting a model before evaluation raises an error.

    Asserts:
        - ModelBuilderError is raised with the correct message.
    """
    builder = ModelBuilder(task="classification")
    with pytest.raises(ModelBuilderError, match="No models evaluated yet"):
        builder.select_best_model()

def test_train_final_model_not_found(classification_data):
    """
    Tests that training a non-existent model raises an error.

    Asserts:
        - ModelBuilderError is raised with the correct message.
    """
    X_train, y_train = classification_data
    builder = ModelBuilder(task="classification")

    with pytest.raises(ModelBuilderError, match="Model 'non_existent_model' not found"):
        builder.train_final_model("non_existent_model", X_train, y_train)

def test_save_model_not_trained():
    """
    Tests that saving a model before it's trained raises an error.

    Asserts:
        - ModelBuilderError is raised with the correct message.
    """
    builder = ModelBuilder(task="classification")
    with pytest.raises(ModelBuilderError, match="Trained model 'xgboost' not found"):
        builder.save_model("xgboost", "dummy_path.joblib")

def test_full_classification_workflow(classification_data, tmp_path):
    """
    Tests the complete end-to-end workflow for a classification task:
    evaluate -> select -> train -> save -> load -> predict.

    Asserts:
        - The best model is selected and present in the models dictionary.
        - The final model is trained and present in trained_models_.
        - The model file is saved to disk.
        - The loaded model is not None.
        - The loaded model produces the same predictions as the original.
        - The loaded model's predict_proba output has the correct shape.
    """
    X_train, y_train = classification_data
    builder = ModelBuilder(task="classification", random_state=42)

    # 1. Evaluate
    builder.evaluate_models(X_train, y_train, cv=3)

    # 2. Select
    builder.select_best_model(metric="test_f1")
    best_model_name = builder.best_model_name_
    assert best_model_name is not None
    assert best_model_name in builder.models

    # 3. Train
    final_model = builder.train_final_model(best_model_name, X_train, y_train)
    assert best_model_name in builder.trained_models_

    # 4. Save
    model_path = tmp_path / f"{best_model_name}.joblib"
    builder.save_model(best_model_name, str(model_path))
    assert model_path.exists()

    # 5. Load
    loaded_model = builder.load_model(str(model_path))
    assert loaded_model is not None

    # 6. Verify predictions
    # Ensure the loaded model gives the exact same predictions as the original
    original_predictions = final_model.predict(X_train)
    loaded_model_predictions = loaded_model.predict(X_train)
    np.testing.assert_array_equal(original_predictions, loaded_model_predictions)

    # Check that predict_proba also works and has the correct shape
    proba = loaded_model.predict_proba(X_train)
    assert proba.shape == (X_train.shape[0], 2)

def test_full_regression_workflow(regression_data, tmp_path):
    """
    Tests the complete end-to-end workflow for a regression task:
    evaluate -> select -> train -> save -> load -> predict.

    Asserts:
        - The best model is selected and present in the models dictionary.
        - The final model is trained and present in trained_models_.
        - The model file is saved to disk.
        - The loaded model is not None.
        - The loaded model produces the same predictions as the original.
        - The loaded model's predict_proba output has the correct shape.
    """
    X_train, y_train = regression_data
    builder = ModelBuilder(task="regression", random_state=42)

    # 1. Evaluate
    builder.evaluate_models(X_train, y_train, cv=3)

    # 2. Select
    builder.select_best_model(metric="test_r2")
    best_model_name = builder.best_model_name_
    assert best_model_name is not None
    assert best_model_name in builder.models

    # 3. Train
    final_model = builder.train_final_model(best_model_name, X_train, y_train)
    assert best_model_name in builder.trained_models_

    # 4. Save
    model_path = tmp_path / f"{best_model_name}.joblib"
    builder.save_model(best_model_name, str(model_path))
    assert model_path.exists()

    # 5. Load
    loaded_model = ModelBuilder.load_model(str(model_path))
    assert loaded_model is not None

    # 6. Verify predictions
    original_predictions = final_model.predict(X_train)
    loaded_model_predictions = loaded_model.predict(X_train)
    
    # We use almost_equal for floating point numbers and remove the
    # predict_proba check, as it does not apply to regression models.
    np.testing.assert_array_almost_equal(original_predictions, loaded_model_predictions)