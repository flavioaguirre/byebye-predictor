# ================================================================
#  tests/test_model_evaluation.py
# ================================================================
# Unit and integration tests for src/model_evaluation.py
#
# Covers:
# - ModelEvaluationError
# - ModelEvaluator (classification & regression)
# - Visualization helpers (confusion matrix, ROC, PR curves)
# - Compare multiple runs
# - Saving plots to a specified directory
#
# Author: Flavio Aguirre
# Date: 2025-08-29
# ================================================================

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.model_evaluation import (  # type: ignore
    ModelEvaluator,
    ModelEvaluationError
)


# ================================================================
# Fixtures
# ================================================================
@pytest.fixture
def binary_classification_data():
    """
    Provides a synthetic binary classification dataset.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    """
    Provides a synthetic regression dataset.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=0.2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ================================================================
# Unit tests: ModelEvaluationError
# ================================================================
def test_model_evaluation_error_logs(caplog):
    """
    Test that ModelEvaluationError logs the error message.

    Asserts:
        - The error message is present in the log.
    """
    with pytest.raises(ModelEvaluationError, match="custom error"):
        raise ModelEvaluationError("custom error")
    assert "custom error" in caplog.text


# ================================================================
# Unit tests: ModelEvaluator init
# ================================================================
def test_invalid_task_raises():
    """
    Test that ModelEvaluator raises an error for an invalid task.

    Asserts:
        - ModelEvaluationError is raised with the correct message.
    """
    with pytest.raises(ModelEvaluationError, match="Task must be"):
        ModelEvaluator(task="clustering")


def test_valid_initialization():
    """
    Test that ModelEvaluator initializes correctly for a valid task.

    Asserts:
        - The task attribute is set correctly.
        - The results_ attribute is initialized as an empty dict.
    """
    evaluator = ModelEvaluator(task="classification")
    assert evaluator.task == "classification"
    assert evaluator.results_ == {}


# ================================================================
# Integration tests: evaluate()
# ================================================================
def test_evaluate_classification(binary_classification_data):
    """
    Test that ModelEvaluator.evaluate works for classification.

    Asserts:
        - 'accuracy' and 'f1' are in the returned metrics.
        - The run_name is present in evaluator.results_.
    """
    X_train, X_test, y_train, y_test = binary_classification_data
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)

    evaluator = ModelEvaluator(task="classification")
    metrics = evaluator.evaluate(model, X_test, y_test, run_name="baseline")

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "baseline" in evaluator.results_


def test_evaluate_regression(regression_data):
    """
    Test that ModelEvaluator.evaluate works for regression.

    Asserts:
        - 'rmse' and 'r2' are in the returned metrics.
        - The run_name is present in evaluator.results_.
    """
    X_train, X_test, y_train, y_test = regression_data
    model = LinearRegression().fit(X_train, y_train)

    evaluator = ModelEvaluator(task="regression")
    metrics = evaluator.evaluate(model, X_test, y_test, run_name="linreg")

    assert "rmse" in metrics
    assert "r2" in metrics
    assert "linreg" in evaluator.results_


def test_evaluate_fails_with_invalid_model(binary_classification_data):
    """
    Test that ModelEvaluator.evaluate raises an error for an invalid model.

    Asserts:
        - ModelEvaluationError is raised with the correct message.
    """
    _, X_test, _, y_test = binary_classification_data
    evaluator = ModelEvaluator(task="classification")

    class Dummy:  # missing predict()
        pass

    with pytest.raises(ModelEvaluationError, match="Evaluation failed"):
        evaluator.evaluate(Dummy(), X_test, y_test, run_name="fail")


# ================================================================
# Unit tests: Visualization helpers
# (Only check they run without error, not actual plots)
# ================================================================
def test_plot_confusion_matrix_runs(binary_classification_data):
    """
    Test that plot_confusion_matrix runs without error.

    Asserts:
        - No exception is raised.
    """
    X_train, X_test, y_train, y_test = binary_classification_data
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluator = ModelEvaluator(task="classification")
    evaluator.plot_confusion_matrix(y_test, y_pred, labels=["No", "Yes"])


def test_plot_roc_curve_runs(binary_classification_data):
    """
    Test that plot_roc_curve runs without error.

    Asserts:
        - No exception is raised.
    """
    X_train, X_test, y_train, y_test = binary_classification_data
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    evaluator = ModelEvaluator(task="classification") 
    evaluator.plot_roc_curve(model, X_test, y_test)


def test_plot_precision_recall_runs(binary_classification_data):
    """
    Test that plot_precision_recall runs without error.

    Asserts:
        - No exception is raised.
    """
    X_train, X_test, y_train, y_test = binary_classification_data
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    evaluator = ModelEvaluator(task="classification")  
    evaluator.plot_precision_recall(model, X_test, y_test)


# ================================================================
# Integration tests: compare_runs()
# ================================================================
def test_compare_runs_success(binary_classification_data):
    """
    Test that compare_runs works and returns a DataFrame with expected indices.

    Asserts:
        - Both run names are present in the DataFrame index.
    """
    X_train, X_test, y_train, y_test = binary_classification_data
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)

    evaluator = ModelEvaluator(task="classification")
    evaluator.evaluate(model, X_test, y_test, run_name="baseline")

    # Fake second run
    evaluator.results_["with_features"] = {
        "accuracy": 0.9, "precision": 0.8, "recall": 0.75, "f1": 0.77
    }

    df = evaluator.compare_runs(metric="f1")
    assert "baseline" in df.index
    assert "with_features" in df.index


def test_compare_runs_fails_with_no_results():
    """
    Test that compare_runs raises an error if no results are present.

    Asserts:
        - ModelEvaluationError is raised with the correct message.
    """
    evaluator = ModelEvaluator(task="classification")
    with pytest.raises(ModelEvaluationError, match="No results"):
        evaluator.compare_runs(metric="f1")


def test_compare_runs_fails_with_invalid_metric(binary_classification_data):
    """
    Test that compare_runs raises an error if the metric is not found.

    Asserts:
        - ModelEvaluationError is raised with the correct message.
    """
    X_train, X_test, y_train, y_test = binary_classification_data
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)

    evaluator = ModelEvaluator(task="classification")
    evaluator.evaluate(model, X_test, y_test, run_name="baseline")

    with pytest.raises(ModelEvaluationError, match="Metric 'foobar' not found"):
        evaluator.compare_runs(metric="foobar")


def test_evaluate_with_plot_saving(tmp_path):
    """
    Test that ModelEvaluator can save plots to a specified path.

    Asserts:
        - The metrics dictionary contains 'rmse' and 'r2'.
        - The plot file is created and is not empty.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    # Create synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    model = LinearRegression().fit(X, y)

    # Initialize evaluator for regression task with save_dir
    save_dir = tmp_path / "plots"
    evaluator = ModelEvaluator(task="regression", save_dir=str(save_dir))

    # Run evaluate and plot, saving the plot
    metrics = evaluator.evaluate(model, X, y, run_name="linreg_test")
    plot_filename = "regression_results.png"

    # Use compare_runs to trigger plot saving
    evaluator.results_["another_run"] = {"rmse": 1.0, "r2": 0.5}
    evaluator.compare_runs(metric="rmse", filename=plot_filename)

    # Check that the metrics were generated
    assert "rmse" in metrics and "r2" in metrics

    # Check that the plot file was created and is not empty
    plot_path = save_dir / plot_filename
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0


def test_plot_confusion_matrix_saves_image(tmp_path, binary_classification_data):
    """
    Test that plot_confusion_matrix saves the image to the specified directory.

    Asserts:
        - The plot file is created and is not empty.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    binary_classification_data : tuple
        Fixture providing classification data.
    """
    X_train, X_test, y_train, y_test = binary_classification_data
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    save_dir = tmp_path / "plots"
    evaluator = ModelEvaluator(task="classification", save_dir=str(save_dir))
    plot_filename = "conf_matrix.png"

    evaluator.plot_confusion_matrix(y_test, y_pred, labels=["No", "Yes"], filename=plot_filename)

    plot_path = save_dir / plot_filename
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path)