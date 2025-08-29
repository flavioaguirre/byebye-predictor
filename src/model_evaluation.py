# ================================================================
#  src/model_evaluation.py
# ================================================================
# Unified module for evaluating ML models with metrics, visualizations,
# and comparative analysis across different pipelines (with/without 
# feature engineering, selection, hyperparameter tuning, and text features).
#
# - Classification & regression support
# - Scikit-learn compatible
# - Generates professional plots for reporting
#
# Author: Flavio Aguirre
# Date: 2025-08-29

# ================================================================

# --- Standard Library ---
from typing import Dict, Any, Optional, Union, List

# --- Third-Party Libraries ---
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, mean_squared_error, r2_score
)

# --- Local Modules ---
from src.utils import get_logger, add_project_root_to_path   # type: ignore
from src.data_loader import log_operation                   # type: ignore

# ================================================================
# Ensuring Project Root is in Path
# ================================================================
add_project_root_to_path()

# ================================================================
# Logger
# ================================================================
logger = get_logger(__name__)


# ================================================================
# Matplotlib warning suppression for non-interactive backends in test environments
# ================================================================
import warnings
if not matplotlib.is_interactive():
    warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")


# ================================================================
# Custom Exception
# ================================================================
class ModelEvaluationError(Exception):
    """
    Custom exception for errors raised during model evaluation.

    Parameters
    ----------
    message : str
        The error message to display.
    """
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(f"ModelEvaluationError: {message}")

# ================================================================
# Main ModelEvaluator Class
# ================================================================
class ModelEvaluator:
    """
    Provides tools for evaluating trained models, visualizing performance,
    and comparing different training strategies.

    Parameters
    ----------
    task : str, {'classification', 'regression'}
        Type of ML problem.
    save_dir : str or None, optional
        Directory to save generated plots. If None, plots are only shown and not saved.
    
    Attributes
    ----------
    task : str
        Stored task type for metrics and visualizations.
    results_ : dict
        Stores evaluation results per model/run.
    save_dir : str or None
        Directory where plots will be saved if provided.

    Raises
    ------
    ModelEvaluationError
        If the task is not 'classification' or 'regression'.
    """

    @log_operation
    def __init__(self, task: str, save_dir: Optional[str] = None):
        """
        Initializes the ModelEvaluator with the specified task and optional plot save directory.

        Parameters
        ----------
        task : str, {'classification', 'regression'}
            The type of machine learning problem.
        save_dir : str or None, optional
            Directory to save generated plots. If None, plots are only shown.

        Raises
        ------
        ModelEvaluationError
            If the task is not 'classification' or 'regression'.
        """
        if task not in ["classification", "regression"]:
            raise ModelEvaluationError("Task must be 'classification' or 'regression'.")
        self.task = task
        self.results_: Dict[str, Dict[str, Any]] = {}
        self.save_dir = save_dir

    # ================================================================
    # Core Evaluation
    # ================================================================
    @log_operation
    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        run_name: str
    ) -> Dict[str, Any]:
        """
        Evaluates a trained model on test data.

        Parameters
        ----------
        model : estimator
            Trained scikit-learn compatible model.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target.
        run_name : str
            Identifier for the run (e.g., "baseline_no_features").
        
        Returns
        -------
        dict
            Dictionary with evaluation metrics.

        Raises
        ------
        ModelEvaluationError
            If evaluation fails for any reason.
        """
        try:
            y_pred = model.predict(X_test)

            if self.task == "classification":
                report = classification_report(y_test, y_pred, output_dict=True)
                metrics = {
                    "accuracy": report["accuracy"],
                    "precision": report["weighted avg"]["precision"],
                    "recall": report["weighted avg"]["recall"],
                    "f1": report["weighted avg"]["f1-score"],
                }
            else:
                metrics = {
                    "rmse": mean_squared_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred)
                }

            self.results_[run_name] = metrics
            return metrics

        except Exception as e:
            raise ModelEvaluationError(f"Evaluation failed: {e}")

    # ================================================================
    # Visualization Helpers
    # ================================================================
    def _save_or_show(self, fig: plt.Figure, filename: Optional[str] = None):
        """
        Saves the plot to the configured directory if save_dir is set, otherwise shows the plot.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to save or show.
        filename : str or None
            The filename to use when saving the plot. If None, the plot is only shown.

        Notes
        -----
        If save_dir is set and filename is provided, the plot is saved as PNG in that directory.
        """
        if self.save_dir and filename:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, filename)
            fig.savefig(path, bbox_inches="tight")
            logger.info(f"Plot saved to {path}")
            plt.close(fig)
        else:
            plt.show()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], title: str = "Confusion Matrix", filename: Optional[str] = None):
        """
        Plots a confusion matrix for classification results and optionally saves it.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        labels : list of str
            List of class labels for axes.
        title : str, default="Confusion Matrix"
            Title for the plot.
        filename : str or None, optional
            If provided, saves the plot with this filename in save_dir.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        self._save_or_show(fig, filename)

    def plot_roc_curve(self, model, X_test, y_test, title: str = "ROC Curve", filename: Optional[str] = None):
        """
        Plots the ROC curve for a binary classifier and optionally saves it.

        Parameters
        ----------
        model : estimator
            Trained classifier with predict_proba method.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series or np.ndarray
            True labels.
        title : str, default="ROC Curve"
            Title for the plot.
        filename : str or None, optional
            If provided, saves the plot with this filename in save_dir.
        """
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        plt.tight_layout()
        self._save_or_show(fig, filename)

    def plot_precision_recall(self, model, X_test, y_test, title: str = "Precision-Recall Curve", filename: Optional[str] = None):
        """
        Plots the precision-recall curve for a binary classifier and optionally saves it.

        Parameters
        ----------
        model : estimator
            Trained classifier with predict_proba method.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series or np.ndarray
            True labels.
        title : str, default="Precision-Recall Curve"
            Title for the plot.
        filename : str or None, optional
            If provided, saves the plot with this filename in save_dir.
        """
        y_prob = model.predict_proba(X_test)[:,1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(recall, precision, lw=2, color="green")
        ax.set_title(title)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        plt.tight_layout()
        self._save_or_show(fig, filename)

    def compare_runs(self, metric: str = "f1", filename: Optional[str] = None):
        """
        Compare different runs (pipelines) visually and optionally save the plot.

        Parameters
        ----------
        metric : str
            Metric to compare across runs.
        filename : str or None, optional
            If provided, saves the plot with this filename in save_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with metrics for each run.

        Raises
        ------
        ModelEvaluationError
            If no results are available or the metric is not found.
        """
        if not self.results_:
            raise ModelEvaluationError("No results to compare. Run evaluate() first.")

        df = pd.DataFrame(self.results_).T
        if metric not in df.columns:
            raise ModelEvaluationError(f"Metric '{metric}' not found in results.")

        fig, ax = plt.subplots(figsize=(10,8))
        df[metric].plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
        ax.set_title(f"Comparison of runs by {metric.upper()}")
        ax.set_ylabel(metric.upper())
        plt.tight_layout()
        self._save_or_show(fig, filename)
        return df


# ================================================================
#  Example Usage
# ================================================================
if __name__ == "__main__":
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    # ================================================================
    #  STEP 1: LOAD OR SIMULATE DATA
    # ================================================================
    print("\n" + "=" * 60)
    print("### STEP 1: LOAD OR SIMULATE DATA ###")
    print("=" * 60)

    # Simulate a binary classification dataset (e.g., churn prediction)
    X_raw, y_raw = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.75, 0.25],  # imbalanced dataset
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"feature_{i}" for i in range(X_raw.shape[1])])    # type: ignore
    y = pd.Series(y_raw, name="churn")

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # ================================================================
    #  STEP 2: SPLIT DATA INTO TRAINING AND TEST SETS
    # ================================================================
    print("\n" + "=" * 60)
    print("### STEP 2: SPLIT DATA ###")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # ================================================================
    #  STEP 3: TRAIN AND SELECT MODELS
    # ================================================================
    print("\n" + "=" * 60)
    print("### STEP 3: TRAIN AND SELECT MODELS ###")
    print("=" * 60)

    from src.model_builder import ModelBuilder  # type: ignore

    builder = ModelBuilder(task="classification", random_state=42)
    builder.evaluate_models(X_train, y_train, cv=5)
    builder.select_best_model(metric="test_f1")
    best_model_name = builder.best_model_name_

    final_model = builder.train_final_model(best_model_name, X_train, y_train)
    print(f"Trained final model: {best_model_name}")

    # ================================================================
    #  STEP 4: EVALUATE MODELS
    # ================================================================
    print("\n" + "=" * 60)
    print("### STEP 4: EVALUATE MODELS ###")
    print("=" * 60)

    # Example: Save all generated graphs in the 'artifacts' directory (but in practice we will do it in reports/.)
    PLOTS_DIR = "./artifacts"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    evaluator = ModelEvaluator(task="classification", save_dir=PLOTS_DIR)

    # Evaluate the best model
    metrics = evaluator.evaluate(final_model, X_test, y_test, run_name="baseline_no_features")
    print("Evaluation metrics:", metrics)

    # ================================================================
    #  STEP 5: VISUALIZATIONS (AND SAVE PLOTS)
    # ================================================================
    print("\n" + "=" * 60)
    print("### STEP 5: VISUALIZATIONS (AND SAVE PLOTS) ###")
    print("=" * 60)

    y_pred = final_model.predict(X_test)

    # Save confusion matrix plot as 'confusion_matrix.png' in PLOTS_DIR
    evaluator.plot_confusion_matrix(
        y_test, y_pred, labels=["No Churn", "Churn"],
        title=f"Confusion Matrix - {best_model_name}",
        filename="confusion_matrix.png"
    )

    # Save ROC curve plot as 'roc_curve.png' in PLOTS_DIR
    evaluator.plot_roc_curve(
        final_model, X_test, y_test,
        title=f"ROC Curve - {best_model_name}",
        filename="roc_curve.png"
    )

    # Save Precision-Recall curve plot as 'precision_recall_curve.png' in PLOTS_DIR
    evaluator.plot_precision_recall(
        final_model, X_test, y_test,
        title=f"Precision-Recall Curve - {best_model_name}",
        filename="precision_recall_curve.png"
    )

    print(f"All plots saved in directory: {PLOTS_DIR}")

    # ================================================================
    #  STEP 6: COMPARE MULTIPLE RUNS (if available)
    # ================================================================
    print("\n" + "=" * 60)
    print("### STEP 6: COMPARE MULTIPLE RUNS ###")
    print("=" * 60)

    # Simulate another run (e.g., with additional features)
    evaluator.results_["with_reddit_features"] = {
        "accuracy": 0.82, "precision": 0.70, "recall": 0.65, "f1": 0.67
    }

    # Save comparison barplot as 'comparison_f1.png' in PLOTS_DIR
    comparison_df = evaluator.compare_runs(metric="f1", filename="comparison_f1.png")
    print("\nComparison of runs:\n", comparison_df)


