# ================================================================
#  src/model_builder.py
# ================================================================
# Unified module for training, evaluating, and managing ML models.
#
#  - Scikit-learn and XGBoost compatibility
#  - DRY utilities for logging and persistence
#  - Supports classification & regression
#  - Clean, modular and industry-ready
#
#  Author: Flavio Aguirre
#  Date: 2025-08-28


# ================================================================
# Importing Libraries
# ================================================================
# --- Standard Library ---
from typing import Dict, Any, Optional, Union


# --- Third-Party Libraries ---
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score, make_scorer
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

# --- Local Modules ---
from src.utils import get_logger, add_project_root_to_path # type: ignore
from src.data_loader import log_operation   # type: ignore



# ================================================================
# Ensuring Project Root is in Path
# ================================================================
add_project_root_to_path()

# ================================================================
#   Logger
# ================================================================
logger = get_logger(__name__)


# ================================================================
#  Custom Exception
# ================================================================
class ModelBuilderError(Exception):
    """
    Custom exception for errors raised by the ModelBuilder class.

    Parameters
    ----------
    message : str
        The error message to display.
    """
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(f"ModelBuilderError: {message}")

# ================================================================
#  Main ModelBuilder Class
# ================================================================
class ModelBuilder:
    """
    Orchestrates machine learning model training, evaluation, selection, and persistence.

    Parameters
    ----------
    task : str, {'classification', 'regression'}
        Type of problem to solve.
    random_state : int, default=42
        Controls reproducibility of algorithms.
    custom_models : dict, optional
        A dictionary of custom models to use instead of the defaults.

    Attributes
    ----------
    models : dict
        Dictionary of model instances to be evaluated and trained.
    results_ : dict
        Stores cross-validation results for each model.
    trained_models_ : dict
        Stores trained model instances.
    best_model_name_ : str or None
        Name of the best model selected.
    best_model_ : estimator or None
        The best model instance selected.
    """

    @log_operation
    def __init__(self, task: str, random_state: int = 42, custom_models: Optional[Dict[str, Any]] = None):
        """
        Initializes the ModelBuilder with the specified task, random state, and optional custom models.

        Parameters
        ----------
        task : str, {'classification', 'regression'}
            The type of machine learning problem.
        random_state : int, default=42
            Random seed for reproducibility.
        custom_models : dict, optional
            Dictionary of custom models to use.
        
        Raises
        ------
        ModelBuilderError
            If the task is not 'classification' or 'regression'.
        """
        if task not in ["classification", "regression"]:
            raise ModelBuilderError("Task must be 'classification' or 'regression'.")
        self.task = task
        self.random_state = random_state
        
        # Allow injection of custom models for flexibility.
        self.models = custom_models if custom_models is not None else self._init_models()
        
        self.results_: Dict[str, Dict[str, float]] = {}
        self.trained_models_: Dict[str, Any] = {}
        self.best_model_name_: Optional[str] = None
        self.best_model_: Optional[Any] = None

    def _init_models(self) -> Dict[str, Any]:
        """
        Initializes a default set of models based on the task type.

        Returns
        -------
        dict
            Dictionary of model names mapped to their instances.
        """
        # MENTOR: Para un problema de churn, es buena práctica empezar con class_weight='balanced'.
        # Esto es un buen ejemplo de por qué la flexibilidad es clave.
        if self.task == "classification":
            return {
                "logistic_regression": LogisticRegression(random_state=self.random_state, max_iter=1000, class_weight='balanced'),
                "decision_tree": DecisionTreeClassifier(random_state=self.random_state),
                "random_forest": RandomForestClassifier(random_state=self.random_state),
                "xgboost": xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
            }
        else:  # regression
            return {
                "linear_regression": LinearRegression(),
                "polynomial_regression": Pipeline([
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("linreg", LinearRegression())
                ]),
                "decision_tree": DecisionTreeRegressor(random_state=self.random_state),
                "random_forest": RandomForestRegressor(random_state=self.random_state),
                "xgboost": xgb.XGBRegressor(random_state=self.random_state)
            }

    @log_operation
    def evaluate_models(self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluates all models using cross-validation on the training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        cv : int, default=5
            Number of folds for cross-validation.

        Returns
        -------
        dict
            Mean evaluation results per model from cross-validation.
        """
        # MENTOR: Definimos las métricas a usar en cross-validation
        scoring = self._get_scoring_metrics()
        
        for name, model in self.models.items():
            try:
                logger.info(f"Cross-validating model: {name}...")
                # MENTOR: cross_validate es más potente que cross_val_score, devuelve múltiples métricas
                cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                
                # MENTOR: Guardamos la media de cada métrica
                self.results_[name] = {metric: np.mean(scores) for metric, scores in cv_results.items()}  # type: ignore
                logger.info(f"Model {name} evaluated successfully.")
                
            except Exception as e:
                logger.warning(f"Skipping {name} during evaluation due to error: {e}")

        return self.results_
    
    @log_operation
    def train_final_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Trains a single, specified model on the entire training dataset.
        
        Parameters
        ----------
        model_name : str
            The name of the model to train (must be in self.models).
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
            
        Returns
        -------
        estimator
            The trained model.

        Raises
        ------
        ModelBuilderError
            If the specified model name is not found in self.models.
        """
        if model_name not in self.models:
            raise ModelBuilderError(f"Model '{model_name}' not found.")
        
        logger.info(f"Training final model: {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models_[model_name] = model
        logger.info(f"Model {model_name} trained successfully.")
        return model

    def _get_scoring_metrics(self) -> Union[Dict[str, Any], str]:
        """
        Returns scoring metrics based on the task.

        Returns
        -------
        dict or str
            Scoring metrics for cross-validation.
        """
        if self.task == "classification":
            # MENTOR: Para churn, F1, Recall y ROC-AUC suelen ser más importantes que Accuracy
            return {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }
        else: # regression
            return {
                'r2': 'r2',
                'neg_root_mean_squared_error': 'neg_root_mean_squared_error'
            }

    @log_operation
    def select_best_model(self, metric: Optional[str] = None) -> Any:
        """
        Selects the best model based on cross-validation results.

        Parameters
        ----------
        metric : str, optional
            Metric to use for selection. If None, defaults to 'test_f1' (classification)
            or 'test_r2' (regression). Note: cross_validate adds 'test_' prefix.

        Returns
        -------
        estimator
            The best performing model (untrained instance).

        Raises
        ------
        ModelBuilderError
            If no models have been evaluated or the metric is not found.
        """
        if not self.results_:
            raise ModelBuilderError("No models evaluated yet. Call evaluate_models first.")

        # MENTOR: cross_validate agrega el prefijo 'test_' a las métricas.
        default_metric = "test_f1" if self.task == "classification" else "test_r2"
        metric = metric or default_metric
        
        # MENTOR: Filtrar modelos que pudieron haber fallado en la evaluación
        valid_results = {name: res for name, res in self.results_.items() if metric in res}
        if not valid_results:
            raise ModelBuilderError(f"Metric '{metric}' not found in any model results.")

        # MENTOR: El valor de R2 debe maximizarse, mientras que neg_rmse también (es negativo).
        # Para otros errores (si los usaras), tendrías que minimizar (usar min()).
        self.best_model_name_ = max(valid_results, key=lambda m: valid_results[m].get(metric, -np.inf))
        
        self.best_model_ = self.models[self.best_model_name_]       # type: ignore
        
        logger.info(f"Best model selected: {self.best_model_name_} with {metric}: {valid_results[self.best_model_name_].get(metric):.4f}")   # type: ignore
        return self.best_model_

    @log_operation
    def save_model(self, name: str, filepath: str) -> None:
        """
        Saves a trained model to disk.

        Parameters
        ----------
        name : str
            Name of the trained model to save.
        filepath : str
            Path to save the model file.

        Raises
        ------
        ModelBuilderError
            If the model has not been trained.
        """
        if name not in self.trained_models_:
            raise ModelBuilderError(f"Trained model '{name}' not found. Train the model first.")
        joblib.dump(self.trained_models_[name], filepath)
        logger.info(f"Model {name} saved to {filepath}")

    @staticmethod
    @log_operation
    def load_model(filepath: str) -> Any:
        """
        Loads a trained model from file.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.

        Returns
        -------
        estimator
            The loaded trained model.
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model

# ================================================================
#  Example Usage for ModelBuilder
# ================================================================
if __name__ == "__main__":
    # Example usage for the ModelBuilder class.
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # ================================================================
    #  STEP 1: LOAD AND PREPARE DATA
    # ================================================================
    # In a real project, this would be replaced by your data loading and
    # preprocessing pipeline (e.g., using your FeatureEngineer).
    # For this example, we simulate a clean DataFrame.
    print("\n" + "=" * 60)
    print("### STEP 1: LOAD AND PREPARE DATA ###")
    print("=" * 60)
    
    # We use a larger example dataset so that metrics are meaningful.
    from sklearn.datasets import make_classification
    X_raw, y_raw = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.8, 0.2], # Simulate an imbalanced churn problem
        flip_y=0.05,
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f'feature_{i}' for i in range(X_raw.shape[1])])    # type: ignore
    y = pd.Series(y_raw, name='churn')
    print(f"Simulated data generated. Shape of X: {X.shape}, Distribution of y:\n{y.value_counts(normalize=True)}")
    
    # ================================================================
    #  STEP 2: SPLIT DATA INTO TRAINING AND TEST SETS
    # ================================================================
    # This is a critical step that is now outside of ModelBuilder.
    # We split the data ONCE. The test set (X_test, y_test) is held out
    # and not touched until final evaluation.
    print("\n" + "=" * 60)
    print("### STEP 2: SPLIT DATA INTO TRAINING AND TEST SETS ###")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # stratify is key for imbalanced data
    )
    print(f"Data split into Train ({X_train.shape[0]} samples) and Test ({X_test.shape[0]} samples).")
    
    # ================================================================
    #  STEP 3: EVALUATE MODEL CANDIDATES USING CROSS-VALIDATION
    # ================================================================
    # We create the builder instance and use the TRAINING set to find the best model type.
    print("\n" + "=" * 60)
    print("### STEP 3: EVALUATE MODEL CANDIDATES (ON TRAINING DATA) ###")
    print("=" * 60)
    
    builder = ModelBuilder(task='classification', random_state=42)
    
    # The method uses cross-validation internally on X_train, y_train
    cv_results = builder.evaluate_models(X_train, y_train, cv=5)
    
    # Display results in a sorted DataFrame
    results_df = pd.DataFrame(cv_results).T # .T transposes to have models as rows
    print("Cross-Validation Results (mean of 5 folds):")
    print(results_df[['test_f1', 'test_roc_auc', 'test_recall', 'test_precision']].sort_values(by='test_f1', ascending=False))  # type: ignore
    
    # ================================================================
    #  STEP 4: SELECT BEST MODEL AND TRAIN IT ON FULL TRAINING DATA
    # ================================================================
    # Based on CV results, we select the best model and retrain it on the entire training set.
    print("\n" + "=" * 60)
    print("### STEP 4: SELECT AND TRAIN FINAL MODEL ###")
    print("=" * 60)
    
    # Select the best model based on F1-score
    builder.select_best_model(metric='test_f1')
    best_model_name = builder.best_model_name_
    print(f"The best model according to F1-score is: '{best_model_name}'")
    
    # Train that specific model on the full training set
    final_model = builder.train_final_model(best_model_name, X_train, y_train)
    print(f"Final model '{best_model_name}' trained with {X_train.shape[0]} samples.")
    
    # ================================================================
    #  STEP 5: FINAL EVALUATION ON THE HELD-OUT TEST SET
    # ================================================================
    # Now we use the test set, which the model has never seen, to get an unbiased
    # report of its real-world performance.
    print("\n" + "=" * 60)
    print("### STEP 5: FINAL EVALUATION ON TEST SET ###")
    print("=" * 60)
    
    y_pred = final_model.predict(X_test)
    print(f"Final classification report for '{best_model_name}':")
    print(classification_report(y_test, y_pred))
    
    # ================================================================
    #  STEP 6: PERSIST (SAVE) AND LOAD THE FINAL MODEL
    # ================================================================
    # The last step is to save our trained model for later use in an API, batch job, etc.
    print("\n" + "=" * 60)
    print("### STEP 6: SAVE AND LOAD THE FINAL MODEL ###")
    print("=" * 60)
    
    MODEL_DIR = "./models"
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{best_model_name}_churn_v1.joblib")
    
    # Save the trained model
    builder.save_model(best_model_name, model_path)
    
    # Load the model from file (using the static method)
    loaded_model = ModelBuilder.load_model(model_path)
    print(f"Loaded model: {loaded_model}")
    
    # Verify that the loaded model works
    sample_customer = X_test.head(1)
    prediction = loaded_model.predict(sample_customer)[0]
    prediction_proba = loaded_model.predict_proba(sample_customer)[0]
    
    print("\n--- Test with loaded model ---")
    print(f"Prediction for a sample customer: {'Churn' if prediction == 1 else 'No Churn'}")
    print(f"Churn probability: {prediction_proba[1]:.2%}")
    print("=" * 60)