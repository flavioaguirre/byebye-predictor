# ================================================================
#   src/preprocess.py
# ================================================================
# Flexible data preprocessing module.

#   Provides a robust, configurable, and scalable preprocessing
#   architecture, designed for rapid experimentation and production
#   deployment.
#
#   Key Features:
#   - Flexibility: Strategies (imputation, scaling, etc.) are
#     configurable upon instantiation.
#   - Multi-Modal Support: Handles structured (numerical,
#     categorical) and text (NLP) data in a single pipeline.
#   - Modularity: Uses custom transformers that integrate natively
#     into scikit-learn pipelines.
#   - Robustness: State management (fit/transform), persistence, and
#     data validation.
#   - Professional Documentation: Comprehensive docstrings to facilitate
#     use and maintenance.

# It has:
# - Custom Exceptions:
#       - class PreprocessingError(Exception)

# - Custom Transformers (Pipeline Components):
#       - class OutlierCapper(BaseEstimator, TransformerMixin)
#       - class TextCleaner(BaseEstimator, TransformerMixin)

# - Main Preprocessor Class
#       - class DataPreprocessor

# - Example Usage

# ------------
#  Author: Flavio Aguirre
#  Date: 2025-08-13

# ================================================================
#  Imports
# ================================================================
# --- Standard Library ---
import os
import re
from typing import List, Optional, Union, Dict

# --- Third-Party Libraries ---
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Local Modules ---
from src.utils import get_logger, add_project_root_to_path
from src.data_loader import _validate_dataframe, log_operation

# Ensure project root is added to sys.path
add_project_root_to_path()

# ================================================================
#  Logger
# ================================================================
logger = get_logger(__name__)


# ================================================================
#  Custom Exceptions
# ================================================================
class PreprocessingError(Exception):
    """Base exception for errors during preprocessing."""


# ================================================================
#  Custom Transformers (Pipeline Components)
# ================================================================
class OutlierCapper(BaseEstimator, TransformerMixin):
    """A transformer to cap outliers using the interquartile range (IQR) method.

    Parameters
    ----------
    multiplier : float or dict, default=1.5
        The IQR multiplier. If a float, it's applied to all columns.
        If a dict, it allows specifying a multiplier per column.
    """
    def __init__(self, multiplier: Union[float, Dict[str, float]] = 1.5):
        if not isinstance(multiplier, (float, dict)):
            raise TypeError("Multiplier must be a float or a dictionary.")
        self.multiplier = multiplier
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OutlierCapper":
        """Calculates the lower and upper bounds for each column in X.

        Parameters
        ----------
        X : pd.DataFrame
            The training data DataFrame.
        y : pd.Series, optional
            Not used, present for scikit-learn compatibility.

        Returns
        -------
        OutlierCapper
            The fitted transformer instance.
        """
        for col in X.columns:
            # Using pandas' quantile is idiomatic and handles NaNs correctly.
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q3 - q1
            current_multiplier = (
                self.multiplier
                if isinstance(self.multiplier, float)
                else self.multiplier.get(col, 1.5)
            )
            self.bounds_[col] = {
                "lower": q1 - current_multiplier * iqr,
                "upper": q3 + current_multiplier * iqr,
            }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Caps the values in X to the bounds calculated during fit.

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with capped outliers.
        """
        X_copy = X.copy()
        for col, bounds in self.bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], bounds["lower"], bounds["upper"])
        return X_copy


class TextCleaner(BaseEstimator, TransformerMixin):
    """A transformer for basic text cleaning."""

    def fit(self, X: pd.Series, y: Optional[pd.Series] = None) -> "TextCleaner":
        """No training is needed, returns self."""
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Applies cleaning rules to a text Series.

        Parameters
        ----------
        X : pd.Series
            A Pandas Series containing text documents.

        Returns
        -------
        pd.Series
            The Series with cleaned text.
        """
        X_cleaned = X.fillna("").astype(str).str.lower()
        X_cleaned = X_cleaned.str.replace(r"https?://\S+|www\.\S+", "", regex=True)
        X_cleaned = X_cleaned.str.replace(r"[^a-z0-9\s]", "", regex=True)
        X_cleaned = X_cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
        return X_cleaned


# ================================================================
#  Main Preprocessor Class
# ================================================================
class DataPreprocessor:
    """
    The complete, intelligent and configurable preprocessing workflow is encapsulated.
    """
    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        text_cols: Optional[List[str]] = None,
        id_cols: Optional[List[str]] = None,
        numeric_imputer: BaseEstimator = SimpleImputer(strategy="median"),
        scaler: Union[BaseEstimator, str] = StandardScaler(),
        outlier_handler: Optional[BaseEstimator] = OutlierCapper(),
        categorical_imputer: BaseEstimator = SimpleImputer(strategy="most_frequent"),
        encoder: BaseEstimator = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ),
        text_vectorizer: BaseEstimator = TfidfVectorizer(max_features=1000),
        ):
        """Initializes the preprocessor with strategies and optional column lists.

        If column lists (numeric_cols, categorical_cols, text_cols) are None,
        they will be auto-detected during the `fit` method.

        Parameters
        ----------
        numeric_cols : list of str, optional
            List of names for numerical columns. If None, auto-detects.
        categorical_cols : list of str, optional
            List of names for categorical columns. If None, auto-detects.
        text_cols : list of str, optional
            List of names for text columns. If None, auto-detects.
        id_cols : list of str, optional
            List of ID columns to be ignored by auto-detection and passed through.
        numeric_imputer : BaseEstimator, default=SimpleImputer(strategy="median")
            Imputation strategy for numerical features.
        scaler : BaseEstimator or str, default=StandardScaler()
            Scaling strategy. Use 'passthrough' to disable.
        outlier_handler : BaseEstimator or None, default=OutlierCapper()
            Strategy for handling outliers. Use None to disable.
        categorical_imputer : BaseEstimator, default=SimpleImputer(strategy="most_frequent")
            Imputation strategy for categorical features.
        encoder : BaseEstimator, default=OneHotEncoder(...)
            Encoding strategy for categorical features.
        text_vectorizer : BaseEstimator, default=TfidfVectorizer(...)
            Strategy for vectorizing text columns.

        Notes
        -----
        The default OneHotEncoder uses `sparse_output=False` to return a dense
        array, which is convenient for DataFrame creation. Older scikit-learn
        versions might not support this argument and would return a sparse matrix.
        """
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.text_cols = text_cols
        self.id_cols = id_cols or []
        self.strategies = {
            "numeric_imputer": numeric_imputer,
            "scaler": scaler,
            "outlier_handler": outlier_handler,
            "categorical_imputer": categorical_imputer,
            "encoder": encoder,
            "text_vectorizer": text_vectorizer,
        }
        self.pipeline: Optional[ColumnTransformer] = None
        self.is_fitted: bool = False

    def _auto_detect_column_types(self, X: pd.DataFrame) -> None:
        """[Private] Intelligently detects column types based on dtype and content."""
        logger.info("Auto-detecting column types...")
        self.numeric_cols, self.categorical_cols, self.text_cols = [], [], []

        potential_cols = X.drop(columns=self.id_cols, errors="ignore").columns

        for col in potential_cols:
            dtype = X[col].dtype
            unique_count = X[col].nunique()

            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
                dtype
            ):
                # Treat low-cardinality numerics (e.g., ratings 1-5, binary 0/1) as
                # CATEGORICAL. This is a key decision for correct preprocessing.
                if unique_count > 1 and unique_count < 20:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            elif pd.api.types.is_object_dtype(
                dtype
            ) or pd.api.types.is_categorical_dtype(dtype):
                # Distinguish TEXT from CATEGORICAL based on string length and unique values.
                avg_len = X[col].astype(str).str.len().mean()
                if unique_count > 2 and avg_len > 35:
                    self.text_cols.append(col)
                else:
                    self.categorical_cols.append(col)

        logger.info(f"Detected Numerical Columns: {self.numeric_cols}")
        logger.info(f"Detected Categorical Columns: {self.categorical_cols}")
        logger.info(f"Detected Text Columns: {self.text_cols}")

    def _build_pipeline(self) -> ColumnTransformer:
        """
        [Private] Builds the ColumnTransformer pipeline based on the configuration.
        """
        numeric_steps = [
            s
            for s in [
                ("imputer", self.strategies["numeric_imputer"]),
                ("outlier_capper", self.strategies["outlier_handler"]),
                ("scaler", self.strategies["scaler"]),
            ]
            if s[1] is not None and s[1] != "passthrough"
        ]
        numeric_transformer = (
            Pipeline(steps=numeric_steps) if numeric_steps else "passthrough"
        )

        categorical_steps = [
            s
            for s in [
                ("imputer", self.strategies["categorical_imputer"]),
                ("encoder", self.strategies["encoder"]),
            ]
            if s[1] is not None
        ]
        categorical_transformer = (
            Pipeline(steps=categorical_steps) if categorical_steps else "passthrough"
        )

        transformers = [
            ("numerical", numeric_transformer, self.numeric_cols),
            ("categorical", categorical_transformer, self.categorical_cols),
        ]

        if self.text_cols:
            for col in self.text_cols:
                text_pipeline = Pipeline(
                    [
                        ("cleaner", TextCleaner()),
                        ("vectorizer", self.strategies["text_vectorizer"]),
                    ]
                )
                transformers.append((f"text_{col}", text_pipeline, [col]))

        return ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

    @log_operation
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        _validate_dataframe(X)
        if (
            self.numeric_cols is None
            and self.categorical_cols is None
            and self.text_cols is None
        ):
            self._auto_detect_column_types(X)

        try:
            self.pipeline = self._build_pipeline()
            self.pipeline.fit(X, y)
            self.is_fitted = True
            logger.info("Preprocessor fitted successfully.")
        except Exception as e:
            raise PreprocessingError(f"An error occurred during fitting: {e}")
        return self

    @log_operation
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted or self.pipeline is None:
            raise PreprocessingError(
                "The preprocessor must be fitted successfully before transforming the data."
                "Make sure to call .fit(training_data) first."
        )
        _validate_dataframe(X)
        
        X_transformed = self.pipeline.transform(X)
        feature_names = self.pipeline.get_feature_names_out()
        
        return pd.DataFrame(X_transformed, index=X.index, columns=feature_names)

    @log_operation
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @log_operation
    def save(self, file_path: str) -> None:
        if not self.is_fitted:
            raise PreprocessingError("Only a fitted preprocessor can be saved.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Preprocessor saved to {file_path}")

    @staticmethod
    @log_operation
    def load(file_path: str) -> "DataPreprocessor":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor file not found at {file_path}")
        preprocessor = joblib.load(file_path)
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor


# ================================================================
#  Example Usage
# ================================================================
if __name__ == "__main__":
    # --- Create a comprehensive sample DataFrame ---
    data = pd.DataFrame(
        {
            "CustomerID": [f"CUST-{i}" for i in range(10)],
            "Tenure": [1, 12, 24, 3, 5, 60, 34, 22, 5, 8],
            "TotalCharges": [
                29.8,
                1889.5,
                108.1,
                1840.7,
                151.6,
                820.5,
                346.4,
                1532.0,
                30000.0,
                588.4,
            ],
            "IsSenior": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],  # Low-cardinality numeric
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "Month-to-month",
                "Two year",
                "One year",
                "Month-to-month",
                "One year",
                "Month-to-month",
            ],
            "SatisfactionScore": [
                "High",
                "Low",
                "Medium",
                "High",
                "Low",
                np.nan,
                "Medium",
                "High",
                "High",
                "Low",
            ],
            "ReviewComment": [
                "great service, very happy!",
                "terrible experience, cancelling my contract http://bad.com",
                "average price and good support, i think i will stay",
                "their APP is SO BAD 1234, really frustrating experience",
                "no complaints so far",
                "excellent!! will renew for sure, top quality",
                "meh, it is okay i guess, not the best not the worst",
                "I will probably churn next month due to poor connection",
                "very expensive for what it is offering to the customer base",
                "good value",
            ],
        }
    )

    print("\n" + "=" * 60)
    print("### SCENARIO 1: Manual Column Specification (For Linear Models) ###")
    print("=" * 60)
    preprocessor_manual = DataPreprocessor(
        numeric_cols=["Tenure", "TotalCharges"],
        categorical_cols=["Contract", "SatisfactionScore", "IsSenior"],
        text_cols=["ReviewComment"],
        id_cols=["CustomerID"],
    )
    processed_manual = preprocessor_manual.fit_transform(data)
    print("DataFrame processed with manual column definition:")
    print(processed_manual.head())
    print(f"Shape: {processed_manual.shape}\n")

    print("\n" + "=" * 60)
    print("### SCENARIO 2: Automatic Column Detection (Intelligent Mode) ###")
    print("=" * 60)
    # Instantiate without providing column lists to trigger auto-detection
    preprocessor_auto = DataPreprocessor(id_cols=["CustomerID"])
    processed_auto = preprocessor_auto.fit_transform(data)
    print("DataFrame processed with automatic column detection:")
    print(processed_auto.head())
    print(f"Shape: {processed_auto.shape}\n")

    print("\n" + "=" * 60)
    print("### SCENARIO 3: Persistence (Save and Load) ###")
    print("=" * 60)
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    FILE_PATH = "artifacts/preprocessor.joblib"

    print(f"Saving the auto-detected preprocessor to: {FILE_PATH}")
    preprocessor_auto.save(FILE_PATH)

    print("Loading the preprocessor from file...")
    loaded_preprocessor = DataPreprocessor.load(FILE_PATH)

    new_data = data.sample(n=3).copy()
    print("\nTransforming new data with the loaded preprocessor:")
    new_data_transformed = loaded_preprocessor.transform(new_data)
    print(new_data_transformed)
    print(f"\nDimensions of new transformed data: {new_data_transformed.shape}")