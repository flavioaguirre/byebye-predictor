# ================================================================
#   src/feature_engineering.py
# ================================================================
# Flexible module for creating and selecting new features.
#
#   This module provides a robust, configurable, and scalable
#   architecture for feature engineering, designed to complement
#   the preprocessing pipeline.
#
#   Key Features:
#   - Modularity: Custom transformers for specific engineering tasks
#     (e.g., polynomial features, date features).
#   - Scikit-learn Compatibility: Integrates seamlessly into a
#     scikit-learn pipeline via ColumnTransformer.
#   - Multi-Modal Support: Handles numerical, categorical, and text
#     features.
#   - Configurability: Allows for the selection and configuration
#     of specific feature engineering strategies upon instantiation.
#   - Professional Documentation: Comprehensive docstrings for clarity.

# ------------
#   Author: Flavio Aguirre
#   Date: 2025-08-27

# ================================================================
# Importing Libraries
# ================================================================
# --- Standard Library ---
import os
import re
from typing import List, Optional, Any, Literal, Callable 

# --- Third-Party Libraries ---
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    OneHotEncoder,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer

# --- Local Modules ---
from utils import get_logger, add_project_root_to_path  
from data_loader import _validate_dataframe, log_operation  
from preprocess import DataProcessor, clean_column_names  


# ================================================================
# Ensuring Project Root is in Path
# ================================================================
add_project_root_to_path()


# ================================================================
#   Logger
# ================================================================
logger = get_logger(__name__)


# ================================================================
#   Custom Exceptions
# ================================================================
class FeatureEngineeringError(Exception):
    """
    Base exception for errors occurring during feature engineering.
    """
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(f"FeatureEngineeringError: {message}")


# ================================================================
#   Helper Function
# ================================================================
def apply_feature_engineering(
    X: pd.DataFrame,
    transformers: List[tuple],
    ) -> pd.DataFrame:
    """
    Aplica un conjunto de transformadores de features a un dataframe(X).
    """
    X_final = X

    for name, transformer in transformers:
        new_features = transformer.transform(X)
        # Aseguramos mantener mismo índice que X
        new_features.index = X.index
        X_final = pd.concat([X_final, new_features], axis=1)
        logger.info(f"Applied transformer '{name}', new shape: {X_final.shape}")
    return X_final


# ================================================================
#   Custom Transformers for Feature Engineering
# ================================================================
class DatetimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts features from datetime columns (e.g., day of week, month, year, day since).

    This transformer is specifically designed to create features relevant to churn analysis,
    such as the number of days since the start of the subscription.

    Parameters
    ----------
    features_to_extract : list, optional
        List of datetime features to extract.
        Options: 'year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter', 'is_month_start', 'is_month_end'.
    reference_date : str or pd.Timestamp, optional
        The reference date to calculate 'days_since'. Defaults to today's date.
    """
    def __init__(self, features_to_extract: Optional[List[str]] = None, reference_date: Optional[Any] = None):
        self.features_to_extract = features_to_extract if features_to_extract is not None else [
            'year', 'month', 'day', 'dayofweek'
        ]
        self.reference_date = pd.to_datetime(reference_date) if reference_date else pd.Timestamp.now()
        self._output_feature_names = None

    def fit(self, X: pd.DataFrame, y=None) -> "DatetimeFeatureTransformer":
        """
        Fits the transformer. This method is stateless for now, but pre-calculates feature names.
        """
        self._output_feature_names = [] # type: ignore
        for col in X.columns:
            for feat in self.features_to_extract:
                self._output_feature_names.append(f'{col}_{feat}') # type: ignore
            self._output_feature_names.append(f'{col}_days_since') # type: ignore
        logger.info(f"Configured DatetimeFeatureTransformer for columns: {X.columns.tolist()}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms datetime columns by extracting new features.
        """
        X_transformed = pd.DataFrame(index=X.index)
        for col in X.columns:
            try:
                dates = pd.to_datetime(X[col], errors='coerce', format="%Y-%m-%d")
                for feat in self.features_to_extract:
                    if hasattr(dates.dt, feat):
                        X_transformed[f'{col}_{feat}'] = getattr(dates.dt, feat)
                    else:
                        logger.warning(f"Feature '{feat}' not found for column '{col}'. Skipping.")
                X_transformed[f'{col}_days_since'] = (self.reference_date - dates).dt.days
            except Exception as e:
                raise FeatureEngineeringError(f"Error processing datetime column '{col}': {e}") from e
        return X_transformed.to_numpy() # type: ignore

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Gets the names of the output features.
        """
        if self._output_feature_names is None:
            raise FeatureEngineeringError("Call fit() before get_feature_names_out().")
        return self._output_feature_names


class CustomCombinationTransformer(BaseEstimator, TransformerMixin):
    """
    A generic transformer to apply a custom function for feature creation.
    
    This is useful for business-specific logic that doesn't fit a standard pattern.

    Parameters
    ----------
    feature_function : callable
        A function that takes a pandas DataFrame and returns a new pandas Series or DataFrame
        of the new feature(s).
    feature_names : list of str
        Names for the new features created by the function.
    """
    def __init__(self, feature_function: callable, feature_names: List[str]):   # type: ignore
        if not callable(feature_function):
            raise TypeError("feature_function must be a callable.")
        if not isinstance(feature_names, list) or not all(isinstance(n, str) for n in feature_names):
            raise TypeError("feature_names must be a list of strings.")
        self.feature_function = feature_function
        self.feature_names = feature_names
    
    def fit(self, X, y=None) -> "CustomCombinationTransformer":
        """
        Fit method for compatibility. This transformer is stateless.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the custom function to create new features.
        """
        try:
            new_features = self.feature_function(X)
            if isinstance(new_features, pd.Series):
                new_features = new_features.to_frame(self.feature_names[0])
            elif not isinstance(new_features, pd.DataFrame):
                new_features = pd.DataFrame(new_features, columns=self.feature_names, index=X.index)    # type: ignore
            if new_features.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Custom function returned {new_features.shape[1]} features, but {len(self.feature_names)} names were provided."
                )
            return new_features
        except Exception as e:
            raise FeatureEngineeringError(f"Error applying custom feature function: {e}") from e

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Returns the names of the new features.
        """
        return self.feature_names


class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Applies TF-IDF vectorization to text data.

    This transformer is critical for analyzing user comments and sentiment, which are
    strong predictors of churn.

    Parameters
    ----------
    max_features : int, optional, default=None
        Maximum number of features (terms) to build from the text.
    """
    def __init__(self, max_features: Optional[int] = None):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        self._output_feature_names = None

    def fit(self, X: pd.DataFrame, y=None) -> "TextFeatureTransformer":
        """
        Fits the TfidfVectorizer on the text data.
        """
        if X.shape[1] > 1:
            logger.warning("TextFeatureTransformer is designed for a single column. Only the first column will be used.")
        text_series = X.iloc[:, 0].fillna('')
        self.vectorizer.fit(text_series)
        self._output_feature_names = self.vectorizer.get_feature_names_out()    # type: ignore
        logger.info(f"Fitted TextFeatureTransformer, generating {len(self._output_feature_names)} features.")   # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms text data using the fitted vectorizer.
        """
        if self._output_feature_names is None:
            raise FeatureEngineeringError("TextFeatureTransformer has not been fitted yet.")
        text_series = X.iloc[:, 0].fillna('')
        transformed_data = self.vectorizer.transform(text_series)
        return transformed_data.toarray()
        
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Gets the names of the output features.
        """
        if self._output_feature_names is None:
            raise FeatureEngineeringError("Call fit() before get_feature_names_out().")
        return list(self._output_feature_names)


class FeatureSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper para Feature Selection en un DataFrame.

    Esta clase aplica el feature_selector definido (ej. SelectKBest, RFECV o un Pipeline
    que combine ambos) y devuelve un DataFrame reducido con las features seleccionadas.

    Parameters
    ----------
    feature_selector : sklearn selector o pipeline
        Cualquier transformador compatible con scikit-learn que implemente
        fit(X, y) y transform(X).
    """

    def __init__(self, feature_selector):
        self.feature_selector = feature_selector
        self._feature_names_out = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.feature_selector.fit(X, y)

        # Caso: selector directo (ej. SelectKBest)
        if hasattr(self.feature_selector, "get_support"):
            mask = self.feature_selector.get_support()
            self._feature_names_out = X.columns[mask].tolist()

        # Caso: pipeline (ej. SelectKBest + RFECV)
        elif hasattr(self.feature_selector, "steps"):
            names = X.columns
            for step_name, step in self.feature_selector.steps:
                X = step.transform(X)  # transformar paso a paso
                if hasattr(step, "get_support"):
                    mask = step.get_support()
                    names = names[mask] if hasattr(names, "__getitem__") else [f"f{i}" for i, keep in enumerate(mask) if keep]
            self._feature_names_out = list(names)

        else:
            # fallback: nombres genéricos
            self._feature_names_out = [
                f"f{i}" for i in range(self.feature_selector.transform(X).shape[1])
            ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_trans = self.feature_selector.transform(X)
        return pd.DataFrame(
            X_trans,
            columns=self._feature_names_out,
            index=X.index
        )

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_selected_features(self, model=None):
        """
        Retorna un DataFrame con las features seleccionadas y,
        opcionalmente, su importancia/coeficiente.
        """
        features = pd.DataFrame({"feature": self._feature_names_out})
        
        if model is not None:
            if hasattr(model, "coef_"):  # Modelos lineales
                importances = model.coef_.ravel()
            elif hasattr(model, "feature_importances_"):  # Árboles/ensambles
                importances = model.feature_importances_
            else:
                importances = None
            
            if importances is not None:
                features["importance"] = importances
                features = features.sort_values("importance", ascending=False)
        
        return features.reset_index(drop=True)

    def save(self, filepath: str) -> None:
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FeatureEngineer":
        return joblib.load(filepath)


# ================================================================
#   Example Usage
# ================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LOAD AND PREPROCESS DATA FOR FEATURE ENGINEERING DEMO")
    print("=" * 60)

    data = pd.DataFrame(
        {
            "CustomerID": [f"CUST-{i}" for i in range(10)],
            "Tenure": [1, 12, 24, 3, 5, 60, 34, 22, 5, 8],
            "TotalCharges": [
                29.8, 1889.5, 108.1, 1840.7, 151.6,
                820.5, 346.4, 1532.0, 30000.0, 588.4
            ],
            "IsSenior": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            "Contract": [
                "Month-to-month", "One year", "Two year", "Month-to-month",
                "Month-to-month", "Two year", "One year", "Month-to-month",
                "One year", "Month-to-month"
            ],
            "SatisfactionScore": [
                "High", "Low", "Medium", "High", "Low",
                "None", "Medium", "High", "High", "Low"
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
            "JoinDate": pd.to_datetime([
                "2020-01-15", "2019-02-20", "2018-03-01", "2022-04-10",
                "2023-05-05", "2015-06-12", "2017-07-25", "2018-08-30",
                "2023-09-01", "2022-10-14"
            ]),
            "Churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
        }
    )

    # ================================================================
    #   Preprocess data
    # ================================================================
    preprocessor = DataProcessor()
    processed_df = preprocessor.process(data)

    # ================================================================
    #   Pipeline 1: Feature Engineering only
    # ================================================================
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Custom feature
    def calculate_cost_per_tenure(df: pd.DataFrame) -> pd.Series:
        """Calculates the average monthly cost for each customer."""
        # We look for the days column from joindate
        days_candidates = [col for col in df.columns if "joindate" in col and "days" in col]
        if not days_candidates:
            raise FeatureEngineeringError("No 'days since join' feature found in dataframe.")
        days_col = days_candidates[0]

        # We look for the total charges column
        charges_candidates = [col for col in df.columns if "totalcharges" in col]
        if not charges_candidates:
            raise FeatureEngineeringError("No 'totalcharges' feature found in dataframe.")
        charges_col = charges_candidates[0]

        return df[days_col].replace(0, np.nan).rdiv(df[charges_col]) / 30

    cost_transformer = CustomCombinationTransformer(calculate_cost_per_tenure, ['cost_per_days_since'])
    final_df_basic = pd.concat([df_basic, cost_transformer.transform(df_basic)], axis=1)

    print("Final DataFrame (Only Feature Engineering):")
    print(final_df_basic.head())

    # ================================================================
    #   Pipeline 2: Feature Engineering + Feature Selection
    # ================================================================
    from sklearn.feature_selection import SelectKBest, chi2

    fe_with_selection = FeatureSelectorTransformer(
        numeric_features=numeric_features,
        datetime_features=datetime_features,
        text_features=text_features,
        categorical_features=categorical_features,
        feature_selector=SelectKBest(score_func=chi2, k=5)  # type: ignore
    )

    final_df_selected = fe_with_selection.fit_transform(
        processed_df.drop(columns=["churn"]),
        processed_df["churn"]
    )

    print("\n" + "=" * 60)
    print("Final DataFrame (Feature Engineering + Selection):")
    print("=" * 60)
    print(final_df_selected.head())

    print("\nSelected Features:")
    print(final_df_selected.columns.tolist())
