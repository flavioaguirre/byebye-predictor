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
from sklearn.compose import ColumnTransformer

# --- Local Modules ---
from src.utils import get_logger, add_project_root_to_path  # type: ignore
from src.data_loader import _validate_dataframe, log_operation  # type: ignore


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


# ================================================================
#   Main FeatureEngineer Class
# ================================================================
class FeatureEngineer:
    """
    Orchestrates the complete feature engineering pipeline for a DataFrame.

    This class takes the output of a DataProcessor and applies configurable
    feature creation and selection strategies based on column types.

    Parameters
    ----------
    numeric_features : list of str, optional
        Numerical columns for transformations (e.g., polynomial features).
    datetime_features : list of str, optional
        Datetime columns for feature extraction.
    text_features : list of str, optional
        Text columns for NLP transformations.
    
    Attributes
    ----------
    pipeline_ : sklearn.compose.ColumnTransformer
        The fitted feature engineering pipeline.
    _feature_names_out : list
        Output feature names after transformation.
    """
    logger.info("Initializing FeatureEngineer Class...")
    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        datetime_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        feature_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None   #type: ignore
    ):
        logger.info("Initializing FeatureEngineer instance...")
        self.numeric_features = numeric_features if numeric_features is not None else []
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.datetime_features = datetime_features if datetime_features is not None else []
        self.text_features = text_features if text_features is not None else []
        self.feature_function = feature_function
        self.pipeline_ = None
        self._feature_names_out = None

    def _build_pipeline(self):
        """Builds the ColumnTransformer pipeline based on feature types."""
        try:
            logger.info("Building feature engineering pipeline...")

            # numeric_pipeline now ONLY for tenure (we left out totalcharges)
            numeric_pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler())
            ])

            try:
                categorical_pipeline = Pipeline([
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
            except TypeError:
                categorical_pipeline = Pipeline([
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)) # type: ignore
                ])

            transformers = [
                ('numeric_fe', numeric_pipeline, [col for col in self.numeric_features if col != 'totalcharges']), # type: ignore
                ('categorical_fe', categorical_pipeline, self.categorical_features),    # type: ignore
                ('datetime_fe', DatetimeFeatureTransformer(), self.datetime_features),
                ('text_fe', TextFeatureTransformer(), self.text_features),
            ]

            # Keep totalcharges in remainder for use in calculate_cost_per_tenure
            self.pipeline_ = ColumnTransformer( # type: ignore
                transformers,
                remainder='passthrough',
                sparse_threshold=0  
            )

            logger.info("Feature engineering pipeline built successfully.")

        except Exception as e:
            raise FeatureEngineeringError(f"Error building pipeline: {e}") from e

    @log_operation
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Builds and fits the feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to fit the pipeline on.
        y : pd.Series, optional
            Target variable (not used for fitting).

        Returns
        -------
        self : FeatureEngineer
            The fitted instance.
        """
        self._build_pipeline()
        logger.info("Fitting feature engineering pipeline...")
        self.pipeline_.fit(df)  # type: ignore
        self._feature_names_out = self.pipeline_.get_feature_names_out()    # type: ignore
        return self

    @log_operation
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted feature engineering pipeline to the data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with new features.
        """
        if self.pipeline_ is None:
            raise FeatureEngineeringError("The FeatureEngineer has not been fitted yet. Call .fit() first.")
        logger.info("Transforming data with the feature engineering pipeline.")
        transformed_data = self.pipeline_.transform(df) #type:ignore
        output_df = pd.DataFrame(
            transformed_data, 
            columns=self._feature_names_out, 
            index=df.index
        )
        logger.info(f"Feature engineering complete. Final shape: {output_df.shape}")
        return output_df

    @log_operation
    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the pipeline to the data and then transforms it.
        """
        return self.fit(df, y).transform(df)

    @log_operation
    def save(self, filepath: str) -> None:
        """
        Saves the fitted FeatureEngineer instance to a .joblib file.
        """
        try:
            joblib.dump(self, filepath)
            logger.info(f"FeatureEngineer instance saved successfully at {filepath}.")
        except Exception as e:
            raise FeatureEngineeringError(f"Error saving FeatureEngineer instance: {e}") from e

    @classmethod
    @log_operation
    def load(cls, filepath: str) -> "FeatureEngineer":
        """
        Loads a FeatureEngineer instance from a .joblib file.
        """
        try:
            loaded_instance = joblib.load(filepath)
            if not isinstance(loaded_instance, cls):
                raise TypeError("Loaded object is not a FeatureEngineer instance.")
            logger.info(f"FeatureEngineer instance loaded successfully from {filepath}.")
            return loaded_instance
        except Exception as e:
            raise FeatureEngineeringError(f"Error loading FeatureEngineer instance: {e}") from e


# ================================================================
#   Example Usage
# ================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("### STEP 1: LOAD AND PREPROCESS DATA ###")
    print("=" * 60)

    data = pd.DataFrame(
        {
            "CustomerID": [f"CUST-{i}" for i in range(10)],
            "Tenure": [1, 12, 24, 3, 5, 60, 34, 22, 5, 8],
            "TotalCharges": [29.8, 1889.5, 108.1, 1840.7, 151.6, 820.5, 346.4, 1532.0, 30000.0, 588.4,],
            "IsSenior": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Month-to-month", "Two year", "One year", "Month-to-month", "One year", "Month-to-month",],
            "SatisfactionScore": ["High", "Low", "Medium", "High", "Low", "None", "Medium", "High", "High", "Low",],
            "ReviewComment": ["great service, very happy!", "terrible experience, cancelling my contract http://bad.com", "average price and good support, i think i will stay", "their APP is SO BAD 1234, really frustrating experience", "no complaints so far", "excellent!! will renew for sure, top quality", "meh, it is okay i guess, not the best not the worst", "I will probably churn next month due to poor connection", "very expensive for what it is offering to the customer base", "good value",],
            "JoinDate": pd.to_datetime(["2020-01-15", "2019-02-20", "2018-03-01", "2022-04-10", "2023-05-05", "2015-06-12", "2017-07-25", "2018-08-30", "2023-09-01", "2022-10-14"])
        }
    )
    
    processed_df = data.copy()
    processed_df.columns = [col.lower() for col in processed_df.columns]
    
    numeric_features = ['tenure', 'totalcharges', 'issenior']
    datetime_features = ['joindate']
    text_features = ['reviewcomment']

    feature_engineer = FeatureEngineer(
        numeric_features=numeric_features,
        datetime_features=datetime_features,
        text_features=text_features
    )
    
    final_df = feature_engineer.fit_transform(processed_df)

    # We adapt the custom feature to the names that actually come out of the pipeline
    def calculate_cost_per_tenure(df: pd.DataFrame) -> pd.Series:
        """Calculates the average monthly cost for each customer."""
        return df['datetime_fe__joindate_days_since'].replace(0, np.nan).rdiv(df['remainder__totalcharges']) / 30

    cost_transformer = CustomCombinationTransformer(calculate_cost_per_tenure, ['cost_per_days_since'])
    final_df = pd.concat([final_df, cost_transformer.transform(final_df)], axis=1)

    print("Final DataFrame with new engineered features:")
    print(final_df.head())