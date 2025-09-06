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
#       - class DatetimeFeatures(BaseEstimator, TransformerMixin)

# Helpers:
#      - clean_column_names()
#      - _clean_feature_names()

# - Main Preprocessor Class
#       - class DataProcessor()

# - Example Usage

# ------------
#  Author: Flavio Aguirre
#  Date: 2025-08-13

# ================================================================
# Importing Libraries
# ================================================================
# --- Standard Library ---
import os
import re
from typing import List, Optional, Union, Dict, Any, Literal

# --- Third-Party Libraries ---
import joblib
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# --- Local Modules ---
from src.utils import get_logger, add_project_root_to_path # type: ignore
from src.data_loader import _validate_dataframe, log_operation, InvalidDataFrameError  # type: ignore

add_project_root_to_path()

# ================================================================
#   Logger
# ================================================================
logger = get_logger(__name__)


# ================================================================
#  NLTK Conditional Downloads
# ================================================================
def setup_nltk_resources():
    """
    Checks and downloads required NLTK resources only if not already present.
    """
    # Dictionary with resource names and their search paths
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "punkt_tab": "tokenizers/punkt_tab"
    }

    for name, path in resources.items():
        try:
            # Try to find the resource
            nltk.data.find(path)
            logger.info(f"NLTK resource '{name}' is already downloaded.")
        except LookupError:
            # If not found, download it
            logger.warning(f"NLTK resource '{name}' not found. Downloading...")
            nltk.download(name)

# --- Usage ---
setup_nltk_resources()


# ================================================================
#   Custom Exceptions
# ================================================================
class PreprocessingError(Exception):
    """
    Base exception for errors occurring during preprocessing.
    """
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(f"PreprocessingError: {message}")


# ================================================================
#   Custom Transformers & Helper Functions
# ================================================================
class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to extract date-related features from datetime columns.

    Attributes
    ----------
    columns_ : list
        List of columns to transform, set during fit.

    Methods
    -------
    fit(X, y=None)
        Learns which columns to transform.
    transform(X)
        Extracts year, month, day, and dayofweek features from datetime columns.
    get_feature_names_out(input_features=None)
        Returns the names of the generated features.

    Raises
    ------
    ValueError
        If input columns cannot be converted to datetime.
    """
    def fit(self, X, y=None):
        """
        Learns which columns to transform.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime columns.
        y : Ignored

        Returns
        -------
        self : DatetimeFeatures
            Fitted transformer.
        """
        self.columns_ = X.columns
        return self

    def transform(self, X):
        """
        Extracts year, month, day, and dayofweek features from datetime columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with new columns for each datetime feature.

        Raises
        ------
        ValueError
            If any column cannot be converted to datetime.
        """
        df = X.copy()
        for col in self.columns_:
            try:
                dt = pd.to_datetime(df[col])
            except Exception as e:
                raise ValueError(f"Column '{col}' cannot be converted to datetime: {e}")
            df[f'{col}_year'] = dt.dt.year #type: ignore
            df[f'{col}_month'] = dt.dt.month #type: ignore
            df[f'{col}_day'] = dt.dt.day #type: ignore
            df[f'{col}_dayofweek'] = dt.dt.dayofweek #type: ignore

        new_cols = [f'{col}_{feat}' for col in self.columns_ for feat in ['year', 'month', 'day', 'dayofweek']]
        return df[new_cols]

    def get_feature_names_out(self, input_features=None):
        """
        Generates the names of the output columns.

        Parameters
        ----------
        input_features : list or None
            Input feature names.

        Returns
        -------
        list
            List of output feature names.
        """
        feature_names = []
        for col in self.columns_:
            feature_names.extend([
                f'{col}_year',
                f'{col}_month',
                f'{col}_day',
                f'{col}_dayofweek'
            ])
        return feature_names


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Transformer to identify and cap outliers in numerical columns.

    Parameters
    ----------
    factor : float, default=1.5
        The multiplier for the interquartile range (IQR) to determine outlier boundaries.

    Attributes
    ----------
    boundaries_ : dict
        Stores the lower and upper bounds for each column.
    columns_ : list
        List of columns fitted.

    Methods
    -------
    fit(X, y=None)
        Calculates outlier boundaries for each numeric column.
    transform(X)
        Clips outliers in the data.
    get_feature_names_out(input_features=None)
        Returns the output feature names.

    Raises
    ------
    TypeError
        If input columns are not numeric.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.boundaries_ = {}
        self.columns_ = []

    def fit(self, X, y=None) -> "OutlierCapper":
        """
        Fit the OutlierCapper to the data, calculating outlier boundaries for each numeric column.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data to fit.
        y : Ignored

        Returns
        -------
        self : OutlierCapper
            Fitted transformer.
        """
        logger.debug("Fitting OutlierCapper.")
        
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.columns_ = X_df.columns # type: ignore

        for col in self.columns_:
            # Ensure the column is numeric
            if pd.api.types.is_numeric_dtype(X_df[col]):
                q1 = X_df[col].quantile(0.25)
                q3 = X_df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (self.factor * iqr)
                upper_bound = q3 + (self.factor * iqr)
                self.boundaries_[col] = (lower_bound, upper_bound)
        
        logger.info(f"Outlier boundaries calculated for {len(self.boundaries_)} numeric columns.")
        return self

    def transform(self, X) -> pd.DataFrame | Any:
        """
        Apply outlier capping to the data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data to transform.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Transformed data with outliers capped.
        """
        logger.debug("Transforming data with OutlierCapper.")
        is_dataframe = isinstance(X, pd.DataFrame)
        X_df = pd.DataFrame(X) if not is_dataframe else X.copy()
        
        for col, (lower, upper) in self.boundaries_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=lower, upper=upper) # type: ignore

        # We return the same type as received
        return X_df if is_dataframe else X_df.values

    def get_feature_names_out(self, input_features=None) -> Any | None:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : list or None
            Input feature names.

        Returns
        -------
        list or None
            Output feature names.
        """
        return input_features


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Robust transformer for cleaning text, fully compatible with scikit-learn.
    Handles pd.DataFrame or pd.Series as input.

    Parameters
    ----------
    lowercase : bool, default=True
        Whether to convert text to lowercase.
    remove_urls : bool, default=True
        Whether to remove URLs from text.
    remove_punctuation : bool, default=True
        Whether to remove punctuation.
    remove_digits : bool, default=True
        Whether to remove digits.
    remove_stopwords : bool, default=False
        Whether to remove stopwords.
    stopwords_lang : str, default='english'
        Language for stopwords removal.
    lemmatize : bool, default=False
        Whether to apply lemmatization.
    stem : bool, default=False
        Whether to apply stemming.

    Attributes
    ----------
    stop_words : set
        Set of stopwords for removal (if enabled).
    lemmatizer : WordNetLemmatizer
        Lemmatizer instance (if enabled).
    stemmer : PorterStemmer
        Stemmer instance (if enabled).

    Methods
    -------
    fit(X, y=None)
        Does nothing, present for compatibility.
    transform(X)
        Cleans and transforms text data.
    get_feature_names_out(input_features=None)
        Returns the output feature names.

    Raises
    ------
    ValueError
        If both `stem` and `lemmatize` are set to True.
        If input DataFrame has more than one column.
    """
    def __init__(self,
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 remove_punctuation: bool = True,
                 remove_digits: bool = True,
                 remove_stopwords: bool = False,
                 stopwords_lang: str = 'english',
                 lemmatize: bool = False,
                 stem: bool = False):
        
        if stem and lemmatize:
            raise ValueError("Cannot enable both 'stem' and 'lemmatize' at the same time.")
            
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.remove_stopwords = remove_stopwords
        self.stopwords_lang = stopwords_lang
        self.lemmatize = lemmatize
        self.stem = stem

        # Initialize only if needed
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(self.stopwords_lang))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        if self.stem:
            self.stemmer = PorterStemmer()

    def fit(self, X, y=None) ->"TextCleaner":
        """
        Fit method for compatibility. Does nothing.

        Parameters
        ----------
        X : Ignored
        y : Ignored

        Returns
        -------
        self : TextCleaner
            Fitted transformer.
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """
        Clean and transform text data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or np.ndarray
            Input text data.

        Returns
        -------
        np.ndarray
            Cleaned text as a 2D numpy array.
        """
        # Accept ndarray, Series, or DataFrame
        if isinstance(X, np.ndarray):
            X_series = pd.Series(X.ravel())
        elif isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("TextCleaner expects a single text column.")
            X_series = X.iloc[:, 0]
        else:
            X_series = X.copy()

        X_cleaned = X_series.fillna("").astype(str)

        if self.lowercase:
            X_cleaned = X_cleaned.str.lower()
        if self.remove_urls:
            X_cleaned = X_cleaned.str.replace(r"https?://\S+|www\.\S+", "", regex=True)
        if self.remove_punctuation:
            X_cleaned = X_cleaned.str.replace(r'[^\w\s]', '', regex=True)
        if self.remove_digits:
            X_cleaned = X_cleaned.str.replace(r'\d+', '', regex=True)

        X_cleaned = X_cleaned.str.replace(r'\s+', ' ', regex=True).str.strip()

        if any([self.remove_stopwords, self.lemmatize, self.stem]):
            X_cleaned = X_cleaned.apply(self._process_tokens)

        # Return as 2D array for sklearn
        return X_cleaned.to_numpy().reshape(-1, 1)

    def _process_tokens(self, text: str) -> str:
        """
        Tokenize and optionally remove stopwords, lemmatize, or stem tokens.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Processed text.
        """
        tokens = word_tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stop_words]
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        if self.stem:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : list, pd.Index, or None
            Input feature names.

        Returns
        -------
        list
            Output feature names.
        """
        if input_features is None:
            return ["cleaned_text"]
        if isinstance(input_features, (list, pd.Index)):
            return list(input_features)
        return [input_features]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans DataFrame column names, converting them to snake_case.

    Example: "Column Name", "columnName" -> "column_name"

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose column names will be cleaned.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with column names in snake_case.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    """
    df_copy = df.copy()
    new_columns = []
    for col in df_copy.columns:
        # 1. Insert an underscore before any capital letter that follows a lowercase letter
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str(col))
        # 2. Insert an underscore before any capital letter that follows a digit or other capital letter
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
        # 3. Replace any non-alphanumeric character (except '_') with a single underscore
        cleaned_col = re.sub(r'[^a-zA-Z0-9_]+', '_', s2)
        
        # 4. Remove leading/trailing underscores and replace multiple underscores with one
        cleaned_col = re.sub(r'^_+|_+$', '', cleaned_col)
        cleaned_col = re.sub(r'_+', '_', cleaned_col)
        
        new_columns.append(cleaned_col)
        
    df_copy.columns = new_columns
    return df_copy


def _clean_feature_names(feature_names: List[str]) -> List[str]:
    """
    Cleans feature names generated by ColumnTransformer and other transformers
    by removing prefixes and standardizing names.

    Parameters
    ----------
    feature_names : list of str
        List of feature names to clean.

    Returns
    -------
    list of str
        Cleaned feature names.
    """
    clean_names = []
    for name in feature_names:
        # 1. Remove ColumnTransformer prefixes (e.g., 'num__', 'cat__')
        if '__' in name:
            name = name.split('__', 1)[1]

        # 2. Convert to snake_case and replace special characters
        # This handles OneHotEncoder output, which can have spaces
        cleaned_name = re.sub(r'[^a-zA-Z0-9]', '_', name) # Replace non-alphanumeric with '_'
        cleaned_name = re.sub(r'_+', '_', cleaned_name) # Replace multiple underscores with one
        cleaned_name = cleaned_name.strip('_').lower() # Remove leading/trailing underscores and lowercase

        clean_names.append(cleaned_name)
    return clean_names


# ================================================================
#   Main DataProcessor Class
# ================================================================
class DataProcessor:
    """
    Orchestrates the complete preprocessing pipeline for a DataFrame.

    Parameters
    ----------
    numerical_cols : list of str, optional
        List of numerical columns to process.
    categorical_cols : list of str, optional
        List of categorical columns to process.
    datetime_cols : list of str, optional
        List of datetime columns to process.
    imputation_strategy : {'mean', 'median', 'most_frequent'}, default='median'
        Strategy for imputing missing values in numerical columns.
    scaling_strategy : {'standard', 'minmax'}, default='standard'
        Scaling strategy for numerical columns.
    text_cardinality_threshold : float, default=0.7
        Threshold for distinguishing high-cardinality text columns.

    Attributes
    ----------
    pipeline_ : sklearn.compose.ColumnTransformer
        The fitted preprocessing pipeline.
    _feature_names_out : list
        Output feature names after transformation.
    low_cardinality_cols_ : list
        List of low-cardinality columns for encoding.
    high_cardinality_cols_ : list
        List of high-cardinality columns to drop.
    numeric_cols_ : list
        List of detected numeric columns.
    datetime_cols_ : list
        List of detected datetime columns.
    comments_cols_ : list
        List of detected comment-like columns.
    high_card_noise_cols_ : list
        List of detected high-cardinality noise columns.

    Methods
    -------
    process(df, y=None)
        Executes the complete data cleaning and preprocessing workflow.
    save(filepath)
        Saves the preprocessor (pipeline and configuration) to a .joblib file.
    load(filepath)
        Loads a previously saved preprocessor from a .joblib file.

    Raises
    ------
    PreprocessingError
        If columns specified do not exist in the DataFrame.
        If saving or loading fails.
    """
    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        datetime_cols: Optional[List[str]] = None,
        imputation_strategy: Literal['mean', 'median', 'most_frequent'] = 'median',
        scaling_strategy: Literal['standard', 'minmax'] = 'standard',
        text_cardinality_threshold: float = 0.7
    ):
        logger.info("Initializing DataProcessor Class...")
        self.numerical_cols = numerical_cols if numerical_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.datetime_cols = datetime_cols if datetime_cols is not None else []
        self.imputation_strategy = imputation_strategy
        self.scaling_strategy = scaling_strategy
        self.text_cardinality_threshold = text_cardinality_threshold

        # Attributes learned during processing
        self.pipeline_ = None
        self._feature_names_out = None
        self.low_cardinality_cols_ = []    
        self.high_cardinality_cols_ = [] 
        self.numeric_cols_ = []
        self.datetime_cols_ = []
        logger.info("DataProcessor initialized successfully!")

    def _validate_manual_columns(self, df: pd.DataFrame) -> None:
        """
        Validates that manual columns exist in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Raises
        ------
        PreprocessingError
            If any specified column does not exist in the DataFrame.
        """
        all_cols = set(df.columns)
        for user_list, name in [
            (self.numerical_cols, "numerical_cols"),
            (self.categorical_cols, "categorical_cols"),
            (self.datetime_cols, "datetime_cols"),
        ]:
            if user_list:
                missing = set(user_list) - all_cols
                if missing:
                    raise PreprocessingError(f"Columns {missing} passed in {name} do not exist in DataFrame.")

    def _detect_comment_columns(self, df: pd.DataFrame, high_cardinality_cols: list) -> dict:
        """
        Distinguishes between high-cardinality columns that look like comments
        and columns that look like IDs/noise.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze.
        high_cardinality_cols : list
            List of high-cardinality columns.

        Returns
        -------
        dict
            Dictionary with keys 'comments' and 'noise', each containing a list of column names.
        """

        comment_cols = []
        noise_cols = []

        for col in high_cardinality_cols:
            series = df[col].dropna().astype(str)

            # Métricas básicas
            avg_len = series.map(len).mean()
            max_len = series.map(len).max()
            space_ratio = (series.str.contains(" ")).mean()  # % con espacios

            # Heurística para comentarios
            if avg_len > 10 and max_len > 30 and space_ratio > 0.3:
                comment_cols.append(col)
            else:
                noise_cols.append(col)

        logger.info(f"Detected comment-like columns: {comment_cols}")
        logger.info(f"Detected high-cardinality noise columns: {noise_cols}")

        return {"comments": comment_cols, "noise": noise_cols}

    def _classify_columns(self, df: pd.DataFrame) -> dict:
        """
        Detects numeric, categorical (low/high cardinality), datetime, and comment columns.
        Respects manual columns and autocompletes the rest.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to classify.

        Returns
        -------
        dict
            Dictionary with keys: 'numeric', 'datetime', 'low_cardinality', 'high_cardinality', 'comments', 'high_card_noise'.
        """

        logger.debug("Classifying dataframe columns with validations and hybrid detection.")

        # Validate manual columns
        self._validate_manual_columns(df)

        # Initialize
        self.numeric_cols_ = []
        self.datetime_cols_ = []
        self.low_cardinality_cols_ = []
        self.high_cardinality_cols_ = []

        manual_numeric = set(self.numerical_cols)
        manual_categorical = set(self.categorical_cols)
        manual_datetime = set(self.datetime_cols)

        for col in df.columns:
            if col in manual_numeric or col in manual_categorical or col in manual_datetime:
                continue

            series = df[col]

            # Numerics
            if ptypes.is_numeric_dtype(series):
                self.numeric_cols_.append(col)
                continue

            # Datetime
            if ptypes.is_datetime64_any_dtype(series):
                self.datetime_cols_.append(col)
                continue

            # Strings/Objects → possible datetime or categorical
            if ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series):
                temp_series = pd.to_datetime(series, format='%d/%m/%Y', errors="coerce")
                if temp_series.notna().sum() / max(series.notna().sum(), 1) > 0.9:
                    self.datetime_cols_.append(col)
                    continue

                if series.notna().sum() == 0:
                    continue

                cardinality_ratio = series.nunique() / series.notna().sum()
                if cardinality_ratio >= self.text_cardinality_threshold:
                    self.high_cardinality_cols_.append(col)
                else:
                    self.low_cardinality_cols_.append(col)

        # We combine manual + auto
        all_numeric = list(manual_numeric.union(self.numeric_cols_))
        all_datetime = list(manual_datetime.union(self.datetime_cols_))
        all_low_cardinality = list(manual_categorical) if manual_categorical else self.low_cardinality_cols_
        all_high_cardinality = self.high_cardinality_cols_

        # Separating comments from noise
        high_card_split = self._detect_comment_columns(df, all_high_cardinality)

        return {
            "numeric": all_numeric,
            "datetime": all_datetime,
            "low_cardinality": all_low_cardinality,
            "high_cardinality": all_high_cardinality,
            "comments": high_card_split["comments"],
            "high_card_noise": high_card_split["noise"]
        }
    
    # logger.info("Building preprocessing pipeline...")
    def _build_pipeline(self):
        """
        Builds the preprocessing pipeline based on the classification of columns.

        Attributes Used
        ---------------
        numeric_cols_ : list
            Numeric columns to process.
        low_cardinality_cols_ : list
            Low-cardinality categorical columns to encode.
        datetime_cols_ : list
            Datetime columns to extract features from.
        comments_cols_ : list
            Comment-like columns to clean as text.
        high_card_noise_cols_ : list
            High-cardinality noise columns to drop.

        Raises
        ------
        ValueError
            If required attributes are not set before building the pipeline.
        """

        logger.debug("Building preprocessing pipeline...")

        # Here we access the lists of already sorted and combined columns
        # that were filled in _classify_columns
        num_cols = self.numeric_cols_
        cat_cols = self.low_cardinality_cols_
        datetime_cols = self.datetime_cols_
        comment_cols = self.comments_cols_
        noise_cols = self.high_card_noise_cols_
        
        # Numerical steps
        numeric_steps = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.imputation_strategy)),
            ('capper', OutlierCapper()),
            ('scaler', StandardScaler() if self.scaling_strategy == 'standard' else MinMaxScaler())
        ])

        # Categorical steps
        categorical_steps = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])

        # Steps for comments
        text_steps = Pipeline(steps=[
            ('cleaner', TextCleaner())
        ])

        # Steps for Datetime
        datetime_steps = Pipeline(steps=[
            ('date_extractor', DatetimeFeatures())
        ])

        # We assemble the pipeline
        self.pipeline_ = ColumnTransformer( # type: ignore
            transformers=[
                ('num', numeric_steps, self.numeric_cols_),
                ('cat', categorical_steps, self.low_cardinality_cols_),
                ('date', datetime_steps, self.datetime_cols_),
                ('text', text_steps, self.comments_cols_),
                ('drop', 'drop', self.high_card_noise_cols_)
            ],
            remainder='passthrough'
        )
        logger.info("Preprocessing pipeline built successfully.")

    # logger.info("Activating Column Name Cleaner for later column inference...")
    def get_processed_feature_names(self, raw_feature_names: list) -> list:
        """
        Public method to expose feature name cleanup functionality.

        Parameters
        ----------
        raw_feature_names : list
            Raw feature names from the pipeline.

        Returns
        -------
        list
            Cleaned feature names.
        """
        return _clean_feature_names(raw_feature_names)

    @log_operation
    def process(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> Any | np.ndarray:
        """
        Executes the complete data cleaning and preprocessing workflow.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to process.
        y : pd.Series, optional
            Target variable (not used).

        Returns
        -------
        pd.DataFrame
            Processed DataFrame.

        Raises
        ------
        PreprocessingError
            If manual columns do not exist in the DataFrame.
        """
        logger.info("Starting data processing workflow.")

        df_clean = clean_column_names(df)

        if self.pipeline_ is None:
            # Sort and validate the columns. This populates the instance attributes.
            column_groups = self._classify_columns(df_clean)
            self.numeric_cols_ = column_groups["numeric"]
            self.low_cardinality_cols_ = column_groups["low_cardinality"]
            self.datetime_cols_ = column_groups["datetime"]
            self.comments_cols_ = column_groups["comments"]
            self.high_card_noise_cols_ = column_groups["high_card_noise"]

            # Build the pipeline based on the already sorted column lists.
            self._build_pipeline()
            
            logger.info("Fitting and transforming data with the pipeline.")
            processed_data = self.pipeline_.fit_transform(df_clean) # type: ignore

            self._feature_names_out = self.pipeline_.get_feature_names_out() # type: ignore
            self._feature_names_out = _clean_feature_names(self._feature_names_out.tolist()) # type: ignore

            processed_df = pd.DataFrame(processed_data, columns=self._feature_names_out, index=df.index)

            logger.info(f"Data processing complete. Final shape: {processed_df.shape}")
            return processed_df
        else:
            logger.info("Pipeline is already fitted. Transforming new data.")
            processed_data = self.pipeline_.transform(df_clean)
            processed_df = pd.DataFrame(processed_data, columns=self._feature_names_out, index=df.index)
            logger.info(f"Data processing complete. Final shape: {processed_df.shape}")
            return processed_df

    @log_operation
    def save(self, filepath: str) -> None:
        """
        Saves the preprocessor (pipeline and configuration) to a .joblib file.

        Parameters
        ----------
        filepath : str
            Path to the file where the preprocessor will be saved.

        Raises
        ------
        PreprocessingError
            If an error occurs during saving.
        """
        try:
            joblib.dump({
                "pipeline": self.pipeline_,
                "numerical_cols": self.numerical_cols,
                "categorical_cols": self.categorical_cols,
                "imputation_strategy": self.imputation_strategy,
                "scaling_strategy": self.scaling_strategy,
                "feature_names_out": self._feature_names_out
            }, filepath)
            logger.info(f"Preprocessor saved successfully at {filepath}.")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
            raise PreprocessingError(f"Error saving preprocessor: {e}") from e

    @classmethod
    @log_operation
    def load(cls, filepath: str) -> "DataProcessor":
        """
        Loads a previously saved preprocessor from a .joblib file.

        Parameters
        ----------
        filepath : str
            Path to the file where the preprocessor is saved.

        Returns
        -------
        DataProcessor
            An instance of the preprocessor ready to use.

        Raises
        ------
        PreprocessingError
            If an error occurs during loading.
        """
        try:
            obj = joblib.load(filepath)
            processor = cls(
                numerical_cols=obj["numerical_cols"],
                categorical_cols=obj["categorical_cols"],
                imputation_strategy=obj["imputation_strategy"],
                scaling_strategy=obj["scaling_strategy"]
            )
            processor.pipeline_ = obj["pipeline"]
            processor._feature_names_out = obj.get("feature_names_out")
            logger.info(f"Preprocessor loaded successfully from {filepath}.")
            return processor
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            raise PreprocessingError(f"Error loading preprocessor: {e}") from e

    logger.info("Preprocess module loaded correctly and ready to be used.")

# ================================================================
#   Example Usage
# ================================================================
if __name__ == "__main__":
    # --- Create a comprehensive sample DataFrame with a datetime column ---
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
            "IsSenior": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
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
            "JoinDate": pd.to_datetime([
                "2020-01-15", "2019-02-20", "2018-03-01", "2022-04-10", "2023-05-05", 
                "2015-06-12", "2017-07-25", "2018-08-30", "2023-09-01", "2022-10-14"
            ])
        }
    )

    print("\n" + "=" * 60)
    print("### SCENARIO 1: Manual Column Specification ###")
    print("=" * 60)
    # Here, the `DataProcessor` will only process the specified columns.
    # The 'join_date' column will remain intact in the output.
    preprocessor_manual = DataProcessor(
        numerical_cols=["tenure", "total_charges", "is_senior"],
        categorical_cols=["contract", "satisfaction_score"],
    )
    processed_manual = preprocessor_manual.process(data)
    print("Processed DataFrame with manual column definition:")
    print(processed_manual.head())
    print(f"Columns in the output: {processed_manual.columns.tolist()}")
    print(f"Shape of the DataFrame: {processed_manual.shape}\n")
    print()

    print("\n" + "=" * 60)
    print("### SCENARIO 2: Automatic Detection (Smart Mode) ###")
    print("=" * 60)
    # The `DataProcessor` infers all column types automatically.
    # This will trigger the new logic to handle the date column.
    preprocessor_auto = DataProcessor()
    processed_auto = preprocessor_auto.process(data)
    print("Processed DataFrame with automatic column detection:")
    print(processed_auto.head())
    print(f"Columns in the output: {processed_auto.columns.tolist()}")
    print(f"Shape of the DataFrame: {processed_auto.shape}\n")
    print(f'{processed_auto.isnull().sum()}')

    
    print("\n" + "=" * 60)
    print("### SCENARIO 3: Persistence (Save and Load) ###")
    print("=" * 60)
    try:
        os.makedirs("artifacts", exist_ok=True)
    except ImportError:
        os.makedirs("artifacts", exist_ok=True)
    FILE_PATH = "artifacts/preprocessor.joblib"

    print(f"Saving the automatically detected preprocessor to: {FILE_PATH}")
    preprocessor_auto.save(FILE_PATH)

    print("Loading the preprocessor from the file...")
    loaded_preprocessor = DataProcessor.load(FILE_PATH)
    print("Processing the original DataFrame with the loaded preprocessor:")
    # The loaded preprocessor will apply the same transformations it learned,
    # including the step through the date column.
    processed_loaded = loaded_preprocessor.process(data)
    print(processed_loaded.head())
    print(f"Columns in the output: {processed_loaded.columns.tolist()}")
    print(f"DataFrame Shape: {processed_loaded.shape}")