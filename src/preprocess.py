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
#       - class DataProcessor

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
# Simply call this function once at the start of your script or application.
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
#   Custom Transformers & Helper Function
# ================================================================
class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Transformer to identify and cap outliers in numerical columns.
    Compatible with both NumPy arrays and DataFrames.

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

        # Return the same type as received
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

    Raises
    ------
    ValueError
        If both `stem` and `lemmatize` are set to True.
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
    """
    logger.info("Initializing DataProcessor Class...")
    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        imputation_strategy: Literal['mean', 'median', 'most_frequent'] = 'median',
        scaling_strategy: Literal['standard', 'minmax'] = 'standard',
        text_cardinality_threshold: float = 0.7
    ):
        self.numerical_cols = numerical_cols if numerical_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.imputation_strategy = imputation_strategy
        self.scaling_strategy = scaling_strategy
        self.text_cardinality_threshold = text_cardinality_threshold
        # Attributes learned during processing
        self.pipeline_ = None
        self._feature_names_out = None
        self.low_cardinality_cols_ = []    
        self.high_cardinality_cols_ = [] 
        
    logger.info("Initializing column classification...")
    def _classify_text_columns(self, df: pd.DataFrame):
        """
        Separates text columns into low and high cardinality.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze for cardinality.
        """
        # NEW INTERNAL FUNCTION
        logger.debug("Classifying text columns by cardinality.")
        for col in self.categorical_cols:
            # Ignore columns with all null values to avoid division by zero
            if df[col].notna().sum() == 0:
                continue

            cardinality_ratio = df[col].nunique() / df[col].notna().sum()
            if cardinality_ratio >= self.text_cardinality_threshold:
                self.high_cardinality_cols_.append(col)
            else:
                self.low_cardinality_cols_.append(col)
        
        if self.high_cardinality_cols_:
            logger.info(f"High-cardinality columns (comments) detected and will be dropped: {self.high_cardinality_cols_}")
        if self.low_cardinality_cols_:
            logger.info(f"Low-cardinality columns (categorical) detected for encoding: {self.low_cardinality_cols_}")


    logger.info("Building preprocessing pipeline...")
    def _build_pipeline(self):
        """
        Builds the preprocessing pipeline based on the configuration.

        Returns
        -------
        None
        """
        numeric_steps = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.imputation_strategy)),
            ('capper', OutlierCapper()),
            ('scaler', StandardScaler() if self.scaling_strategy == 'standard' else MinMaxScaler())
        ])

        categorical_steps = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])

        self.pipeline_ = ColumnTransformer(     #type:ignore
            transformers=[
                ('num', numeric_steps, self.numerical_cols),
                # Apply OneHotEncoder only to low-cardinality columns
                ('cat', categorical_steps, self.low_cardinality_cols_),
                # High-cardinality columns (comments) are dropped
                ('drop_text', 'drop', self.high_cardinality_cols_)
            ],
            remainder='passthrough'
        )
        logger.info("Preprocessing pipeline built successfully.")

    logger.info("Starting data processing...")
    def process(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> Any | np.ndarray:
        """
        Executes the complete data cleaning and preprocessing workflow.

        1. Cleans column names.
        2. Infers data types.
        3. Builds and executes the preprocessing pipeline.

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
        """
        logger.info("Starting data processing workflow.")

        # Step 1: Clean column names
        df_clean = clean_column_names(df)

        if not self.numerical_cols and not self.categorical_cols:
            logger.debug("Inferring column types.")
            self.numerical_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
            self.categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
            logger.info(f"Inferred {len(self.numerical_cols)} numerical and {len(self.categorical_cols)} text/categorical columns.")
        
            # Classify text columns before building the pipeline
            self._classify_text_columns(df_clean)

            # Step 3: Build and execute the pipeline
            self._build_pipeline()
            logger.info("Fitting and transforming data with the pipeline.")
            processed_data = self.pipeline_.fit_transform(df_clean) #type:ignore

            # Rebuild the DataFrame with correct column names
            self._feature_names_out = self.pipeline_.get_feature_names_out() #type:ignore
            processed_df = pd.DataFrame(processed_data, columns=self._feature_names_out, index=df.index)

            logger.info(f"Data processing complete. Final shape: {processed_df.shape}")
            return processed_df
        else:
            logger.warning("No columns to process. Returning original DataFrame.")
            return clean_column_names(df)

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
    preprocessor_manual = DataProcessor(
        numerical_cols=["Tenure", "TotalCharges"],
        categorical_cols=["Contract", "SatisfactionScore", "IsSenior"],
    )
    processed_manual = preprocessor_manual.process(data)
    print("DataFrame processed with manual column definition:")
    print(processed_manual)
    print(f"Shape: {processed_manual.shape}\n")

    print("\n" + "=" * 60)
    print("### SCENARIO 2: Automatic Column Detection (Intelligent Mode) ###")
    print("=" * 60)
    # Instantiate without providing column lists to trigger auto-detection
    preprocessor_auto = DataProcessor()
    processed_auto = preprocessor_auto.process(data)
    print("DataFrame processed with automatic column detection:")
    print(processed_auto.head()) # type: ignore
    print(f"Shape: {processed_auto.shape}\n")

    print("\n" + "=" * 60)
    print("### SCENARIO 3: Persistence (Save and Load) ###")
    print("=" * 60)
    # Ensure artifacts directory exists
    # Ensure artifacts directory exists using the utility function if available
    try:
        from src.utils import ensure_dir_exists  # type: ignore
        ensure_dir_exists("artifacts")
    except ImportError:
        os.makedirs("artifacts", exist_ok=True)
    FILE_PATH = "artifacts/preprocessor.joblib"

    print(f"Saving the auto-detected preprocessor to: {FILE_PATH}")
    preprocessor_auto.save(FILE_PATH)

    print("Loading the preprocessor from file...")
    loaded_preprocessor = DataProcessor.load(FILE_PATH)
    print("Processing the original DataFrame with the loaded preprocessor:")
    processed_loaded = loaded_preprocessor.process(data)
    print(processed_loaded.head()) # type: ignore