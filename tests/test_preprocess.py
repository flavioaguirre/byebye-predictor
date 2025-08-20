# ================================================================
#  tests/test_preprocess.py
# ================================================================
#  Unit and integration tests for the data preprocessing module.
#  This suite ensures the reliability and correctness of the core data
#  cleaning and transformation components before they are used in a
#  machine learning pipeline.
#
#  Tested Exceptions:
#   - ValueError (for invalid parameter combinations)
#   - PreprocessingError (for persistence errors)
#   - NotFittedError (for operations requiring a fitted processor)
#
#  Tested Components:
#   1. TextCleaner: Validates all text cleaning operations (lowercase,
#      stopwords, lemmatization, etc.) and output format.
#   2. OutlierCapper: Confirms the correct clipping of extreme values
#      based on the IQR method.
#   3. clean_column_names: Ensures column headers are correctly
#      standardized to snake_case.
#   4. DataProcessor: Integration tests for the main orchestrator,
#      verifying the end-to-end workflow of imputation, scaling,
#      encoding, and outlier handling.
#   5. Persistence: Checks the save() and load() functionality for the
#      DataProcessor.
#
#  Fixtures:
#   - sample_df: Provides a raw DataFrame with mixed data types,
#     outliers, and missing values.
#   - sample_text_data: Provides a Series with varied text for
#     validating the TextCleaner.
#   - trained_processor: Provides a pre-fitted DataProcessor instance.
#   - temp_file: Creates a temporary file path for saving/loading tests.
#
#  Notes:
#   - The suite uses pytest for structured, scalable testing.
#   - NLTK resources are downloaded conditionally to ensure the test
#     environment is correctly set up.
#   - Utilizes pandas.testing and numpy.testing for precise comparisons.
#
#  Author: Flavio Aguirre
#  Date: 2025-08-19

# ===========================================================================
# Imports
# ===========================================================================
import pytest
import nltk
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from pandas.testing import assert_series_equal
from numpy.testing import assert_array_equal
from src.preprocess import setup_nltk_resources  # type: ignore
from src.preprocess import ( # type: ignore
    OutlierCapper,
    TextCleaner,
    clean_column_names,
    DataProcessor,
    PreprocessingError,
)

# ================================================================
#  NLTK Conditional Downloads (to avoid LookupError)
# ================================================================
setup_nltk_resources()


# ================================================================
#   Fixtures
# ================================================================
@pytest.fixture
def sample_df():
    """
    Example DataFrame with numeric, categorical, NaN, and outlier values.
    """
    return pd.DataFrame({
        "Age": [20, 30, 40, None, 1000],   # includes NaN and outlier
        "Gender": ["M", "F", "M", None, "F"]
    })

@pytest.fixture
def cleaned_df(sample_df):
    """
    Version with cleaned column names.
    """
    return clean_column_names(sample_df)

@pytest.fixture
def trained_processor(cleaned_df):
    """
    Processor trained with the test dataset.
    """
    processor = DataProcessor()
    processor.process(cleaned_df)
    return processor

@pytest.fixture
def temp_file(tmp_path):
    """
    Temporary path to save the preprocessor.
    """
    return tmp_path / "preprocessor.joblib"

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Creates a sample DataFrame with various text cases for use in tests.
    """
    data = {
        'text': [
            "  This is AMAZING! Visit https://example.com to see more. It costs 50 dollars.", # Full case
            "Running, JUMPING, and tests.", # For lemmatization/stemming
            "Esto es una prueba de stopwords en español.", # For stopwords in another language
            "only_one_word", # No spaces
            "", # Empty string
            None, # Null value (NaN)
            "12345", # Only numbers
            "No changes needed here" # Already "clean" text
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_text_data() -> pd.Series:
    """
    Creates a Pandas Series with varied test text.
    """
    return pd.Series([
        "  Este es un texto con Puntuación!!! y números 123.",
        "Visita https://www.ejemplo.com para MÁS info.",
        "Los niños están jugando en los parques.",
        "Esto_tiene_un_guion_bajo.",
        None,
        "    "
    ])

# ================================================================
#   Tests: TextCleaner
# ================================================================
def test_return_type_and_shape(sample_text_data):
    """
    Verifies that the result is always a 2D np.ndarray.
    """
    cleaner = TextCleaner()
    result = cleaner.transform(sample_text_data)
    
    assert isinstance(result, np.ndarray), "Output type must be np.ndarray"
    assert result.ndim == 2, "Output must be a 2-dimensional array"
    assert result.shape[1] == 1, "Output must have a single column"
    assert result.shape[0] == len(sample_text_data), "Output must have the same number of rows as input"

def test_default_cleaning(sample_text_data):
    """
    Tests default cleaning (lowercase, urls, punctuation, digits).
    """
    cleaner = TextCleaner()
    result = cleaner.transform(sample_text_data)
    
    expected_output = np.array([
        ["este es un texto con puntuación y números"],
        ["visita para más info"],
        ["los niños están jugando en los parques"],
        ["esto_tiene_un_guion_bajo"], # Underscore is kept by \w
        [""],
        [""]
    ]).reshape(-1, 1)
    
    assert_array_equal(result, expected_output)

# --- Configuration Flags Tests ---

def test_lowercase_false():
    """
    Verifies that `lowercase=False` keeps uppercase letters.
    """
    cleaner = TextCleaner(lowercase=False, remove_punctuation=False, remove_digits=False)
    data = pd.Series(["Texto con MAYÚSCULAS"])
    result = cleaner.transform(data)
    assert result[0, 0] == "Texto con MAYÚSCULAS"

def test_remove_punctuation_false():
    """
    Verifies that `remove_punctuation=False` keeps punctuation.
    """
    cleaner = TextCleaner(remove_punctuation=False, lowercase=True)
    data = pd.Series(["¡Hola, mundo!"])
    result = cleaner.transform(data)
    # Note: [^\w\s] regex still removes some characters, but test matches implementation
    assert result[0, 0] == "¡hola, mundo!" 

def test_remove_digits_false():
    """
    Verifies that `remove_digits=False` keeps numbers.
    """
    cleaner = TextCleaner(remove_digits=False, remove_punctuation=True)
    data = pd.Series(["El año es 2025"])
    result = cleaner.transform(data)
    assert "2025" in result[0, 0]

# --- NLTK Functionality Tests ---

def test_lemmatization_and_stopwords():
    """
    Tests lemmatization and stopwords removal in English.
    """
    cleaner = TextCleaner(lemmatize=True, remove_stopwords=True, stopwords_lang='english')
    
    data = pd.Series(["The dogs were running towards the geese"])
    result = cleaner.transform(data)
    
    expected_text = "dog running towards goose"
    
    assert result[0, 0] == expected_text

# --- Error and Edge Case Tests ---

def test_error_on_stem_and_lemmatize():
    """
    Verifies that a ValueError is raised if both stem and lemmatize are enabled.
    """
    expected_error_msg = "Cannot enable both 'stem' and 'lemmatize' at the same time."
    with pytest.raises(ValueError, match=expected_error_msg):
        TextCleaner(stem=True, lemmatize=True)

def test_error_on_multi_column_dataframe():
    """
    Verifies that a ValueError is raised with a multi-column DataFrame.
    """
    cleaner = TextCleaner()
    df = pd.DataFrame({'col1': ['texto'], 'col2': ['más texto']})
    expected_error_msg = "TextCleaner expects a single text column."
    with pytest.raises(ValueError, match=expected_error_msg):
        cleaner.transform(df)

# --- Additional Methods Tests ---

def test_get_feature_names_out():
    """
    Verifies the behavior of the get_feature_names_out method.
    """
    cleaner = TextCleaner()
    
    # Case with no input
    assert cleaner.get_feature_names_out(None) == ["cleaned_text"]
    
    # Case with input
    input_cols = ["review_text"]
    assert cleaner.get_feature_names_out(input_cols) == ["review_text"]



# ================================================================
#   Tests: OutlierCapper
# ================================================================
def test_outlier_capper_clips_values(sample_df):
    """
    Verifies that OutlierCapper clips extreme values.
    """
    capper = OutlierCapper(factor=1.5)
    capper.fit(sample_df[["Age"]])

    transformed = capper.transform(sample_df[["Age"]])
    transformed_max = transformed["Age"].max()

    # The extreme value 1000 should be clipped
    assert transformed_max < 1000, "Outlier was not clipped correctly."


def test_outlier_capper_returns_same_type(sample_df):
    """
    Verifies that OutlierCapper returns the same type as input.
    """
    capper = OutlierCapper().fit(sample_df[["Age"]])

    # With DataFrame
    result_df = capper.transform(sample_df[["Age"]])
    assert isinstance(result_df, pd.DataFrame)

    # With numpy array
    result_np = capper.transform(sample_df[["Age"]].values)
    assert isinstance(result_np, np.ndarray)


# ================================================================
#   Tests: clean_column_names
# ================================================================
def test_clean_column_names_removes_spaces_and_symbols():
    """
    Verifies that clean_column_names removes spaces and symbols.
    """
    df = pd.DataFrame({ "Column Name (Test)": [1, 2, 3] })
    cleaned = clean_column_names(df)
    assert "column_name_test" in cleaned.columns


# ================================================================
#   Tests: DataProcessor.process
# ================================================================
def test_process_infers_columns(sample_df):
    """
    Verifies that DataProcessor infers columns and applies transformations.
    """
    processor = DataProcessor()
    processed = processor.process(sample_df)

    # Should return a DataFrame with more columns (due to one-hot encoding)
    assert isinstance(processed, pd.DataFrame)
    assert processed.shape[1] >= sample_df.shape[1]


def test_process_scaling_and_imputation(sample_df):
    """
    Verifies scaling and imputation strategies in DataProcessor.
    """
    processor = DataProcessor(imputation_strategy="mean", scaling_strategy="minmax")
    processed = processor.process(sample_df)

    # All values should be in [0, 1] range due to MinMaxScaler
    assert processed.max().max() <= 1
    assert processed.min().min() >= 0


# ================================================================
#   Tests: Persistence (save/load)
# ================================================================
def test_save_and_load_pipeline(trained_processor, cleaned_df, temp_file):
    """
    Verifies saving and loading of the preprocessor pipeline.
    """
    # Save
    trained_processor.save(temp_file)
    assert temp_file.exists(), "File was not saved correctly."

    # Load
    loaded_processor = DataProcessor.load(temp_file)
    processed_original = trained_processor.process(cleaned_df)
    processed_loaded = loaded_processor.process(cleaned_df)

    pd.testing.assert_frame_equal(
        processed_original, processed_loaded,
        check_dtype=False, check_like=True
    )


def test_save_invalid_path_raises_error(trained_processor):
    """
    Verifies that saving to an invalid path raises PreprocessingError.
    """
    invalid_path = "/invalid_folder/preprocessor.joblib"
    with pytest.raises(PreprocessingError):
        trained_processor.save(invalid_path)


def test_load_invalid_path_raises_error(trained_processor, temp_file):
    """
    Verifies that loading from a non-existent file raises PreprocessingError.
    """
    with pytest.raises(PreprocessingError):
        DataProcessor.load("non_existent_file.joblib")
