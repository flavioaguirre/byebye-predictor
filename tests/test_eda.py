# ================================================================
#  tests/test_eda.py
# ================================================================
#  Unit tests for EDA (Exploratory Data Analysis) module.
#  Provides robust and comprehensive test coverage for:
#
#  Tested Exceptions:
#   - EDAError
#   - InvalidDataFrameError
#   - ColumnNotFoundError
#   - InvalidParameterError
#
#  Tested internal utilities:
#    - _validate_columns
#    - _validate_non_empty_list
#    - _sample_if_needed
#    - _limit_columns_for_plotting
#    - _create_subplots
#
#  Tested main functions:
#   1. validate_columns
#   2. validate_non_empty_list
#   3. limit_columns_for_plotting
#   4. sample_if_needed
#   5. create_subplots
#   6. dataframe_overview
#   7. data_types_overview
#   8. plot_missing_values
#   9. plot_correlation_matrix
#   10. plot_target_distribution
#   11. plot_numerical_by_target
#   12. plot_categorical_distribution
#   13. plot_numerical_distributions
#   14. export_plot_to_html
#   15. detect_outliers_iqr
#   16. normality_test
#   17. skewness_kurtosis_overview
#   18. generate_eda_report
#   19. plot_top_frequencies
#
#  Fixtures:
#   - sample_df: Provides a mock dataset for testing EDA functions.
#
#  Notes:
#   - Tests use pytest for structured and maintainable validation.
#   - Warnings related to Seaborn deprecations are monitored but do not break tests.
#
#  Author: Flavio Aguirre
#  Date: 2025-08-05

# ===========================================================================
# Imports
# ===========================================================================
import pytest
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # To avoid problems in CI
from src.eda import (
    # Custom Exceptions
    EDAError,
    InvalidDataFrameError,
    ColumnNotFoundError,
    InvalidParameterError,

    # Configuration
    config,

    # Internal Utilities
    _save_plot,
    _validate_columns,
    _validate_non_empty_list,
    _limit_columns_for_plotting,
    _sample_if_needed,
    _create_subplots,

    # Main EDA Functions
    dataframe_overview,
    data_types_overview,
    plot_missing_values,
    plot_correlation_matrix,
    plot_target_distribution,
    plot_numerical_by_target,
    plot_categorical_distribution,
    plot_numerical_distributions,
    detect_outliers_iqr,
    normality_test,
    export_plot_to_html,
    skewness_kurtosis_overview,
    generate_eda_report,
    plot_top_frequencies,
)

# ================================================================
# Fixtures configurations
# ================================================================
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [10, 20, 30, 40, 50],
        "cat1": ["A", "B", "A", "B", "C"],
        "target": [0, 1, 0, 1, 0]
    })

@pytest.fixture
def empty_df():
    return pd.DataFrame()

@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "num1": [1, None, 3, 4, None],
        "num2": [10, 20, None, 40, 50],
        "cat1": ["A", None, "A", "B", "C"]
    })


# ================================================================
# Unit tests (internal functions)
# ================================================================
def test_validate_columns_success(sample_df):
    """
    Test that _validate_columns successfully validates existing columns.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - No exception is raised when all columns exist in the DataFrame.
    """
    _validate_columns(sample_df, ["num1", "cat1"])

def test_validate_columns_fail(sample_df):
    """
    Test that _validate_columns raises ColumnNotFoundError for missing columns.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - Raises ColumnNotFoundError when a specified column does not exist.
    """
    with pytest.raises(ColumnNotFoundError):
        _validate_columns(sample_df, ["missing_col"])

def test_validate_non_empty_list_success():
    """
    Test that _validate_non_empty_list validates a non-empty list.

    Asserts
    -------
    - No exception is raised when the provided list is not empty.
    """
    _validate_non_empty_list(["col"], "test")

def test_validate_non_empty_list_fail():
    """
    Test that _validate_non_empty_list raises InvalidParameterError for an empty list.

    Asserts
    -------
    - Raises InvalidParameterError when the provided list is empty.
    """
    with pytest.raises(InvalidParameterError):
        _validate_non_empty_list([], "test")

def test_limit_columns_for_plotting():
    """
    Test that _limit_columns_for_plotting enforces the maximum number of columns.

    Asserts
    -------
    - The returned list of columns does not exceed the configured limit (40).
    """
    cols = [f"col_{i}" for i in range(50)]
    limited = _limit_columns_for_plotting(cols)
    assert len(limited) <= 40

def test_sample_if_needed(sample_df):
    """
    Test that _sample_if_needed correctly samples the DataFrame if it exceeds the sample size.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The returned DataFrame length is less than or equal to the configured sample size.
    """
    config.sample_size = 3
    df_sampled = _sample_if_needed(sample_df)
    assert len(df_sampled) <= 3

def test_create_subplots_single():
    """
    Test that _create_subplots creates the expected number of subplot axes.

    Asserts
    -------
    - A single subplot is created when only one column is provided.
    """
    fig, axes = _create_subplots(["col"])
    assert len(axes) == 1


# ======================================================================
# Unit tests of statistical functions
# ======================================================================
def test_normality_test(sample_df):
    """
    Test that normality_test computes normality statistics for numerical columns.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The result is a DataFrame.
    - The result contains the 'p_value' column or is empty if no numerical columns are available.
    """
    result = normality_test(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert "p_value" in result.columns or result.empty

def test_skewness_kurtosis_overview(sample_df):
    """
    Test that skewness_kurtosis_overview returns skewness and kurtosis values.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The result DataFrame contains 'skewness' and 'kurtosis' columns.
    """
    result = skewness_kurtosis_overview(sample_df)
    assert "skewness" in result.columns
    assert "kurtosis" in result.columns

def test_detect_outliers_iqr(sample_df):
    """
    Test that detect_outliers_iqr identifies outliers using the IQR method.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The result is an Index or NumPy array of outlier positions.
    """
    outliers = detect_outliers_iqr(sample_df, "num1")
    assert isinstance(outliers, (pd.Index, np.ndarray))


# =========================================================================
# Unit tests of overview functions
# =========================================================================
def test_dataframe_overview(sample_df):
    """
    Test that dataframe_overview generates a DataFrame summary.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The result DataFrame contains at least 'dtype' and 'n_missing' columns.
    """
    result = dataframe_overview(sample_df)
    assert "dtype" in result.columns
    assert "n_missing" in result.columns

def test_data_types_overview(sample_df):
    """
    Test that data_types_overview correctly classifies DataFrame columns.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The result includes 'categorical_cols' and 'numerical_cols' keys.
    """
    result = data_types_overview(sample_df)
    assert "categorical_cols" in result
    assert "numerical_cols" in result

# =========================================================================
# Unit tests of plots functions
# =========================================================================

def test_plot_missing_values(sample_df):
    """
    Test that plot_missing_values generates a valid figure for missing values.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The returned figure object is not None.
    """
    fig, ax = plot_missing_values(sample_df)
    assert fig is not None

def test_plot_correlation_matrix(sample_df):
    """
    Test that plot_correlation_matrix generates a correlation heatmap.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The correlation matrix has at least one row.
    - The returned figure and axes are not None.
    """
    fig, ax, corr = plot_correlation_matrix(sample_df)
    assert corr.shape[0] > 0
    assert fig is not None
    assert ax is not None

def test_plot_target_distribution(sample_df):
    """
    Test that plot_target_distribution creates a target variable distribution plot.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The returned figure object is not None.
    """
    fig, ax = plot_target_distribution(sample_df, "target")
    assert fig is not None

def test_plot_numerical_by_target(sample_df):
    """
    Test that plot_numerical_by_target generates numerical feature distributions by target.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The number of axes matches the number of numerical columns provided.
    """
    fig, axes = plot_numerical_by_target(sample_df, ["num1"], "target")
    assert len(axes) == 1

def test_plot_categorical_distribution(sample_df):
    """
    Test that plot_categorical_distribution generates categorical feature distributions.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The number of axes matches the number of categorical columns provided.
    """
    fig, axes = plot_categorical_distribution(sample_df, ["cat1"])
    assert len(axes) == 1

def test_plot_numerical_distributions(sample_df):
    """
    Test that plot_numerical_distributions generates plots for all numerical features.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.

    Asserts
    -------
    - The number of axes matches the number of numerical columns provided.
    """
    fig, axes = plot_numerical_distributions(sample_df, ["num1", "num2"])
    assert len(axes) == 2


# =========================================================================
# Unit test of export plot to html functions
# =========================================================================
def test_export_plot_to_html(sample_df, tmp_path):
    """
    Test that export_plot_to_html saves a plot as an HTML file.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.
    tmp_path : pathlib.Path
        Temporary directory for testing file output.

    Asserts
    -------
    - The returned file path has a '.html' extension.
    - The file is successfully created in the temporary directory.
    """
    fig, _ = plot_missing_values(sample_df)
    config.out_dir = tmp_path
    html_path = export_plot_to_html(fig, "test_plot")
    assert html_path.endswith(".html")
    assert tmp_path.joinpath("test_plot.html").exists()

# =========================================================================
# Unit test generate eda report
# =========================================================================
def test_generate_eda_report(sample_df, tmp_path):
    """
    Test that generate_eda_report creates a complete EDA report.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Fixture providing a sample dataset.
    tmp_path : pathlib.Path
        Temporary directory for testing file output.

    Asserts
    -------
    - The returned report includes 'overview' and 'corr_plot' keys.
    - Expected HTML files (e.g., missing values and correlation matrix plots) are generated.
    """
    config.out_dir = tmp_path
    report = generate_eda_report(sample_df, target_col="target")
    assert "overview" in report
    assert "corr_plot" in report
    assert tmp_path.joinpath("missing_values.html").exists()
    assert tmp_path.joinpath("correlation_matrix.html").exists()


# =========================================================================
# Unit tests of plot_top_frequencies function
# =========================================================================
def test_plot_top_frequencies_basic():
    """
    Basic sanity check:
    - accepts a pandas Series
    - respects top_n
    - returns (fig, ax) without errors
    """
    # Arrange
    freq_series = pd.Series(
        [10, 5, 3, 1],
        index=["word_a", "word_b", "word_c", "word_d"],
        name="freq",
    )

    # Act
    fig, ax = plot_top_frequencies(
        freq_series,
        top_n=2,
        title="Test Top Frequencies",
        xlabel="Freq",
        ylabel="Token",
        horizontal=True,
    )

    # Assert: types
    assert isinstance(fig, plt.Figure)
    # Axes can be AxesSubplot or similar, but inheriting from Axes
    from matplotlib.axes import Axes
    assert isinstance(ax, Axes)

    # Assert: only top_n bars are plotted
    bars = [p for p in ax.patches]
    assert len(bars) == 2

    # Assert: the Y-axis labels correspond to the two most frequent
    yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    # Since freq_series was ordered [10, 5, 3, 1], top 2 = word_a, word_b
    assert "word_a" in yticklabels[0] or "word_a" in yticklabels[1]
    assert "word_b" in yticklabels[0] or "word_b" in yticklabels[1]

    plt.close(fig)


def test_plot_top_frequencies_invalid_input():
    """
    Should raise InvalidParameterError when freq_series is not a pandas Series.
    """
    # Arrange
    invalid_input = [("a", 1), ("b", 2)]

    # Act & Assert
    try:
        plot_top_frequencies(invalid_input)
        # if no exception is raised, the test should fail
        assert False, "Expected InvalidParameterError for non-Series input"
    except InvalidParameterError:
        # OK: the expected exception was raised
        pass