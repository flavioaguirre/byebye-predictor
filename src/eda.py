# ================================================================
#  src/eda.py
# ================================================================
#  EDA (Exploratory Data Analysis) utilities module.
#  Provides robust, scalable, and production-ready functions for:
#   Custom Exceptions:
#   - EDAError
#   - InvalidDataFrameError
#   - ColumnNotFoundError
#   - InvalidParameterError

#  Internal utilities:
#    - EDAConfig
#    - _save_plot
#    - _validate_columns
#    - _validate_non_empty_list
#    - _sample_if_needed
#    - _limit_columns_for_plotting
#    - _create_subplots

#  Decorators:
#    - auto_save_plot
#    - safe_eda

#  Main functions:
#   1. DataFrame Overview
#   2. Data Types Overview
#   3. Missing Values
#   4. Correlation Matrix
#   5. Target Distribution
#   6. Numerical Features by Target
#   7. Categorical Distribution
#   8. Numerical Distributions
#   9. Outlier Detection
#   10. Normality Test
#   11. Export plots
#   12. Skewness & Kurtosis Overview
#   13. Automated EDA Report

#  Author: Flavio Aguirre
#  Date: 2025-08-02

# ================================================================
# Imports
# ================================================================
# ── Standard Library ───────────────────────────────────────────
import os
import base64
from io import BytesIO
from datetime import datetime
from typing import Callable, Optional, Literal, List, Dict, Tuple, Union, Any

# ── Third-Party Libraries ──────────────────────────────────────
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest, shapiro

# ── Local Modules ──────────────────────────────────────────────
from src.utils import get_logger, add_project_root_to_path  # type: ignore
from src.data_loader import _validate_dataframe, log_operation  # type: ignore

# Ensure project root is added to sys.path (only needed once)
add_project_root_to_path()



# ================================================================
# Logger
# ================================================================
logger = get_logger(__name__)



# ================================================================
# Custom Exceptions
# ================================================================
class EDAError(Exception):
    """Base exception for EDA errors."""


class InvalidDataFrameError(EDAError):
    """Raised when an invalid or empty DataFrame is provided."""


class ColumnNotFoundError(EDAError):
    """Raised when specified columns are not found in the DataFrame."""


class InvalidParameterError(EDAError):
    """Raised when a parameter value is invalid."""



# ================================================================
# Configuration EDA Module
# ================================================================
class EDAConfig:
    """
    Configuration class for the EDA module.

    Attributes
    ----------
    out_dir : str
        Directory to save plots.
    style : str
        Seaborn style to apply globally.
    sample_size : Optional[int]
        Maximum number of rows to sample for visualizations (saves resources).

    Methods
    -------
    apply() -> None
        Applies the configured visualization style to Seaborn.
    
    Example
    -------
    >>> from src.eda import config
    >>> config.out_dir = "reports/figures/eda"
    >>> config.sample_size = 5000
    >>> config.apply()
    """

    def __init__(self, out_dir: str = "reports/figures/eda", style: str = "darkgrid", sample_size: Optional[int] = None):
        self.out_dir = out_dir
        self.style = style
        self.sample_size = sample_size

    def apply(self) -> None:
        """Applies the current configuration (e.g., seaborn style)."""
        sns.set_style(self.style)

config = EDAConfig()
config.apply()



# ================================================================
# Internal Utilities
# ================================================================
# ── save_plot ──────────────────────────────────────────────
@log_operation
def _save_plot(fig: plt.Figure, fig_name: str) -> None:
    """
    [Internal]
    Save a Matplotlib figure with a timestamp.
    Saves a figure as a PNG file in the configured output directory.
    Automatically creates the directory if it does not exist.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object to save.
    fig_name : str
        Base name for the saved figure file (timestamp is added automatically).

    Raises
    ------
    OSError
        If the figure cannot be saved due to file system errors (e.g. permission denied).
    """
    os.makedirs(config.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.out_dir, f"{fig_name}_{timestamp}.png")
    fig.savefig(path, bbox_inches="tight")
    logger.info(f"Image saved: {path}")


# ── validate_non_empty_list ──────────────────────────────────────────────
def _validate_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """
    [Internal]
    Validate that the specified columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    columns : list of str
        List of column names to check.

    Raises
    ------
    ColumnNotFoundError
        If one or more specified columns are not found in the DataFrame.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        logger.error(f"Columns not found: {missing}")
        raise ColumnNotFoundError(f"Columns not found in DataFrame: {missing}")


# ── Local Modules ──────────────────────────────────────────────
def _validate_non_empty_list(columns: List[str], name: str) -> None:
    """
    [Internal]
    Validate that a list of columns is not empty.

    Parameters
    ----------
    columns : list of str
        List of column names to validate.
    name : str
        Name to display in the warning message (e.g. "numerical columns").

    Raises
    ------
    InvalidParameterError
        If the list is empty.
    """
    if not columns:
        logger.error(f"The list of {name} is empty. No analysis can be performed.")
        raise InvalidParameterError(f"The list of {name} cannot be empty.")


# ── limit_columns_for_plotting ──────────────────────────────────────────────
MAX_PLOTS = 40  # Default
def _limit_columns_for_plotting(columns: List[str]) -> List[str]:
    """
    [Internal]
    Limit the number of columns for plotting to avoid performance issues.

    Parameters
    ----------
    columns : list of str
        List of column names to plot.

    Returns
    -------
    list of str
        Trimmed list of column names if the limit is exceeded.
    """
    if len(columns) > MAX_PLOTS:
        logger.warning(f"Too many columns ({len(columns)}). Limiting to the first {MAX_PLOTS}.")
        return columns[:MAX_PLOTS]
    return columns


# ── sample_if_needed ──────────────────────────────────────────────
def _sample_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Internal]
    Sample the DataFrame if its size exceeds the configured limit.
    If a `sample_size` is defined in `EDAConfig` and the DataFrame has more rows than this limit,
    a random sample of rows is returned. Otherwise, the original DataFrame is returned.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to sample.

    Returns
    -------
    pd.DataFrame
        The sampled DataFrame if sampling is required; otherwise, the original DataFrame.

    Raises
    ------
    InvalidDataFrameError
        If the input is not a valid DataFrame or is empty.
    """
    if config.sample_size and len(df) > config.sample_size:
        logger.warning(f"Sampling {config.sample_size} rows for visualization.")
        return df.sample(config.sample_size, random_state=42)
    return df


# ── create_subplots ──────────────────────────────────────────────
def _create_subplots(cols: List[str], figsize=(8, 4)) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    [Internal]
    Create a series of stacked subplots for a list of features.
    Automatically adjusts the figure size based on the number of features.

    Parameters
    ----------
    cols : list of str
        List of feature names for which subplots will be created.
    figsize : tuple of (float, float), optional
        Base figure size for a single subplot. Default is (8, 4).

    Returns
    -------
    tuple
        (fig, axes):
        - fig: Matplotlib Figure object.
        - axes: List of Matplotlib Axes objects (or a single-element list if only one column is provided).
    """
    fig, axes = plt.subplots(len(cols), 1, figsize=(figsize[0], figsize[1] * len(cols)))
    if len(cols) == 1:
        axes = [axes]
    return fig, axes



# ================================================================
# Decorators
# ================================================================
# ── auto_save_plot ──────────────────────────────────────────────
def auto_save_plot(func) -> Callable:
    """
    Decorator to automatically save plots returned by a plotting function.
    This decorator wraps plotting functions that return (fig, ax) or (fig, axes)
    and saves the figure to the configured output directory if `save=True` is provided.

    Parameters
    ----------
    func : Callable
        The plotting function to wrap. It must return a tuple containing a Matplotlib Figure
        and one or more Axes objects.

    Returns
    -------
    Callable
        Wrapped function that adds automatic plot-saving functionality.

    Raises
    ------
    OSError
        If the plot cannot be saved to the configured output directory.
    """
    def wrapper(*args, save: bool = False, fig_name: Optional[str] = None, **kwargs):
        result = func(*args, **kwargs)

        # Si la función devuelve (fig, ax, *otros)
        if isinstance(result, tuple) and len(result) >= 2 and hasattr(result[0], "savefig"):
            fig, ax, *rest = result
            if save:
                file_path = os.path.join(config.out_dir, f"{fig_name or func.__name__}.png")
                fig.savefig(file_path)
                logger.info(f"Plot saved: {file_path}")
            return (fig, ax, *rest)
        else:
            return result

    return wrapper

# ── safe_eda ──────────────────────────────────────────────
def safe_eda(func: Callable) -> Callable:
    """
    Decorator for safe execution of EDA functions.
    Catches known and unknown exceptions, logs them, and re-raises with EDA-specific context.

    Parameters
    ----------
    func : Callable
        EDA function to decorate.

    Returns
    -------
    Callable
        Wrapped function with error handling.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EDAError as e:
            logger.error(f"EDA error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}")
            raise EDAError(f"Unexpected error in {func.__name__}") from e
    return wrapper



# ================================================================
# 1. DataFrame Overview
# ================================================================
@log_operation
def dataframe_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary overview of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        - dtype: Data type of each feature.
        - n_unique: Number of unique values.
        - n_missing: Number of missing values.
        - %_missing: Percentage of missing values.

    Raises
    ------
    InvalidDataFrameError
        If the input is not a valid DataFrame or is empty.
    """
    _validate_dataframe(df)
    logger.info("Generating DataFrame overview.")
    return (
        pd.DataFrame({
            "dtype": df.dtypes,
            "n_unique": df.nunique(),
            "n_missing": df.isna().sum(),
            "%_missing": df.isna().mean() * 100,
        })
        .sort_values(by="%_missing", ascending=False)
    )


# ================================================================
# 2. Data Types Overview
# ================================================================
@log_operation
def data_types_overview(df: pd.DataFrame, return_df: bool = False) -> pd.DataFrame | Dict:
    """
    Categorize DataFrame columns into data type groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    dict
        A dictionary with:
        - categorical_cols: List of categorical column names.
        - numerical_cols: List of numerical column names.
        - binary_numeric_cols: List of binary numerical column names.
        - continuous_numeric_cols: List of continuous numerical column names.

    Raises
    ------
    InvalidDataFrameError
        If the input is not a valid DataFrame or is empty.
    """
    _validate_dataframe(df)
    logger.info("Analyzing data types.")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    binary = [col for col in num_cols if df[col].nunique() == 2]
    continuous = [col for col in num_cols if col not in binary]

    result_dict = {
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
        "binary_numeric_cols": binary,
        "continuous_numeric_cols": continuous,
    }

    if return_df:
        return pd.DataFrame(
            [(col, t) for t, cols in result_dict.items() for col in cols],
            columns=["column_name", "data_type"]    #type: ignore
        )

    return result_dict


# ================================================================
# 3. Missing Values
# ================================================================
@log_operation
@auto_save_plot
@safe_eda
def plot_missing_values(df: pd.DataFrame, threshold: float = 0.0) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot missing values percentages for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    threshold : float, optional
        Minimum percentage of missing values to display (between 0 and 1). Default is 0.0.

    Returns
    -------
    tuple
        (fig, ax):
        - fig: Matplotlib Figure object.
        - ax: Matplotlib Axes object.

    Raises
    ------
    InvalidDataFrameError
        If the input is not a valid DataFrame or is empty.
    InvalidParameterError
        If the threshold is not between 0 and 1.
    """
    _validate_dataframe(df)
    if not (0 <= threshold <= 1):
        raise InvalidParameterError("Threshold must be between 0 and 1.")
    df = _sample_if_needed(df)
    missing = df.isnull().mean().sort_values(ascending=False)   # type: ignore
    missing = pd.DataFrame(missing[missing > threshold])
    fig, ax = plt.subplots(figsize=(10, 6))
    if not missing.empty:
        sns.barplot(x=missing.values * 100, y=missing.index, palette="Reds_d", ax=ax)
        ax.set_xlabel("Percentage of Missing Values")
        ax.set_ylabel("Features")
        ax.set_title("Missing Data by Feature")
    else:
        logger.info("No missing values above threshold.")
    fig.tight_layout()
    return fig, ax


# ================================================================
# 4. Correlation Matrix
# ================================================================
@log_operation
@auto_save_plot
@safe_eda
def plot_correlation_matrix(
    df: pd.DataFrame,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    annot: bool = False,
    max_features: int = 40
    ) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plot a correlation matrix heatmap and return correlation data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    method : {"pearson", "kendall", "spearman"}, optional
        Correlation method to use. Default is "pearson".
    annot : bool, optional
        Whether to display correlation values in the heatmap. Default is False.
    max_features : int, optional
        Maximum number of features to include in the plot. Default is 40.

    Returns
    -------
    tuple
        - fig : matplotlib.figure.Figure
            The generated Matplotlib figure.
        - ax : matplotlib.axes.Axes
            The heatmap axes.
        - corr : pd.DataFrame
            Correlation matrix.

    Raises
    ------
    InvalidDataFrameError
        If the DataFrame is invalid or empty.
    """
    _validate_dataframe(df)
    df = _sample_if_needed(df)
    corr = df.select_dtypes(include=[np.number]).corr(method=method)

    if corr.shape[0] > max_features:
        logger.warning(f"Too many features for correlation matrix ({corr.shape[0]}). Trimming to top {max_features}.")
        corr = corr.iloc[:max_features, :max_features]

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, annot=annot, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    ax.set_title(f"{method.capitalize()} Correlation Matrix")
    fig.tight_layout()
    return fig, ax, corr


# ================================================================
# 5. Target Distribution
# ================================================================
@log_operation
@auto_save_plot
@safe_eda
def plot_target_distribution(df: pd.DataFrame, target_col: str) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the distribution of a target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    target_col : str
        The target column to visualize.

    Returns
    -------
    tuple
        (fig, ax):
        - fig: Matplotlib Figure object.
        - ax: Matplotlib Axes object.

    Raises
    ------
    InvalidDataFrameError
        If the input is not a valid DataFrame or is empty.
    ColumnNotFoundError
        If the target column is not present in the DataFrame.
    """
    _validate_dataframe(df)
    _validate_columns(df, [target_col])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, hue=target_col, palette="pastel", legend=False, ax=ax)
    ax.set_title(f"Distribution of Target: {target_col}")
    ax.grid(axis="y")
    return fig, ax


# ================================================================
# 6. Numerical Features by Target
# ================================================================
@log_operation
@auto_save_plot
@safe_eda
def plot_numerical_by_target(df: pd.DataFrame, numerical_cols: List[str], target_col: str)-> tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot numerical features against a target variable using box plots.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    numerical_cols : list of str
        List of numerical column names to plot.
    target_col : str
        The target column to group by.

    Returns
    -------
    tuple
        (fig, axes):
        - fig: Matplotlib Figure object.
        - axes: List of Matplotlib Axes objects.

    Raises
    ------
    InvalidDataFrameError
        If the input is not a valid DataFrame or is empty.
    ColumnNotFoundError
        If any of the numerical columns or the target column is not found.
    """
    _validate_dataframe(df)
    _validate_columns(df, numerical_cols + [target_col])
    fig, axes = _create_subplots(numerical_cols)
    for ax, col in zip(axes, numerical_cols):
        sns.boxplot(x=target_col, y=col, data=df, hue=target_col, palette="pastel", ax=ax, legend=False, orient="v")
        ax.set_title(f"{col} by {target_col}")
    fig.tight_layout()
    return fig, axes


# ================================================================
# 7. Categorical Distribution
# ================================================================
@log_operation
@auto_save_plot
@safe_eda
def plot_categorical_distribution(df: pd.DataFrame, cat_cols: List[str]) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot the distribution of categorical features.
    Creates count plots for each categorical column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    cat_cols : list of str
        List of categorical column names to plot.

    Returns
    -------
    tuple
        (fig, axes):
        - fig: Matplotlib Figure object.
        - axes: List of Matplotlib Axes objects.

    Raises
    ------
    InvalidDataFrameError
        If the input is not a valid DataFrame or is empty.
    ColumnNotFoundError
        If any of the categorical columns are not found in the DataFrame.
    """
    _validate_dataframe(df)
    _validate_columns(df, cat_cols)
    fig, axes = _create_subplots(cat_cols)
    for ax, col in zip(axes, cat_cols):
        sns.countplot(x=col, data=df, hue=col, palette="pastel", legend=False, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, axes


# ================================================================
# 8. Numerical Distributions
# ================================================================
@log_operation
@auto_save_plot
@safe_eda
def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: List[str]) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot the distributions of numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    numerical_cols : list of str
        List of numerical column names to plot.

    Returns
    -------
    tuple
        - fig : matplotlib.figure.Figure
            The generated Matplotlib figure.
        - axes : list of matplotlib.axes.Axes
            List of axes for each numerical column.

    Raises
    ------
    InvalidDataFrameError
        If the DataFrame is invalid or empty.
    ColumnNotFoundError
        If any of the specified columns do not exist in the DataFrame.
    InvalidParameterError
        If the list of numerical columns is empty.
    """
    _validate_dataframe(df)
    _validate_non_empty_list(numerical_cols, "numerical columns")
    _validate_columns(df, numerical_cols)

    numerical_cols = _limit_columns_for_plotting(numerical_cols)
    df = _sample_if_needed(df)

    fig, axes = _create_subplots(numerical_cols)
    for ax, col in zip(axes, numerical_cols):
        sns.histplot(df[col], kde=True, stat="density", bins=30, ax=ax, color="blue", alpha=0.5)
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    return fig, axes


# ================================================================
# 9. Outlier Detection
# ================================================================
@log_operation
@safe_eda
def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Index | np.ndarray:
    """
    Detect outliers in a numerical feature using the IQR method.
    Identifies outliers as any value below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    column : str
        The column to check for outliers.

    Returns
    -------
    pd.Index
        Index positions of rows identified as outliers.

    Raises
    ------
    InvalidDataFrameError
        If the input is not a valid DataFrame or is empty.
    ColumnNotFoundError
        If the specified column is not found in the DataFrame.
    """
    _validate_dataframe(df)
    _validate_columns(df, [column])
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df.index[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]


# ================================================================
# 10. Normality Test
# ================================================================
@log_operation
@safe_eda
def normality_test(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    method: Literal["dagostino", "shapiro"] = "dagostino"
    ) -> pd.DataFrame:
    """
    Perform normality tests for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    numerical_cols : list of str, optional
        List of numerical columns to test. If None, all numerical columns will be tested.
    method : {"dagostino", "shapiro"}, optional
        Statistical test for normality:
        - "dagostino": D'Agostino and Pearson's test (default).
        - "shapiro": Shapiro-Wilk test.

    Returns
    -------
    pd.DataFrame
        A DataFrame with:
        - feature : str
            Feature name.
        - statistic : float
            Test statistic.
        - p_value : float
            P-value of the test.
        - normal : bool
            Whether the feature distribution is likely normal (p > 0.05).

    Raises
    ------
    InvalidDataFrameError
        If the DataFrame is invalid or empty.
    ColumnNotFoundError
        If any specified numerical columns do not exist.
    InvalidParameterError
        If no numerical columns are found.
    """
    _validate_dataframe(df)
    cols = numerical_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    _validate_non_empty_list(cols, "numerical columns")
    _validate_columns(df, cols)

    results = []
    for col in cols:
        series = df[col].dropna()
        if len(series) < 8:
            logger.warning(f"Skipping normality test for '{col}' (n < 8).")
            continue
        stat, p = (normaltest(series) if method == "dagostino" else shapiro(series))
        results.append({"feature": col, "statistic": stat, "p_value": p, "normal": p > 0.05})
    df_result = pd.DataFrame(results)
    if df_result.empty:
        return df_result  # Devuelve DataFrame vacío sin intentar ordenar
    return df_result.sort_values(by="p_value")


# =============================================================================
# 11. Export plots 
# =============================================================================
@log_operation
@safe_eda
def export_plot_to_html(fig: plt.Figure, file_name: str) -> str:
    """
    Export a Matplotlib figure as an HTML file with embedded Base64 image.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to export.
    file_name : str
        Name of the output HTML file (without extension).

    Returns
    -------
    str
        Path to the generated HTML file.

    Raises
    ------
    OSError
        If the HTML file cannot be written.
    """
    os.makedirs(config.out_dir, exist_ok=True)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    html_path = os.path.join(config.out_dir, f"{file_name}.html")
    with open(html_path, "w") as f:
        f.write(f'<img src="data:image/png;base64,{img_base64}" />')
    logger.info(f"HTML report generated: {html_path}")
    return html_path


# ================================================================
# 12. Skewness & Kurtosis Overview
# ================================================================
@log_operation
@safe_eda
def skewness_kurtosis_overview(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
    """
    Compute skewness and kurtosis for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    numerical_cols : list of str, optional
        List of numerical columns to analyze. If None, all numerical columns are used.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - feature: str
            Feature name.
        - skewness: float
            Measure of distribution asymmetry.
        - kurtosis: float
            Measure of tail heaviness.

    Raises
    ------
    InvalidDataFrameError
        If the DataFrame is invalid or empty.
    ColumnNotFoundError
        If any of the specified numerical columns do not exist.
    InvalidParameterError
        If no numerical columns are found.
    """
    _validate_dataframe(df)
    cols = numerical_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    _validate_non_empty_list(cols, "numerical columns")
    _validate_columns(df, cols)

    logger.info("Calculating skewness and kurtosis.")

    result = pd.DataFrame({
        "feature": cols,
        "skewness": [df[col].skew() for col in cols],
        "kurtosis": [df[col].kurtosis() for col in cols],
    })

    result["_abs_skew"] = result["skewness"].abs()
    return result.sort_values(by="_abs_skew", ascending=False).drop(columns="_abs_skew")


# ================================================================
# 13. Automated EDA Report
# ================================================================
@log_operation
@safe_eda
def generate_eda_report(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    max_features: int = 30
    ) -> Dict[str, Union[pd.DataFrame, str, Any, None]]:
    """
    Generate an automated EDA report with key insights and exported plots.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    target_col : str, optional
        Target column for supervised EDA visualizations (e.g. target distribution, numerical by target).
        If None, target-specific visualizations will be skipped.
    max_features : int, optional
        Maximum number of features for correlation and distribution plots. Default is 30.

    Returns
    -------
    dict
        Dictionary containing:
        - overview : pd.DataFrame
            General DataFrame overview.
        - types : dict
            Data types overview (categorical, numerical, etc.).
        - missing_plot : str
            Path to missing values plot (HTML).
        - corr_plot : str
            Path to correlation matrix plot (HTML).
        - skew_kurt : pd.DataFrame
            Skewness and kurtosis summary.
        - normality : pd.DataFrame
            Normality test results.
        - numerical_dist_plot : str
            Path to numerical distributions plot (HTML).
        - target_plot : str (if target_col provided)
            Path to target distribution plot (HTML).

    Raises
    ------
    InvalidDataFrameError
        If the DataFrame is invalid or empty.
    ColumnNotFoundError
        If the target column does not exist (when provided).
    """
    _validate_dataframe(df)
    logger.info("Generating automated EDA report.")

    # 1. Overview
    overview = dataframe_overview(df)
    types = data_types_overview(df)

    # 2. Missing values plot
    fig, ax = plot_missing_values(df)
    missing_html = export_plot_to_html(fig, "missing_values")

    # 3. Correlation matrix
    fig, ax, _ = plot_correlation_matrix(df, max_features=max_features)
    corr_html = export_plot_to_html(fig, "correlation_matrix")

    # 4. Skewness & kurtosis
    skew_kurt = skewness_kurtosis_overview(df, types["numerical_cols"])

    # 5. Normality test
    normality = normality_test(df, types["numerical_cols"])

    # 6. Numerical distributions
    fig, axes = plot_numerical_distributions(df, types["numerical_cols"][:max_features])
    numerical_dist_html = export_plot_to_html(fig, "numerical_distributions")

    # 7. Target-specific plots
    target_html = None
    if target_col:
        _validate_columns(df, [target_col])
        fig, ax = plot_target_distribution(df, target_col)
        target_html = export_plot_to_html(fig, f"target_distribution_{target_col}")

    return {
        "overview": overview,
        "types": types,
        "missing_plot": missing_html,
        "corr_plot": corr_html,
        "skew_kurt": skew_kurt,
        "normality": normality,
        "numerical_dist_plot": numerical_dist_html,
        "target_plot": target_html,
    }


# =========================================================================