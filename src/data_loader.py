# ================================================================
#  src/data_loader.py
# ================================================================
#  This module provides functions for loading, saving, and manipulating datasets.
#  It supports various file formats including CSV, Excel, and JSON.

# Custom Exceptions:
#   - DataLoaderError
#   - UnsupportedFileTypeError
#   - InvalidDataFrameError
#   - URLNotAccessibleError

#  Internal utilities:
#    - _validate_file_exists
#    - _validate_dataframe
#    - _ensure_out_dir_exists
#    - _timestamped_path
#    - _validate_url_accessible
#    - _download_telco_dataset

#  Decorators:
#    - log_operation: Logs the start and end of a function operation.

#  Main functions:
#    1. load_csv
#    2. load_excel
#    3. load_json
#    4. load_csv_from_url
#    5. load_multiple_datasets
#    6. list_csv_files
#    7. preview_df
#    8. save_df
#    9. loader_telco_data
#    10. fetch_reddit_comments
#    11. loader_reddit_comments
#    12. save_metadata


#  Author: Flavio Aguirre
#  Date: 2025-08-01

# ================================================================
# Imports
# ================================================================
# ── Standard Library ───────────────────────────────────────────
import os
import subprocess
import sys
import warnings
from datetime import datetime
from typing import Callable, Optional, Literal, List, Dict, Tuple, Union, Any

# ── Third-Party Libraries ──────────────────────────────────────
from pathlib import Path
import requests
import pandas as pd
import praw

# ── Local Modules ──────────────────────────────────────────────
from utils import get_logger, add_project_root_to_path  # type: ignore


# ================================================================
# Additional Setup
# ================================================================
# Ensure project root is added to sys.path
add_project_root_to_path()

# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ================================================================
# Logger
# ================================================================
logger = get_logger(__name__)


# ================================================================
# Custom Exceptions
# ================================================================
class DataLoaderError(Exception):
    """Base exception for DataLoader errors."""


class UnsupportedFileTypeError(DataLoaderError):
    """Raised when a file type is not supported."""


class InvalidDataFrameError(DataLoaderError):
    """Raised when a DataFrame is invalid or empty."""


class URLNotAccessibleError(DataLoaderError):
    """Raised when a URL is not accessible or returns an error."""


# ================================================================
# Configuration
# ================================================================
class DataLoaderConfig:
    """
    Configuration class for data loading and saving operations.

    Attributes
    ----------
    out_dir : Path
        Default directory for saving exported datasets.
    encoding : str
        Default encoding for file I/O operations.
    """

    def __init__(self, out_dir: Union[str, Path] = "data/raw", encoding: str = "utf-8"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.encoding = encoding


config = DataLoaderConfig(out_dir=Path(__file__).resolve().parent.parent / "data/raw")


# ================================================================
# Internal utilities
# ================================================================
def _validate_file_exists(path: Union[str, Path]) -> None:
    """
    [Internal] Validate if a file exists at the given path.

    Parameters
    ----------
    path : str or Path
        Path to the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")


def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    [Internal] Validate if the input is a valid DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    InvalidDataFrameError
        If the DataFrame is invalid or empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataFrameError("Input must be a pandas DataFrame.")
    if df.empty:
        raise InvalidDataFrameError("DataFrame is empty.")


def _ensure_out_dir_exists() -> None:
    """
    [Internal] Ensure the output directory exists. Creates it if missing.
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)


def _validate_url_accessible(url: str) -> None:
    """
    [Internal] Validate if a URL is accessible.

    Parameters
    ----------
    url : str
        URL to check.

    Raises
    ------
    URLNotAccessibleError
        If the URL is not accessible or returns an error.
        
    """
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        if response.status_code >= 400:
            raise URLNotAccessibleError(f"URL not accessible: {url}")
    except requests.RequestException:
        raise URLNotAccessibleError(f"URL not accessible: {url}")


def _timestamped_path(filename: str) -> Path:
    """
    [Internal] Generate a file path with a timestamp for saving datasets.

    Parameters
    ----------
    filename : str
        File name (without path).

    Returns
    -------
    Path
        Full file path including timestamp prefix.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.out_dir / f"{timestamp}_{filename}"


def _download_telco_dataset() -> str:
    dataset_slug = "blastchar/telco-customer-churn"
    file_name = "telco_customer_churn.csv"
    os.makedirs(config.out_dir, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(dataset_slug, file_name, path=config.out_dir)
        return os.path.join(config.out_dir, file_name)
    except Exception as e:
        logger.error(f"Kaggle API failed: {e}. Trying fallback URL...")
        fallback_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = load_csv_from_url(fallback_url)
        local_path = os.path.join(config.out_dir, file_name)
        df.to_csv(local_path, index=False)
        return local_path


# ================================================================
# Decorators
# ================================================================
def log_operation(func) -> Callable:
    """
    [Internal] Decorator to log the start and end of a function operation.

    Parameters
    ----------
    func : Callable
        Function to be decorated.

    Returns
    -------
    Callable
        Wrapped function with logging.

    Raises
    ------
    Exception
        If the wrapped function raises an exception, it will be logged.
    """
    def wrapper(*args, **kwargs):
        logger.debug(f"Starting: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Completed: {func.__name__}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


# ================================================================
# 1. load_csv
# ================================================================
@log_operation
def load_csv(filepath: Union[str, Path], dtypes: Optional[dict] = None) -> pd.DataFrame:
    """    
    Load a CSV file into a pandas DataFrame.
    This function reads a CSV file and returns a DataFrame. It can also accept
    a dictionary of column names and their data types for type inference.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    dtypes : dict, optional
        Dictionary of column names and their data types. Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    InvalidDataFrameError
        If the loaded DataFrame is invalid or empty.
    """
    _validate_file_exists(filepath)
    df = pd.read_csv(filepath, encoding=config.encoding, dtype=dtypes)
    _validate_dataframe(df)
    logger.info(f"Loaded CSV: {filepath} | Shape: {df.shape}")
    return df


# ================================================================
# 2. load_excel
# ================================================================
@log_operation
def load_excel(filepath: Union[str, Path], sheet_name: Union[str, int] = 0) -> pd.DataFrame:
    """
    Load an Excel file into a pandas DataFrame.
    This function reads an Excel file and returns a DataFrame. It can also accept
    a specific sheet name or index to load.

    Parameters
    ----------
    filepath : str or Path
        Path to the Excel file.
    sheet_name : str or int, optional
        Name or index of the sheet to load. Default is 0 (first sheet).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    InvalidDataFrameError
        If the loaded DataFrame is invalid or empty.
    """
    _validate_file_exists(filepath)
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    _validate_dataframe(df)
    logger.info(f"Loaded Excel: {filepath} | Sheet: {sheet_name} | Shape: {df.shape}")
    return df
    
    
# ================================================================
# 3. load_json
# ================================================================
@log_operation
def load_json(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a JSON file into a pandas DataFrame.
    This function reads a JSON file and returns a DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the JSON file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    InvalidDataFrameError
        If the loaded DataFrame is invalid or empty.
    """
    _validate_file_exists(filepath)
    df = pd.read_json(filepath)
    _validate_dataframe(df)
    logger.info(f"Loaded json: {filepath} | Shape: {df.shape}")
    return df


# ================================================================
# 4. load_csv_from_url
# ================================================================
@log_operation
def load_csv_from_url(url: str) -> pd.DataFrame:
    """
    Load a CSV file from a URL into a pandas DataFrame.
    This function fetches a CSV file from a given URL and returns a DataFrame.

    Parameters
    ----------
    url : str
        URL of the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.

    Raises
    ------
    URLNotAccessibleError
        If the URL is not accessible or returns an error.
    InvalidDataFrameError
        If the loaded DataFrame is invalid or empty.
    """
    _validate_url_accessible(url)
    df = pd.read_csv(url, encoding=config.encoding)
    _validate_dataframe(df)
    logger.info(f"Loaded CSV from: {url} | Shape: {df.shape}")
    return df


# ================================================================
# 5. load_multiple_datasets
# ================================================================
@log_operation
def load_multiple_datasets(paths: Dict[str, Union[str, Path]]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple datasets from various file paths into a dictionary of DataFrames.
    This function accepts a dictionary where keys are dataset names and values are file paths.
    It supports CSV, Excel, and JSON file formats.

    Parameters
    ----------
    paths : dict
        Dictionary with dataset names as keys and file paths as values.

    Returns
    -------
    dict
        Dictionary with dataset names as keys and DataFrames as values.

    Raises
    ------
    ValueError
        If paths is not a dictionary or is empty.
    FileNotFoundError
        If any file path does not exist.
    UnsupportedFileTypeError
        If a file type is not supported.
    InvalidDataFrameError
        If any loaded DataFrame is invalid or empty.
        
    """
    if not isinstance(paths, dict) or not paths:
        raise ValueError("Paths must be a non-empty dictionary.")
    datasets = {}
    
    for name, path in paths.items():
        _validate_file_exists(path)
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            datasets[name] = load_csv(path)
        elif ext in [".xlsx", ".xls"]:
            datasets[name] = load_excel(path)
        elif ext == ".json":
            datasets[name] = load_json(path)
        else:
            raise UnsupportedFileTypeError(f"Unsupported file extension: {ext}")
    logger.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    return datasets


# ================================================================
# 6. list_csv_files
# ================================================================
def list_csv_files(directory: Union[str, Path]) -> List[Path]:
    """
    List all CSV files in a specified directory.
    This function returns a list of all CSV files in the given directory.

    Parameters
    ----------
    directory : str or Path
        Directory path to search for CSV files.

    Returns
    -------
    List[Path]
        List of Paths to CSV files in the directory.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If the output directory is not set in the configuration.
    """
    if not isinstance(directory, Path):
        directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return list(directory.glob("*.csv"))


# ================================================================
# 7. preview_df
# ================================================================
def preview_df(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Preview the first n rows of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to preview.
    n : int, optional
        Number of rows to display. Default is 5.

    Returns
    -------
    pd.DataFrame
        The first n rows of the DataFrame.

    Raises
    ------
    InvalidDataFrameError
        If the DataFrame is invalid or empty.
    ValueError
        If n is not a positive integer.
    """
    _validate_dataframe(df)
    if n <= 0:
        raise ValueError("n must be positive.")
    return df.head(n)


# ================================================================
# 8. save_df
# ================================================================
@log_operation
def save_df(df: pd.DataFrame,filename: str,fmt: Literal["csv", "excel", "json"],**kwargs) -> Path:
    """
    Save a DataFrame to a file in the specified format (CSV, Excel, JSON).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Name of the file to save the DataFrame to (without extension).
    fmt : Literal["csv", "excel", "json"]
        Format to save the DataFrame in. Supported formats: "csv", "excel", "json".
    **kwargs : dict
        Additional keyword arguments for the saving function (e.g., index, orient, indent).
        
    Returns
    -------
    Path
        Path to the saved file.

    Raises
    ------
    ValueError
        If the filename is not a non-empty string or if the format is unsupported.
    InvalidDataFrameError
        If the DataFrame is invalid or empty.
    UnsupportedFileTypeError
        If the specified format is not supported.
    """
    _validate_dataframe(df)
    _ensure_out_dir_exists()
    if not isinstance(filename, str) or not filename:
        raise ValueError("Filename must be a non-empty string.")
    if fmt not in ["csv", "excel", "json"]:
        raise UnsupportedFileTypeError(f"Unsupported format: {fmt}")

    if fmt == "csv":
        path = _timestamped_path(f"{filename}.csv")
        df.to_csv(path, index=kwargs.get("index", False))
    elif fmt == "excel":
        path = _timestamped_path(f"{filename}.xlsx")
        df.to_excel(path, index=kwargs.get("index", False))
    elif fmt == "json":
        path = _timestamped_path(f"{filename}.json")
        df.to_json(path, orient=kwargs.get("orient", "records"), indent=kwargs.get("indent"))

    logger.info(f"DataFrame saved: {path} | Shape: {df.shape}")
    return path


# ================================================================
# 9. loader_telco_data
# ================================================================
def loader_telco_data() -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset, downloading it if not found locally.

    Parameters
    ----------
    path : str
        Path where the Telco dataset should be located or downloaded.

    Returns
    -------
    pd.DataFrame
        Loaded Telco Customer Churn dataset.

    Raises
    ----
    DataLoaderError
        If the dataset cannot be loaded or is empty.
    """
    file_path = Path(config.out_dir / "telco_customer_data.csv")
    if not file_path.exists():
        logger.warning("Telco dataset not found locally. Downloading from Kaggle or fallback URL...")
        file_path = _download_telco_dataset()  # type: ignore

    try:
        df = load_csv(file_path)
        if df.empty:
            raise DataLoaderError("Telco dataset is empty.")
        logger.info(f"Telco data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        raise DataLoaderError(f"Failed to load Telco dataset: {e}")


# ================================================================
# 10. fetch_reddit_comments
# ================================================================
def fetch_reddit_comments(
    url: str | None = None,
    reddit: praw.Reddit | None = None,
    cache_path: str = f"{config.out_dir}/public_comments.csv"
) -> pd.DataFrame:
    """
    Fetch Reddit comments from a given URL using the PRAW API.
    Uses a cached file if available, otherwise fetches and caches the comments.

    Parameters
    ----------
    url : str, optional
        Reddit submission URL. If None, uses config.REDDIT_URL.
    reddit : praw.Reddit, optional
        Authenticated PRAW Reddit instance. If None, builds one from config.
    cache_path : str, optional
        Path to cache the comments CSV file. Default is 'data/raw/public_comments.csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Reddit comments.

    Raises
    ------
    URLNotAccessibleError
        If the Reddit URL is not accessible.
    DataLoaderError
        If comments cannot be fetched or are empty.
    """
    if url is None:
        url = "https://www.reddit.com/r/argentina/comments/1i924b2/movistar_av%C3%ADspense/"
    if reddit is None:
        reddit = praw.Reddit(
            client_id=config.reddit_client_id,
            client_secret=config.reddit_client_secret,
            user_agent=config.reddit_user_agent,
        )

    _validate_url_accessible(url)

    if os.path.exists(cache_path):
        try:
            df = load_csv(cache_path)
            if not df.empty:
                logger.info(f"Loaded cached Reddit comments from {cache_path} with shape {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"Failed to load cached file {cache_path}, refetching... Error: {e}")

    try:
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=None)

        comments_info = []
        for comment in submission.comments.list():
            comments_info.append({
                "comment_id": comment.id,
                "body": comment.body,
                "author": str(comment.author),
                "created_utc": comment.created_utc,
                "score": comment.score,
                "parent_id": comment.parent_id,
                "is_submitter": comment.is_submitter,
            })

        df = pd.DataFrame(comments_info)
        if df.empty:
            raise DataLoaderError("No comments extracted from Reddit.")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Reddit comments fetched and cached at {cache_path} with shape {df.shape}")

        return df
    except Exception as e:
        raise DataLoaderError(f"Failed to fetch Reddit comments: {e}")



# ================================================================
# 11. loader_reddit_comments
# ================================================================
def loader_reddit_comments() -> pd.DataFrame:
    """
    Load the Reddit comments dataset, downloading it if not found locally.

    Returns
    -------
    pd.DataFrame
        Loaded Reddit comments dataset.

    Raises
    ----
    DataLoaderError
        If the dataset cannot be loaded or is empty.
    """
    file_path = Path(config.out_dir / "public_comments.csv")
    if not file_path.exists():
        logger.warning("Reddit comments dataset not found locally. Downloading from Kaggle or fallback URL...")  # type: ignore

    try:
        df = load_csv(file_path)
        if df.empty:
            raise DataLoaderError("Reddit comments dataset is empty.")
        logger.info(f"Reddit comments data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        raise DataLoaderError(f"Failed to load Reddit comments dataset: {e}")

# ================================================================
# 12. save_metadata
# ================================================================
def save_metadata(df: pd.DataFrame, filename: str):
    """
    Generate a simple data dictionary (in markdown format) for documentation purposes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame for which to generate the data dictionary.
    filename : str
        Name of the markdown file to save the data dictionary.

    Raises
    ------
    InvalidDataFrameError
        If the DataFrame is invalid or empty.
    """
    _validate_dataframe(df)
    dict_path = os.path.join(config.out_dir, filename)
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("# Data Dictionary\n\n")
        for col in df.columns:
            f.write(f"## {col}\n")
            f.write(f"- dtype: {df[col].dtype}\n")
            f.write(f"- nulls: {df[col].isna().sum()} ({df[col].isna().mean()*100:.2f}%)\n")
            f.write(f"- uniques: {df[col].nunique()}\n\n")
    logger.info(f"Data dictionary saved at {dict_path}")


# ================================================================
# Usage
# ================================================================
# Step 1: Load Structured Data (Telco)
# ================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Step 1: Load Structured Data (Telco)")
    print("="*60)
    df_telco = loader_telco_data()

    print(f"Telco dataset shape: {df_telco.shape}")
    print(preview_df(df_telco))

    # Save data metadata
    save_metadata(df_telco, "metadata_telco_customer_churn.md")

    # ================================================================
    # Step 2: Load Unstructured Data (Reddit)
    # ================================================================
    print("\n" + "="*60)
    print("Step 2: Load Unstructured Data (Reddit)")
    print("="*60)
    reddit = praw.Reddit(
        client_id="JXtHcKXMFMcrgRGTkQdqdA",   # replace with authentication credentials
        client_secret="YBN-CAYX5b7cJzgqc6jXUwBwcACXIw",
        user_agent="scraper_app"
    )

    URL = "https://www.reddit.com/r/argentina/comments/1i924b2/movistar_av%C3%ADspense/"
    df_reddit = fetch_reddit_comments(URL, reddit)

    print(f"Reddit dataset shape: {df_reddit.shape}")
    print(preview_df(df_reddit))

    reddit_path = os.path.join(config.out_dir, "public_comments.csv")

    # Save metadata
    save_metadata(df_reddit, "metadata_public_comments.md")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*60)
    print("Data Acquisition Completed Successfully")
    print("="*60)
    telco_path = os.path.join(config.out_dir, "telco_customer_churn.csv")
    print(f"- Telco dataset: {df_telco.shape} saved at {telco_path}")
    print(f"- Reddit dataset: {df_reddit.shape} saved at {reddit_path}")