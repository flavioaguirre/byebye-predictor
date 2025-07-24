#                                   data_loader.py

# This module provides functions to load and save datasets in various formats
# including CSV, Excel, and JSON. It also includes validation functions for file existence
# and DataFrame structure, along with logging for tracking operations.

# =================================================================================
import os
import pandas as pd
import logging
from typing import Optional, Union, Dict, List

# Set up basic logger
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)



# =================================================================================
#                       1. Validation function for file existence

def validate_file_exists(path: str) -> None:
    """
    Validate if a file exists at the given path.

    ------
    Args:
        path (str): Path to the file to validate.
    Returns:
        None.
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        
    """
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    logger.info(f"File exists: {path}")





# =================================================================================
#                       2. Function to validate DataFrame columns

def validate_columns(df: pd.DataFrame, expected_cols: List[str]) -> None:
    """
    Validate if expected columns exist in a DataFrame.

    ------
    Args:
        df (pd.DataFrame): The DataFrame to validate.
        expected_cols (list): List of expected column names.
    Returns:
        None.
    Raises:
        ValueError: If any expected columns are missing in the DataFrame.

    """
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        logger.error(f"\nMissing columns: {missing}")
        raise ValueError(f"\nMissing columns: {missing}")
    logger.info("\nAll expected columns are present.")





# =================================================================================
#                       3. Function to upload CSV files

def load_csv(filepath: str, encoding: str = "utf-8", dtypes: Optional[dict] = None) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame, optionally with explicit dtypes.

    ------
    
    Args:
        filepath (str): Path to the CSV file.
        encoding (str): Encoding of the CSV file. Default is 'utf-8'.
        dtypes (dict, optional): Dictionary of column data types.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        pd.errors.EmptyDataError: If the CSV file is empty or invalid.
        Exception: If there is an error loading the CSV file.
        
    """
    validate_file_exists(filepath)
    try:
        df = pd.read_csv(filepath, encoding=encoding, dtype=dtypes)
        logger.info(f"\nDataset loaded from: {filepath} | Shape: {df.shape}\n")
        return df
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty or invalid CSV file: {filepath}")
        raise e
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise e





# =================================================================================
#                       4. Function to upload Excel files

def load_excel(filepath: str, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
    """
    Load an Excel file into a DataFrame.
    
    ------
    Args:
        filepath (str): Path to the Excel file.
        sheet_name (str or int): Name or index of the sheet to load. Default is the first sheet.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        Exception: If there is an error loading the Excel file.
        
    """
    validate_file_exists(filepath)
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        logger.info(f"Excel file loaded from: {filepath} | Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel: {e}")
        raise e





# =================================================================================
#                       5. Function to upload JSON files

def load_json(filepath: str) -> pd.DataFrame:
    """
    Load a JSON file into a DataFrame.
    
    ------
    Args:
        filepath (str): Path to the JSON file.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        Exception: If there is an error loading the JSON file.
        
    """
    validate_file_exists(filepath)
    try:
        df = pd.read_json(filepath)
        logger.info(f"JSON file loaded from: {filepath} | Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        raise e





# =================================================================================
#                       6. Function to validate DataFrame shape

def load_csv_from_url(url: str) -> pd.DataFrame:
    """
    Load CSV directly from a URL.

    ------
    Args:
        url (str): URL of the CSV file.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    Raises:
        Exception: If there is an error loading the CSV from the URL.
        
    """
    try:
        df = pd.read_csv(url)
        logger.info(f"CSV loaded from URL: {url} | Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV from URL: {e}")
        raise e





# =================================================================================
#                       7. Function previewing the dataframe

def preview_df(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Preview the first n rows of a DataFrame.

    ------ 
    Args:
        df (pd.DataFrame): The DataFrame to preview.
        n (int): Number of rows to preview. Default is 5.
    Returns:
        pd.DataFrame: The first n rows of the DataFrame.
    Raises:
        ValueError: If the DataFrame is empty or n is not a positive integer.
        
    """
    logger.info(f"Previewing dataset: shape={df.shape}")
    return df.head(n)





# =================================================================================
#                       8. Function to load datasets from various sources

def load_multiple_datasets(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple datasets from given paths using the appropriate reader
    based on file extension (.csv, .json, .xlsx).

    ------
    Args:
        paths (dict): A dictionary where keys are dataset names and values are file paths.
    Returns:
        dict: A dictionary of loaded pandas DataFrames.
    Raises:
        ValueError: If file extension is not supported.
        Exception: For other loading errors (delegated from individual loaders).
    """
    datasets = {}

    for name, path in paths.items():
        logger.info(f"Loading dataset '{name}' from: {path}")
        ext = os.path.splitext(path)[-1].lower()

        try:
            if ext == ".csv":
                df = load_csv(path)
            elif ext == ".json":
                df = load_json(path)
            elif ext in [".xlsx", ".xls"]:
                df = load_excel(path)
            else:
                logger.error(f"Unsupported file extension for {name}: {ext}")
                raise ValueError(f"Unsupported file extension: {ext}")
            
            datasets[name] = df
        except Exception as e:
            logger.error(f"Failed to load dataset '{name}': {e}")
            raise e

    return datasets





# =================================================================================
#                       9. Function to list all CSV files in a directory

def list_csv_files(directory: str) -> List[str]:
    """
    List all CSV files in a given directory.

    ------
    Args:
        directory (str): Path to the directory to search for CSV files.
    Returns:
        list: List of paths to CSV files in the directory.
    Raises:
        FileNotFoundError: If the directory does not exist.
        
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")

    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    logger.info(f"Found {len(csv_files)} CSV files in {directory}")
    
    return csv_files





# =================================================================================
#                      10. Function to save DataFrame to CSV

def save_df_to_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Save a DataFrame to CSV.

    ------
    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): Path where the CSV file will be saved.
        index (bool): Whether to write row names (index). Default is False.
    Returns:
        None.
    Raises:
        Exception: If there is an error saving the DataFrame to CSV.
        
    """
    try:
        df.to_csv(path, index=index)
        logger.info(f"DataFrame saved to {path} | Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        raise e





# =================================================================================
#                      11. Function to save DataFrame to Excel

def save_df_to_excel(df: pd.DataFrame, path: str, sheet_name: str = "Sheet1", index: bool = False) -> None:
    """
    Save a DataFrame to an Excel file.

    ------
    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): Path where the Excel file will be saved.
        sheet_name (str): Name of the sheet in the Excel file. Default is 'Sheet1'.
        index (bool): Whether to write row names (index). Default is False.
    Returns:
        None.
    Raises:
        Exception: If there is an error saving the DataFrame to Excel.
        
    """
    try:
        df.to_excel(path, sheet_name=sheet_name, index=index)
        logger.info(f"DataFrame saved to {path} | Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving Excel: {e}")
        raise e





# =================================================================================
#                      12. Function to save DataFrame to JSON

def save_df_to_json(df: pd.DataFrame, path: str, orient: str = "records", indent: Optional[int] = None) -> None:
    """
    Save a DataFrame to a JSON file.

    ------
    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): Path where the JSON file will be saved.
        orient (str): Format of the JSON string. Default is 'records'.
        indent (int, optional): Number of spaces to use for indentation. Default is None.
    Returns:
        None.
    Raises:
        Exception: If there is an error saving the DataFrame to JSON.
        
    """
    try:
        df.to_json(path, orient=orient, indent=indent)
        logger.info(f"DataFrame saved to {path} | Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        raise e





# =================================================================================
#                    13. Function to save DataFrame with auto-detection of format

def save_df_auto(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Save a DataFrame to a file, automatically detecting the format from the file extension.

    Supports .csv, .json, and .xlsx formats.

    ------
    Args:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): Destination file path (must include extension).
        index (bool): Whether to write row indices. Default is False.

    Raises:
        ValueError: If the file extension is unsupported.
        Exception: For other I/O errors.
        
    """
    ext = os.path.splitext(filepath)[-1].lower()

    try:
        if ext == ".csv":
            df.to_csv(filepath, index=index)
        elif ext == ".json":
            df.to_json(filepath, orient="records", lines=True)
        elif ext in [".xlsx", ".xls"]:
            df.to_excel(filepath, index=index)
        else:
            logger.error(f"Unsupported file extension for saving: {ext}")
            raise ValueError(f"Unsupported file extension: {ext}")

        logger.info(f"DataFrame saved to: {filepath}")

    except Exception as e:
        logger.error(f"Failed to save DataFrame to {filepath}: {e}")
        raise e
