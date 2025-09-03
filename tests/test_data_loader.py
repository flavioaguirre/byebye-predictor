# ============================================================================
# tests/test_data_loader.py
# ============================================================================
# Tests for Data Loader Module in src.data_loader

# This module contains tests for the data loading and saving functions defined
# in src/data_loader.py. The tests cover:
#
# 1. Internal Utilities:
#    - _validate_file_exists
#    - _validate_dataframe
#    - _ensure_out_dir_exists
#    - _validate_url_accessible
#    - _timestamped_path
#
# 2. Main Functions:
#    - load_csv
#    - load_excel
#    - load_json
#    - load_csv_from_url
#    - load_multiple_datasets
#    - list_csv_files
#    - preview_df
#    - save_df
#    - fetch_reddit_comments
#    - loader_telco_data
#    - loader_reddit_comments
#
# Author: Flavio Aguirre
# Date: 2025-08-01

# ============================================================================

from pathlib import Path
import pandas as pd
import pytest
import requests
from unittest.mock import patch, MagicMock

from src.data_loader import (   # type: ignore
    _validate_file_exists,
    _validate_dataframe,
    _ensure_out_dir_exists,
    _validate_url_accessible,
    _timestamped_path,
    load_csv,
    load_excel,
    load_json,
    load_csv_from_url,
    load_multiple_datasets,
    list_csv_files,
    preview_df,
    save_df,
    fetch_reddit_comments,
    loader_telco_data,
    loader_reddit_comments,
    InvalidDataFrameError,
    URLNotAccessibleError,
    UnsupportedFileTypeError,
    DataLoaderError,
)


# ============================================================================
# 1. Tests: Internal Utilities
# ============================================================================

def test_validate_file_exists_raises(tmp_path: Path):
    """
    Test that _validate_file_exists raises FileNotFoundError for a missing file.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Asserts
    -------
    - FileNotFoundError is raised if the file does not exist.
    """
    missing_file = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        _validate_file_exists(missing_file)


def test_validate_dataframe_accepts_valid_df():
    """
    Test that _validate_dataframe accepts a valid DataFrame.

    Asserts
    -------
    - No exception is raised for a non-empty pandas DataFrame.
    """
    df = pd.DataFrame({"a": [1, 2]})
    _validate_dataframe(df)  # Should not raise


def test_validate_dataframe_rejects_non_dataframe():
    """    
    Test that _validate_dataframe raises InvalidDataFrameError for non-DataFrame input.

    Asserts
    -------
    - InvalidDataFrameError is raised when input is not a pandas DataFrame.
    """
    with pytest.raises(InvalidDataFrameError):
        _validate_dataframe("not_a_dataframe")  # type: ignore[arg-type]


def test_validate_dataframe_rejects_empty_df():
    """
    Test that _validate_dataframe raises InvalidDataFrameError for an empty DataFrame.

    Asserts
    -------
    - InvalidDataFrameError is raised when the DataFrame has no rows.
    """
    df = pd.DataFrame()
    with pytest.raises(InvalidDataFrameError):
        _validate_dataframe(df)


def test_ensure_out_dir_exists_creates_directory(tmp_path, monkeypatch):
    """
    Test that _ensure_out_dir_exists creates the output directory if it does not exist.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for isolated testing.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to isolate modifications to sys.path.

    Asserts
    -------
    - The output directory is created successfully.
    """
    from src.data_loader import config  # type: ignore

    monkeypatch.setattr(config, "out_dir", tmp_path / "exports")
    _ensure_out_dir_exists()
    assert config.out_dir.exists()


def test_validate_url_accessible_success(requests_mock):
    """
    Test that _validate_url_accessible does not raise for an accessible URL.

    Parameters
    ----------
    requests_mock : requests_mock.Mocker
        Pytest fixture used to mock HTTP requests.

    Asserts
    -------
    - No exception is raised for a valid URL returning status code 200.
    """
    requests_mock.head("http://fake-url.com", status_code=200)
    _validate_url_accessible("http://fake-url.com")


def test_validate_url_accessible_raises(monkeypatch):
    """
    Test that _validate_url_accessible raises URLNotAccessibleError for an inaccessible URL.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to isolate modifications to sys.path.

    Asserts
    -------
    - URLNotAccessibleError is raised when the URL is unreachable or returns an error.
    """
    mock_response = MagicMock(status_code=404)
    monkeypatch.setattr(requests, "head", lambda *_, **__: mock_response)
    with pytest.raises(URLNotAccessibleError):
        _validate_url_accessible("http://bad-url.com")


def test_timestamped_path_creates_unique_filename(tmp_path, monkeypatch):
    """
    Test that _timestamped_path generates a unique filename with a timestamp.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to isolate modifications to sys.path.

    Asserts
    -------
    - The returned file path contains the given filename and a timestamp prefix.
    """
    from src.data_loader import config  # type: ignore
    monkeypatch.setattr(config, "out_dir", tmp_path)
    path = _timestamped_path("data.csv")
    assert path.parent == config.out_dir
    assert path.name.endswith("data.csv")


# ============================================================================
# 2. Tests: Main Functions
# ============================================================================

def test_load_csv_success(tmp_path):
    """
    Test that load_csv successfully loads a CSV file into a DataFrame.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Asserts
    -------
    - The resulting object is a pandas DataFrame.
    - The DataFrame has the expected shape and content.
    """
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"col": [1, 2]}).to_csv(csv_file, index=False)
    df = load_csv(csv_file)
    assert not df.empty
    assert "col" in df.columns


def test_load_csv_raises_for_missing_file():
    """
    Test that load_csv raises FileNotFoundError for a non-existent CSV file.

    Asserts
    -------
    - FileNotFoundError is raised if the CSV file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        load_csv("missing.csv")


def test_load_excel_success(tmp_path):
    """
    Test that load_excel successfully loads an Excel file into a DataFrame.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Asserts
    -------
    - The resulting object is a pandas DataFrame.
    - The DataFrame has the expected shape and content.
    """
    excel_file = tmp_path / "data.xlsx"
    pd.DataFrame({"col": [1, 2]}).to_excel(excel_file, index=False)
    df = load_excel(excel_file)
    assert not df.empty
    assert "col" in df.columns


def test_load_json_success(tmp_path):
    """
    Test that load_json successfully loads a JSON file into a DataFrame.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Asserts
    -------
    - The resulting object is a pandas DataFrame.
    - The DataFrame has the expected shape and content.
    """
    json_file = tmp_path / "data.json"
    pd.DataFrame({"col": [1, 2]}).to_json(json_file, orient="records")
    df = load_json(json_file)
    assert not df.empty
    assert "col" in df.columns


def test_validate_url_accessible_success(requests_mock):
    """
    Test that _validate_url_accessible does not raise for an accessible URL.

    Parameters
    ----------
    requests_mock : requests_mock.Mocker
        Pytest fixture used to mock HTTP requests.

    Asserts
    -------
    - No exception is raised for a valid URL returning status code 200.
    """
    requests_mock.head("http://fake-url.com", status_code=200)
    _validate_url_accessible("http://fake-url.com")


def test_load_multiple_datasets_success(tmp_path):
    """
    Test that load_multiple_datasets successfully loads multiple datasets.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Asserts
    -------
    - The returned dictionary contains the expected dataset names.
    - Each value in the dictionary is a valid pandas DataFrame.
    """
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"col": [1]}).to_csv(csv_file, index=False)
    datasets = load_multiple_datasets({"dataset1": csv_file})
    assert "dataset1" in datasets
    assert not datasets["dataset1"].empty


def test_load_multiple_datasets_unsupported_extension(tmp_path):
    """
    Test that load_multiple_datasets raises UnsupportedFileTypeError for an invalid file extension.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Asserts
    -------
    - UnsupportedFileTypeError is raised for unsupported file formats.
    """
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("invalid")
    with pytest.raises(UnsupportedFileTypeError):
        load_multiple_datasets({"bad": bad_file})


def test_list_csv_files_returns_only_csv(tmp_path):
    """
    Test that list_csv_files returns only CSV files from a directory.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Asserts
    -------
    - The returned list contains only CSV files.
    - Non-CSV files are excluded.
    """
    csv_file = tmp_path / "file.csv"
    txt_file = tmp_path / "file.txt"
    csv_file.write_text("col\n1")
    txt_file.write_text("text")
    files = list_csv_files(tmp_path)
    assert csv_file in files
    assert txt_file not in files


def test_preview_df_returns_n_rows():
    """
    Test that preview_df returns the correct number of rows from a DataFrame.

    Asserts
    -------
    - The returned DataFrame has exactly `n` rows.
    """
    df = pd.DataFrame({"col": range(10)})
    result = preview_df(df, n=3)
    assert len(result) == 3


def test_preview_df_invalid_n_raises():
    """
    Test that preview_df raises ValueError for an invalid `n` parameter.

    Asserts
    -------
    - ValueError is raised when `n` is less than or equal to zero.
    """
    df = pd.DataFrame({"col": [1]})
    with pytest.raises(ValueError):
        preview_df(df, n=0)


def test_save_df_creates_file(tmp_path, monkeypatch):
    """
    Test that save_df saves a DataFrame in the specified format.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to isolate modifications to sys.path.

    Asserts
    -------
    - The output file is created successfully.
    - The file has the expected format and extension.
    """
    from src.data_loader import config  # type: ignore
    monkeypatch.setattr(config, "out_dir", tmp_path)

    df = pd.DataFrame({"col": [1]})
    path = save_df(df, "test", fmt="csv")
    assert path.exists()
    assert path.suffix == ".csv"


# ======================================================================

def test_fetch_reddit_comments_with_mock():
    """Test that fetch_reddit_comments works with a mock Reddit instance.

    This test ensures that the function can process comments correctly
    without making real API calls.
    
    Asserts
    -------
    - The returned DataFrame is not empty.
    - The DataFrame contains the expected columns.
    """

    mock_reddit = MagicMock()
    mock_submission = MagicMock()
    mock_comment = MagicMock()
    mock_comment.id = "abc123"
    mock_comment.body = "Test body"
    mock_comment.author = "tester"
    mock_comment.created_utc = 1234567890
    mock_comment.score = 10
    mock_comment.parent_id = "t1_parent"
    mock_comment.is_submitter = False
    mock_submission.comments.list.return_value = [mock_comment]
    mock_reddit.submission.return_value = mock_submission

    with patch("src.data_loader._validate_url_accessible") as mock_validate:
        mock_validate.return_value = None

        df = fetch_reddit_comments("http://fakeurl", mock_reddit)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert set(df.columns) == {
            "comment_id", "body", "author",
            "created_utc", "score", "parent_id", "is_submitter"
        }


# ================================================================
# Tests for loader_telco_data
# ================================================================
@pytest.fixture
def telco_csv(tmp_path):
    """
    Fixture that creates a minimal Telco CSV file for testing.

    Returns
    -------
    Path
        Path to the created CSV file.
    """
    df = pd.DataFrame({
        "customerID": ["0001", "0002"],
        "gender": ["Female", "Male"],
        "Churn": ["No", "Yes"]
    })
    path = tmp_path / "telco_customer_data.csv"
    df.to_csv(path, index=False)
    return path


def test_loader_telco_data_reads_existing_file(monkeypatch, telco_csv):
    """
    Test that loader_telco_data reads an existing Telco CSV file correctly.

    Asserts:
        - The returned DataFrame has the expected shape and columns.
    """
    from src import data_loader #type: ignore
    monkeypatch.setattr(data_loader.config, "out_dir", telco_csv.parent)
    df = loader_telco_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert set(df.columns) == {"customerID", "gender", "Churn"}


def test_loader_telco_data_raises_on_empty(monkeypatch, tmp_path):
    """
    Test that loader_telco_data raises DataLoaderError if the CSV is empty.

    Asserts:
        - DataLoaderError is raised.
    """
    empty_path = tmp_path / "telco_customer_data.csv"
    empty_path.write_text("")
    from src import data_loader #type: ignore
    monkeypatch.setattr(data_loader.config, "out_dir", tmp_path)
    with pytest.raises(DataLoaderError, match="No columns to parse from file"):
        loader_telco_data()


def test_loader_telco_data_raises_on_missing(monkeypatch, tmp_path):
    """
    Test that loader_telco_data raises DataLoaderError if the file is missing and cannot be downloaded.

    Asserts:
        - DataLoaderError is raised.
    """
    from src import data_loader #type: ignore
    monkeypatch.setattr(data_loader.config, "out_dir", tmp_path)
    def raise_dl_error(*args, **kwargs):
        raise Exception("Download failed")
    monkeypatch.setattr(data_loader, "_download_telco_dataset", raise_dl_error)
    with pytest.raises(Exception, match="Download failed"):
        loader_telco_data()


# ================================================================
# Tests for loader_reddit_comments
# ================================================================
@pytest.fixture
def reddit_csv(tmp_path):
    """
    Fixture that creates a minimal Reddit comments CSV file for testing.

    Returns
    -------
    Path
        Path to the created CSV file.
    """
    df = pd.DataFrame({
        "comment_id": ["abc123", "def456"],
        "body": ["Great service!", "Terrible experience."],
        "author": ["user1", "user2"],
        "created_utc": [1693700000, 1693700100],
        "score": [10, -2],
        "parent_id": ["t3_xyz", "t3_xyz"],
        "is_submitter": [False, True]
    })
    path = tmp_path / "public_comments.csv"
    df.to_csv(path, index=False)
    return path


def test_loader_reddit_comments_reads_existing_file(monkeypatch, reddit_csv):
    """
    Test that loader_reddit_comments reads an existing Reddit comments CSV file correctly.

    Asserts:
        - The returned DataFrame has the expected shape and columns.
    """
    from src import data_loader #type: ignore
    monkeypatch.setattr(data_loader.config, "out_dir", reddit_csv.parent)
    df = loader_reddit_comments()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 7)
    assert set(df.columns) == {
        "comment_id", "body", "author", "created_utc", "score", "parent_id", "is_submitter"
    }


def test_loader_reddit_comments_raises_on_empty(monkeypatch, tmp_path):
    """
    Test that loader_reddit_comments raises DataLoaderError if the CSV is empty.

    Asserts:
        - DataLoaderError is raised.
    """
    empty_path = tmp_path / "public_comments.csv"
    empty_path.write_text("")
    from src import data_loader #type: ignore
    monkeypatch.setattr(data_loader.config, "out_dir", tmp_path)
    with pytest.raises(DataLoaderError, match="No columns to parse from file"):
        loader_reddit_comments()


def test_loader_reddit_comments_raises_on_missing(monkeypatch, tmp_path):
    """
    Test that loader_reddit_comments raises DataLoaderError if the file is missing.

    Asserts:
        - DataLoaderError is raised.
    """
    from src import data_loader #type: ignore
    monkeypatch.setattr(data_loader.config, "out_dir", tmp_path)
    with pytest.raises(DataLoaderError, match="Failed to load Reddit comments dataset"):
        loader_reddit_comments()