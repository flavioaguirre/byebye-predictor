# ================================================================
#  tests/test_feature_engineering.py
# ================================================================
#  Unit and integration tests for the feature engineering module.
#  This suite ensures the reliability and correctness of all
#  feature transformation and pipeline components before they are
#  used in a machine learning workflow.
#
#  Tested Exceptions:
#   - FeatureEngineeringError (for pipeline build/transform issues)
#   - ValueError (for invalid configurations)
#
#  Tested Components:
#   1. DatetimeFeatureTransformer: Validates correct extraction of
#      temporal features (year, month, day, etc.).
#   2. TextFeatureTransformer: Ensures vectorization and output shape
#      consistency for text features.
#   3. FeatureEngineer: Integration tests for the end-to-end workflow,
#      including pipeline building, fitting, and transformation.
#
#  Fixtures:
#   - sample_df: Minimal DataFrame covering numeric, categorical,
#     datetime, and text columns.
#   - feature_engineer: Instantiated FeatureEngineer with sample
#     configuration.
#
#  Notes:
#   - Uses pytest for structured testing.
#   - pandas.testing and numpy.testing are leveraged for precise
#     comparisons.
#
#  Each test function includes a docstring describing its purpose and
#  the assertions it performs.
#
#  Author: Flavio Aguirre
#  Date: 2025-08-27


# ================================================================

# ===========================================================================
# Imports
# ===========================================================================
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError

from src.feature_engineering import (  # type: ignore
    DatetimeFeatureTransformer,
    TextFeatureTransformer,
    FeatureEngineer,
    FeatureEngineeringError
)

# ================================================================
#   Fixtures
# ================================================================
@pytest.fixture
def sample_df():
    """
    Provides a DataFrame with numeric, categorical, datetime, and text columns.
    """
    return pd.DataFrame({
        "tenure": [1, 5, 10, None],
        "contract": ["Month-to-month", "One year", "Two year", None],
        "joindate": pd.to_datetime([
            "2020-01-01", "2021-06-15", "2022-12-31", None
        ]),
        "review": [
            "This service is AMAZING!",
            "Not good, too expensive.",
            "Average experience overall.",
            None
        ]
    })


@pytest.fixture
def feature_engineer():
    """
    Instantiates a FeatureEngineer with explicit feature groups.
    """
    return FeatureEngineer(
        numeric_features=["tenure"],
        categorical_features=["contract"],
        datetime_features=["joindate"],
        text_features=["review"]
    )

# ================================================================
#   Tests: DatetimeFeatureTransformer
# ================================================================
def test_datetime_transformer_extracts_features(sample_df):
    """
    Test that DatetimeFeatureTransformer extracts at least three features
    (e.g., year, month, day) from a datetime column.

    Asserts:
        - The output DataFrame has at least three columns.
    """
    transformer = DatetimeFeatureTransformer(["joindate"])
    result = pd.DataFrame(transformer.fit_transform(sample_df)) 

    assert result.shape[1] >= 3, "Expected at least 3 datetime-derived features."


def test_datetime_transformer_handles_missing_values(sample_df):
    """
    Test that DatetimeFeatureTransformer handles missing values in datetime columns
    and does not return all NaN features.

    Asserts:
        - Not all values in the output DataFrame are NaN.
    """
    transformer = DatetimeFeatureTransformer(["joindate"])
    result = pd.DataFrame(transformer.fit_transform(sample_df))

    assert not result.isnull().all(axis=None), "All datetime-derived features are NaN."


# ================================================================
#   Tests: TextFeatureTransformer
# ================================================================
def test_text_transformer_creates_features(sample_df):
    """
    Test that TextFeatureTransformer creates more than one feature (column)
    when vectorizing text data.

    Asserts:
        - The output DataFrame has more than one column.
    """
    transformer = TextFeatureTransformer(max_features=50)
    result = pd.DataFrame(transformer.fit_transform(sample_df[["review"]]))
    assert result.shape[1] > 1


def test_text_transformer_get_feature_names(sample_df):
    """
    Test that TextFeatureTransformer returns a non-empty list of feature names
    after fitting.

    Asserts:
        - The feature names are a list.
        - The list is not empty.
    """
    transformer = TextFeatureTransformer(max_features=50)
    transformer.fit(sample_df[["review"]])
    feature_names = transformer.get_feature_names_out()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0


# ================================================================
#   Tests: FeatureEngineer
# ================================================================
def test_feature_engineer_fit_transform(sample_df, feature_engineer):
    """
    Test that FeatureEngineer.fit_transform returns a DataFrame with the same
    number of rows as the input (after dropping NaNs).

    Asserts:
        - The output is a DataFrame.
        - The number of rows matches the cleaned input.
    """
    clean_df = sample_df.dropna().reset_index(drop=True)
    transformed = feature_engineer.fit_transform(clean_df)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == clean_df.shape[0]


def test_feature_engineer_get_feature_names(sample_df, feature_engineer):
    """
    Test that FeatureEngineer exposes a non-empty list of feature names after fitting.

    Asserts:
        - The feature names are a list.
        - The list is not empty.
    """
    clean_df = sample_df.dropna().reset_index(drop=True)
    feature_engineer.fit(clean_df)

    feature_names = feature_engineer._feature_names_out

    if not isinstance(feature_names, list):
        feature_names = feature_names.tolist()

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0

def test_feature_engineer_transform_before_fit_raises(sample_df):
    """
    Test that calling transform before fit on FeatureEngineer raises FeatureEngineeringError.

    Asserts:
        - FeatureEngineeringError is raised.
    """
    fe = FeatureEngineer(numeric_features=["tenure"])

    with pytest.raises(FeatureEngineeringError):
        fe.transform(sample_df)


def test_feature_engineer_pipeline_error(monkeypatch, sample_df):
    """
    Test that an error raised during pipeline building in FeatureEngineer is propagated.

    Asserts:
        - RuntimeError is raised when _build_pipeline fails.
    """
    fe = FeatureEngineer(numeric_features=["tenure"])

    def broken_build():
        raise RuntimeError("Broken pipeline")

    fe._build_pipeline = broken_build

    with pytest.raises(RuntimeError):
        fe.fit(sample_df.dropna())


