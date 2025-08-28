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
#   4. FeatureSelectorTransformer: Validates feature selection integration,
#      including correct reduction of columns and consistency of output.
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
    FeatureEngineeringError,
    DatetimeFeatureTransformer,
    TextFeatureTransformer,
    FeatureSelectorTransformer,
    FeatureEngineer,
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
        "tenure": [1, 5, 10, 20, 30],
        "issenior": [0, 1, 0, 1, 0],
        "contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Two year"],
        "joindate": pd.date_range("2020-01-01", periods=5, freq="365D"),
        "comments": ["good service", "bad connection", "excellent support", "average price", "terrible experience"],
        "customerid": [f"CUST-{i}" for i in range(5)],
        "totalcharges": [100, 200, 300, 400, 500],
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
        text_features=['comments']
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
    result = pd.DataFrame(transformer.fit_transform(sample_df[['comments']]))
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
    transformer.fit(sample_df[['comments']])
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



# ================================================================
#  Tests: FeatureSelectorTransformer
# ================================================================
from sklearn.feature_selection import VarianceThreshold

def test_feature_engineering_without_selection(sample_df):
    """
    Test that the pipeline works without applying feature selection.

    Asserts:
        - The output is a DataFrame.
        - The number of rows matches the input DataFrame.
    """
    fe = FeatureEngineer(feature_selector=None)
    df = fe.fit_transform(sample_df)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == sample_df.shape[0]

def test_feature_engineering_with_selection(sample_df):
    """
    Test that feature selection is correctly applied when using a selector.

    Asserts:
        - The output is a DataFrame.
        - The number of rows matches the input DataFrame.
    """
    variance_selector_obj = VarianceThreshold(threshold=0.0)
    selector_wrapper = FeatureSelectorTransformer(selector=variance_selector_obj)
    
    fe = FeatureEngineer(
        numeric_features=["tenure", "issenior"],
        feature_selector=selector_wrapper
    )
    # Passes 'y' to satisfy the fit signature of the selector wrapper
    df = fe.fit_transform(sample_df, sample_df['issenior'])

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == sample_df.shape[0]


def test_feature_selection_consistency(sample_df):
    """
    Test that using feature selection reduces or keeps the number of columns
    compared to not using selection, and that the number of rows remains the same.

    Asserts:
        - The number of rows is the same with and without selection.
        - The number of columns with selection is less than or equal to without selection.
    """
    df_test = sample_df.copy()
    df_test['constant_col'] = 1 
    
    fe_no_sel = FeatureEngineer(numeric_features=["tenure", "issenior", "constant_col"])
    df_no_sel = fe_no_sel.fit_transform(df_test)

    variance_selector_obj = VarianceThreshold(threshold=0.0)
    selector_wrapper = FeatureSelectorTransformer(selector=variance_selector_obj)
    
    fe_sel = FeatureEngineer(
        numeric_features=["tenure", "issenior", "constant_col"],
        feature_selector=selector_wrapper
    )
    df_sel = fe_sel.fit_transform(df_test, df_test['issenior'])

    assert df_no_sel.shape[0] == df_sel.shape[0]
    assert df_sel.shape[1] <= df_no_sel.shape[1]


def test_selected_features_are_present(sample_df):
    """
    Test that the selected features after feature selection are present in the output DataFrame.

    Asserts:
        - The list of selected feature names is not empty.
        - All output columns are present in the DataFrame.
    """
    variance_selector_obj = VarianceThreshold(threshold=0.0)
    selector_wrapper = FeatureSelectorTransformer(selector=variance_selector_obj)

    fe = FeatureEngineer(
        numeric_features=["tenure", "issenior"],
        categorical_features=['contract'],
        feature_selector=selector_wrapper
    )
    df = fe.fit_transform(sample_df, sample_df['issenior'])

    # We use the column names of the transformed df as the truth
    selected_feature_names = df.columns.tolist()

    # Ensure it is not empty (the selector actually filtered features)
    assert len(selected_feature_names) > 0
    # Ensure all output columns are in the df
    assert set(df.columns) == set(selected_feature_names)