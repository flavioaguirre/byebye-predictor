# ================================================================
#  tests/test_feature_engineering.py
# ================================================================
#  Unit tests for the feature engineering helpers used in the Telco
#  feature engineering notebook.
#
#  Scope (current project state):
#   - FeatureSelectorTransformer: validates that a wrapped selector
#     integrates correctly with pandas DataFrames and preserves row
#     consistency.
#
#  Notes:
#   - This file has been simplified to reflect the components that
#     actually exist and are used in src/feature_engineering.py
#     for the Telco use case.
#
#  Author: Flavio Aguirre
#  Date: 2025-11-28
# ================================================================

import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold

from src.feature_engineering import FeatureSelectorTransformer  # type: ignore


@pytest.fixture
def sample_df():
    """
    Provides a simple DataFrame to test feature selection behavior.
    """
    return pd.DataFrame(
        {
            "tenure": [1, 5, 10, 20, 30],
            "issenior": [0, 1, 0, 1, 0],
            "constant_col": [1, 1, 1, 1, 1],
        }
    )


def test_feature_selector_transformer_reduces_or_keeps_columns(sample_df):
    """
    Test that FeatureSelectorTransformer, when wrapping a variance-based selector,
    keeps the same number of rows and reduces or keeps the number of columns.

    Asserts:
        - The number of rows is preserved.
        - The number of columns after selection is less than or equal to the original.
    """
    selector = VarianceThreshold(threshold=0.0)
    wrapper = FeatureSelectorTransformer(feature_selector=selector)

    X_selected = wrapper.fit_transform(sample_df, sample_df["issenior"])

    assert isinstance(X_selected, pd.DataFrame)
    assert X_selected.shape[0] == sample_df.shape[0]
    assert X_selected.shape[1] <= sample_df.shape[1]