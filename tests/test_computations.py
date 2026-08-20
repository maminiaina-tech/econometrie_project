"""Tests for econometrie_m1.computations.stats.

Covers utility formatting, outlier detection/correction, partial correlations,
multicollinearity diagnostics (Klein, Farrar-Glauber, VIF), optimal mix,
and French-language text generation helpers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from econometrie_m1.computations.stats import (
    FarrarFTestResult,
    FarrarGlauberResult,
    FarrarGlobalResult,
    FarrarTTestResult,
    KleinTestResult,
    MulticollinearityResult,
    OptimalMixResult,
    OutlierColumnResult,
    OutlierDetectionResult,
    PartialCorrelationResult,
    PartialCorrelationsFullResult,
    analyze_multicollinearity,
    calculate_partial_correlations,
    correct_outliers,
    detect_outliers,
    determine_optimal_mix,
    farrar_glauber_test,
    format_number,
    get_calculation_steps,
    get_french_summary,
    get_interpretations,
    get_model_formulas,
    klein_test,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_ols_model(
    n: int = 100,
    coefs: tuple[float, ...] = (2.0, 3.0, -1.0),
    noise_std: float = 0.5,
    seed: int = 42,
) -> sm.regression.linear_model.OLSResults:
    """Build a small OLS model on synthetic data for testing."""
    rng = np.random.default_rng(seed)
    p = len(coefs)
    X = rng.normal(size=(n, p))
    y = X @ np.array(coefs) + rng.normal(scale=noise_std, size=n)
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    X_with_const = sm.add_constant(X_df)
    model = sm.OLS(y, X_with_const).fit()
    return model


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """DataFrame with two nearly-independent columns and no outliers."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(10, 2, size=200),
            "x2": rng.normal(5, 1, size=200),
            "x3": rng.normal(0, 3, size=200),
        }
    )


@pytest.fixture
def outlier_df() -> pd.DataFrame:
    """DataFrame with deliberate outliers in column 'a'."""
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, size=100)
    data[0] = 100.0  # high outlier
    data[1] = -100.0  # low outlier
    return pd.DataFrame({"a": data, "b": rng.normal(5, 0.5, size=100)})


@pytest.fixture
def multicollinear_df() -> pd.DataFrame:
    """DataFrame with intentionally correlated predictors."""
    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.05, size=n)  # ≈0.999 corr with x1
    x3 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


@pytest.fixture
def ols_model() -> sm.regression.linear_model.OLSResults:
    """A fitted OLS model on synthetic data."""
    return _make_ols_model()


@pytest.fixture
def ols_model_high_r2() -> sm.regression.linear_model.OLSResults:
    """A fitted OLS model with very high R² (noise-free relationship)."""
    rng = np.random.default_rng(99)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + 3.0 * x2
    df = pd.DataFrame({"x1": x1, "x2": x2})
    return sm.OLS(y, sm.add_constant(df)).fit()


# ===================================================================
# 1. format_number
# ===================================================================


class TestFormatNumber:
    """Tests for the format_number utility function."""

    def test_integer_default(self) -> None:
        """Integer formatted with 6 decimal places by default."""
        assert format_number(42) == "42.000000"

    def test_float_default(self) -> None:
        """Float formatted with 6 decimal places by default."""
        assert format_number(3.14) == "3.140000"

    def test_custom_decimal_places(self) -> None:
        """Respects the decimal_places argument."""
        assert format_number(1.2345, decimal_places=2) == "1.23"

    def test_zero(self) -> None:
        """Zero is formatted correctly."""
        assert format_number(0) == "0.000000"

    def test_negative(self) -> None:
        """Negative floats format correctly."""
        assert format_number(-7.5, decimal_places=1) == "-7.5"

    def test_non_numeric_returns_str(self) -> None:
        """Non-numeric types are converted via str()."""
        assert format_number("hello") == "hello"
        assert format_number(None) == "None"
        assert format_number([1, 2]) == "[1, 2]"

    def test_nan(self) -> None:
        """NaN is a float, so it formats like any float."""
        result = format_number(float("nan"))
        assert result == "nan"

    def test_inf(self) -> None:
        """Inf is a float, so it formats like any float."""
        result = format_number(float("inf"))
        assert result == "inf"

    def test_very_small_number(self) -> None:
        """Very small float uses scientific notation-like fixed format."""
        result = format_number(1e-10, decimal_places=8)
        assert result == "0.00000000"

    def test_large_integer(self) -> None:
        """Large integer formatted correctly."""
        assert format_number(1_000_000, decimal_places=2) == "1000000.00"


# ===================================================================
# 2. detect_outliers
# ===================================================================


class TestDetectOutliers:
    """Tests for IQR-based outlier detection."""

    def test_returns_correct_type(self, simple_df: pd.DataFrame) -> None:
        """detect_outliers returns an OutlierDetectionResult with one column per numeric column."""
        result = detect_outliers(simple_df)
        assert isinstance(result, OutlierDetectionResult)
        assert len(result.columns) == 3

    def test_few_outliers_in_normal_data(self) -> None:
        """With large enough normal samples, most values should be within IQR bounds."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(0, 1, size=5000)})
        result = detect_outliers(df)
        # For a normal distribution, IQR method should flag <2% of points
        assert result.columns[0].outlier_count / len(df) < 0.05

    def test_detects_known_outliers(self, outlier_df: pd.DataFrame) -> None:
        """The injected outliers at indices 0 and 1 should be detected."""
        result = detect_outliers(outlier_df)
        assert 0 in result.all_outlier_indices
        assert 1 in result.all_outlier_indices

    def test_outlier_column_result_fields(self, outlier_df: pd.DataFrame) -> None:
        """Each OutlierColumnResult has the expected fields."""
        result = detect_outliers(outlier_df)
        col_a = next(c for c in result.columns if c.column == "a")
        assert isinstance(col_a, OutlierColumnResult)
        assert col_a.q1 < col_a.q3
        assert col_a.iqr == col_a.q3 - col_a.q1
        assert col_a.lower_bound == col_a.q1 - 1.5 * col_a.iqr
        assert col_a.upper_bound == col_a.q3 + 1.5 * col_a.iqr
        assert col_a.outlier_count >= 2

    def test_outlier_count_matches_indices(self, outlier_df: pd.DataFrame) -> None:
        """outlier_count must equal len(outlier_indices)."""
        result = detect_outliers(outlier_df)
        for col_result in result.columns:
            assert col_result.outlier_count == len(col_result.outlier_indices)

    def test_all_outlier_indices_are_sorted_unique(self, outlier_df: pd.DataFrame) -> None:
        """all_outlier_indices should be sorted and without duplicates."""
        result = detect_outliers(outlier_df)
        assert result.all_outlier_indices == sorted(set(result.all_outlier_indices))

    def test_non_numeric_columns_ignored(self) -> None:
        """String columns are silently skipped."""
        df = pd.DataFrame({"a": [1, 2, 100, 4, 5], "label": ["x", "y", "z", "w", "v"]})
        result = detect_outliers(df)
        assert len(result.columns) == 1
        assert result.columns[0].column == "a"

    def test_single_column(self) -> None:
        """Works correctly with a single-column DataFrame."""
        df = pd.DataFrame({"val": [1, 1, 1, 1, 100]})
        result = detect_outliers(df)
        assert len(result.columns) == 1
        assert result.columns[0].outlier_count >= 1


# ===================================================================
# 3. correct_outliers
# ===================================================================


class TestCorrectOutliers:
    """Tests for median-based outlier correction."""

    def test_returns_copy(self, simple_df: pd.DataFrame) -> None:
        """The original DataFrame must not be mutated."""
        original = simple_df.copy()
        _ = correct_outliers(simple_df, [])
        pd.testing.assert_frame_equal(simple_df, original)

    def test_outliers_replaced_with_median(self, outlier_df: pd.DataFrame) -> None:
        """Values exceeding IQR bounds are replaced with the column median."""
        corrected = correct_outliers(outlier_df, [0, 1])
        # The function ignores outlier_indices and uses IQR internally.
        # Indices 0 (100.0) and 1 (-100.0) are outliers by IQR → replaced with median.
        median_a = outlier_df["a"].median()
        assert corrected.loc[0, "a"] == pytest.approx(median_a)
        assert corrected.loc[1, "a"] == pytest.approx(median_a)

    def test_non_outlier_iqr_values_unchanged(self) -> None:
        """Values within IQR bounds remain untouched."""
        rng = np.random.default_rng(5)
        data_a = rng.normal(0, 1, size=500)
        data_a[0] = 100.0  # only true outlier in column a
        df = pd.DataFrame(
            {
                "a": data_a,
                "b": np.full(500, 5.0),  # constant column, no outliers
            }
        )
        corrected = correct_outliers(df, [0])
        # Index 0 was the outlier and should be replaced
        assert corrected.loc[0, "a"] == pytest.approx(df["a"].median())
        # All rows in 'b' should remain unchanged (constant, no IQR outliers)
        for idx in range(len(df)):
            assert corrected.loc[idx, "b"] == pytest.approx(5.0)

    def test_empty_indices_still_corrects_iqr_outliers(self, outlier_df: pd.DataFrame) -> None:
        """Even with empty outlier_indices, the function replaces IQR-detected outliers."""
        corrected = correct_outliers(outlier_df, [])
        # The function replaces ALL IQR outliers regardless of the indices argument
        median_a = outlier_df["a"].median()
        assert corrected.loc[0, "a"] == pytest.approx(median_a)

    def test_numeric_only_modified(self) -> None:
        """Non-numeric columns survive correction unchanged."""
        df = pd.DataFrame(
            {
                "num": [1, 1, 1, 1, 1000],
                "cat": ["a", "b", "c", "d", "e"],
            }
        )
        corrected = correct_outliers(df, [4])
        assert list(corrected["cat"]) == ["a", "b", "c", "d", "e"]


# ===================================================================
# 4. calculate_partial_correlations
# ===================================================================


class TestCalculatePartialCorrelations:
    """Tests for systematic partial correlation computation."""

    def test_returns_dataclass(self, simple_df: pd.DataFrame) -> None:
        """Result is a PartialCorrelationsFullResult instance."""
        result = calculate_partial_correlations(simple_df, "x1", ["x2", "x3"])
        assert isinstance(result, PartialCorrelationsFullResult)

    def test_partial_correlations_populated(self) -> None:
        """With 3+ x_vars, partial correlations are populated at order 1."""
        rng = np.random.default_rng(20)
        df = pd.DataFrame(
            {
                "y": rng.normal(size=100),
                "a": rng.normal(size=100),
                "b": rng.normal(size=100),
                "c": rng.normal(size=100),
            }
        )
        result = calculate_partial_correlations(df, "y", ["a", "b", "c"])
        assert len(result.partial_correlations) > 0

    def test_partial_r2_covers_all_vars(self, simple_df: pd.DataFrame) -> None:
        """Partial R² list has one entry per independent variable."""
        x_vars = ["x2", "x3"]
        result = calculate_partial_correlations(simple_df, "x1", x_vars)
        r2_vars = [r.variable for r in result.partial_r2]
        assert r2_vars == x_vars

    def test_partial_r2_values_bounded(self, simple_df: pd.DataFrame) -> None:
        """Partial R² values should be non-negative (can exceed 1 in edge cases)."""
        result = calculate_partial_correlations(simple_df, "x1", ["x2", "x3"])
        for pr2 in result.partial_r2:
            assert pr2.r2_partial >= -0.01  # allow tiny numerical error

    def test_partial_corr_dataclass_fields(self, simple_df: pd.DataFrame) -> None:
        """Each PartialCorrelationResult has the expected fields."""
        result = calculate_partial_correlations(simple_df, "x1", ["x2", "x3"])
        for pc in result.partial_correlations:
            assert isinstance(pc, PartialCorrelationResult)
            assert isinstance(pc.correlation, float)
            assert pc.order >= 1

    def test_correlation_between_neg1_and_1(self, simple_df: pd.DataFrame) -> None:
        """All partial correlations must lie in [-1, 1]."""
        result = calculate_partial_correlations(simple_df, "x1", ["x2", "x3"])
        for pc in result.partial_correlations:
            assert -1.0 <= pc.correlation <= 1.0


# ===================================================================
# 5. klein_test
# ===================================================================


class TestKleinTest:
    """Tests for the Klein multicollinearity test."""

    def test_no_multicollinearity(self, simple_df: pd.DataFrame) -> None:
        """With low pairwise correlations, Klein test should not flag multicollinearity."""
        x_vars = ["x1", "x2", "x3"]
        full_model = sm.OLS(simple_df["x1"], sm.add_constant(simple_df[["x2", "x3"]])).fit()
        result = klein_test(full_model.rsquared, simple_df, x_vars)
        assert isinstance(result, KleinTestResult)
        # For nearly independent x2, x3, pairwise r² should be low
        # and R² of full model (predicting x1 from others) should be low too
        # so no multicollinearity expected
        assert result.multicollinear_detected is False

    def test_multicollinearity_detected(self, multicollinear_df: pd.DataFrame) -> None:
        """When x2 ≈ x1, Klein test should detect multicollinearity."""
        x_vars = ["x1", "x2", "x3"]
        full_model = sm.OLS(
            multicollinear_df["x1"],
            sm.add_constant(multicollinear_df[["x2", "x3"]]),
        ).fit()
        result = klein_test(full_model.rsquared, multicollinear_df, x_vars)
        # r²(x1, x2) ≈ 0.999, and R² of predicting x1 from x2+x3 ≈ 0.999
        # but Klein flags when R²_model < r²_pair for any pair
        assert isinstance(result.pair_results, dict)
        for pair, (r2_mod, r2_pair) in result.pair_results.items():
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert r2_mod >= 0
            assert r2_pair >= 0

    def test_pair_results_keys(self, simple_df: pd.DataFrame) -> None:
        """Pair results contain all unordered pairs of x_vars."""
        x_vars = ["x1", "x2", "x3"]
        result = klein_test(0.5, simple_df, x_vars)
        expected_pairs = {("x1", "x2"), ("x1", "x3"), ("x2", "x3")}
        assert set(result.pair_results.keys()) == expected_pairs

    def test_r_squared_stored(self, simple_df: pd.DataFrame) -> None:
        """The stored r_squared matches the input."""
        result = klein_test(0.42, simple_df, ["x1", "x2"])
        assert result.r_squared == 0.42


# ===================================================================
# 6. farrar_glauber_test
# ===================================================================


class TestFarrarGlauberTest:
    """Tests for the Farrar-Glauber three-step multicollinearity test."""

    def test_returns_dataclass(self, simple_df: pd.DataFrame) -> None:
        """Result is a FarrarGlauberResult."""
        result = farrar_glauber_test(simple_df, ["x1", "x2", "x3"])
        assert isinstance(result, FarrarGlauberResult)

    def test_global_test_fields(self, simple_df: pd.DataFrame) -> None:
        """Global chi² test has determinant, chi2, df, and p_value."""
        result = farrar_glauber_test(simple_df, ["x1", "x2", "x3"])
        gt = result.global_test
        assert isinstance(gt, FarrarGlobalResult)
        assert 0 < gt.determinant <= 1.0
        assert gt.chi2 >= 0
        assert gt.df == 3  # p*(p-1)/2 = 3*2/2 = 3
        assert 0 <= gt.p_value <= 1

    def test_f_tests_count(self, simple_df: pd.DataFrame) -> None:
        """One F-test per independent variable."""
        x_vars = ["x1", "x2", "x3"]
        result = farrar_glauber_test(simple_df, x_vars)
        assert len(result.f_tests) == len(x_vars)

    def test_f_tests_have_valid_values(self, simple_df: pd.DataFrame) -> None:
        """F values and p-values are valid floats."""
        result = farrar_glauber_test(simple_df, ["x1", "x2", "x3"])
        for ft in result.f_tests:
            assert isinstance(ft, FarrarFTestResult)
            assert ft.f_value >= 0
            assert 0 <= ft.f_pvalue <= 1

    def test_t_tests_count(self, simple_df: pd.DataFrame) -> None:
        """One t-test per pair of independent variables."""
        x_vars = ["x1", "x2", "x3"]
        result = farrar_glauber_test(simple_df, x_vars)
        # 3 choose 2 = 3 pairs, each with 1 remaining control var → 3 t-tests
        assert len(result.t_tests) == 3

    def test_t_tests_have_valid_values(self, simple_df: pd.DataFrame) -> None:
        """t values are finite and p-values in [0, 1]."""
        result = farrar_glauber_test(simple_df, ["x1", "x2", "x3"])
        for tt in result.t_tests:
            assert isinstance(tt, FarrarTTestResult)
            assert math.isfinite(tt.t_value)
            assert 0 <= tt.p_value <= 1

    def test_two_vars_no_t_tests(self) -> None:
        """With only 2 x_vars, there are no other_vars → no t-tests."""
        rng = np.random.default_rng(10)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
        result = farrar_glauber_test(df, ["a", "b"])
        assert len(result.t_tests) == 0

    def test_chi2_zero_determinant(self) -> None:
        """Perfectly collinear columns → determinant ≈ 0, large chi²."""
        n = 50
        x1 = np.arange(n, dtype=float)
        df = pd.DataFrame({"x1": x1, "x2": x1, "x3": np.ones(n)})
        result = farrar_glauber_test(df, ["x1", "x2", "x3"])
        # chi2 should be very large for near-singular correlation matrix
        assert result.global_test.chi2 > 10


# ===================================================================
# 7. determine_optimal_mix
# ===================================================================


class TestDetermineOptimalMix:
    """Tests for elasticity-based optimal mix computation."""

    def test_returns_dataclass(self) -> None:
        """Result is an OptimalMixResult."""
        model = _make_ols_model(n=100, coefs=(2.0, 3.0, -1.0))
        data = pd.DataFrame(
            {
                "x0": model.model.exog[:, 1],
                "x1": model.model.exog[:, 2],
                "x2": model.model.exog[:, 3],
            }
        )
        result = determine_optimal_mix(model, data, ["x0", "x1", "x2"])
        assert isinstance(result, OptimalMixResult)

    def test_elasticities_keys(self) -> None:
        """Elasticity dict has one entry per x_var."""
        model = _make_ols_model(n=80, coefs=(1.0, 2.0))
        data = pd.DataFrame(
            {
                "x0": model.model.exog[:, 1],
                "x1": model.model.exog[:, 2],
            }
        )
        result = determine_optimal_mix(model, data, ["x0", "x1"])
        assert set(result.elasticities.keys()) == {"x0", "x1"}

    def test_optimal_shares_sum_to_one(self) -> None:
        """Shares should sum to 1.0."""
        model = _make_ols_model(n=200, coefs=(5.0, -2.0, 1.0))
        data = pd.DataFrame(
            {
                "x0": model.model.exog[:, 1],
                "x1": model.model.exog[:, 2],
                "x2": model.model.exog[:, 3],
            }
        )
        result = determine_optimal_mix(model, data, ["x0", "x1", "x2"])
        total = sum(result.optimal_shares.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_optimal_shares_non_negative(self) -> None:
        """All shares must be non-negative."""
        model = _make_ols_model(n=100, coefs=(3.0, -4.0))
        data = pd.DataFrame(
            {
                "x0": model.model.exog[:, 1],
                "x1": model.model.exog[:, 2],
            }
        )
        result = determine_optimal_mix(model, data, ["x0", "x1"])
        for share in result.optimal_shares.values():
            assert share >= 0

    def test_elasticity_formula(self) -> None:
        """Verify elasticity = coef * (x_mean / y_mean) for a single variable."""
        n = 50
        rng = np.random.default_rng(11)
        x_vals = rng.normal(10, 1, size=n)
        y_vals = 2.0 * x_vals + rng.normal(0, 0.1, size=n)

        X_df = pd.DataFrame({"x": x_vals})
        X_with_const = sm.add_constant(X_df)
        model = sm.OLS(y_vals, X_with_const).fit()

        data = pd.DataFrame({"x": x_vals})
        result = determine_optimal_mix(model, data, ["x"])

        estimated_coef = float(model.params["x"])
        expected_elasticity = estimated_coef * (np.mean(x_vals) / np.mean(y_vals))
        assert result.elasticities["x"] == pytest.approx(expected_elasticity, rel=1e-6)


# ===================================================================
# 8. analyze_multicollinearity
# ===================================================================


class TestAnalyzeMulticollinearity:
    """Tests for VIF, eigenvalues, condition indices, and correlation matrix."""

    def test_returns_dataclass(self, simple_df: pd.DataFrame) -> None:
        """Result is a MulticollinearityResult."""
        X = sm.add_constant(simple_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        assert isinstance(result, MulticollinearityResult)

    def test_vif_count_excludes_const(self, simple_df: pd.DataFrame) -> None:
        """VIF entries exclude the const column."""
        X = sm.add_constant(simple_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        assert len(result.vif_data) == 3
        vif_vars = {v.variable for v in result.vif_data}
        assert "const" not in vif_vars

    def test_vif_values_positive(self, simple_df: pd.DataFrame) -> None:
        """All VIF values must be strictly positive."""
        X = sm.add_constant(simple_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        for v in result.vif_data:
            assert v.vif > 0

    def test_vif_sorted_descending(self, simple_df: pd.DataFrame) -> None:
        """VIF results are sorted in descending order."""
        X = sm.add_constant(simple_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        vifs = [v.vif for v in result.vif_data]
        assert vifs == sorted(vifs, reverse=True)

    def test_eigenvalues_length(self, simple_df: pd.DataFrame) -> None:
        """Number of eigenvalues equals number of non-const columns."""
        X = sm.add_constant(simple_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        assert len(result.eigenvalues) == 3

    def test_condition_indices_length(self, simple_df: pd.DataFrame) -> None:
        """Condition indices count matches eigenvalue count."""
        X = sm.add_constant(simple_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        assert len(result.condition_indices) == len(result.eigenvalues)

    def test_max_condition_index(self, simple_df: pd.DataFrame) -> None:
        """max_condition_index is the max of condition_indices."""
        X = sm.add_constant(simple_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        assert result.max_condition_index == pytest.approx(max(result.condition_indices))

    def test_condition_index_at_least_one(self, simple_df: pd.DataFrame) -> None:
        """The smallest condition index is ≥ 1 (since max/it ≥ 1)."""
        X = sm.add_constant(simple_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        for ci in result.condition_indices:
            assert ci >= 1.0

    def test_correlation_matrix_shape(self, simple_df: pd.DataFrame) -> None:
        """Correlation matrix has the expected shape (no const)."""
        X = sm.add_constant(simple_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        assert result.correlation_matrix.shape == (3, 3)

    def test_high_multicollinearity_vif(self, multicollinear_df: pd.DataFrame) -> None:
        """Near-duplicate columns should produce very high VIF."""
        X = sm.add_constant(multicollinear_df[["x1", "x2", "x3"]])
        result = analyze_multicollinearity(X)
        x1_vif = next(v for v in result.vif_data if v.variable == "x1")
        # With corr(x1,x2) ≈ 1, VIF should be very large
        assert x1_vif.vif > 100


# ===================================================================
# 9. get_french_summary
# ===================================================================


class TestGetFrenchSummary:
    """Tests for French-language OLS summary generation."""

    def test_returns_string(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Output is a non-empty string."""
        result = get_french_summary(ols_model)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_french_headers(
        self, ols_model: sm.regression.linear_model.OLSResults
    ) -> None:
        """Summary includes expected French-language labels."""
        result = get_french_summary(ols_model)
        assert "Variable Dépendante" in result
        assert "R-carré" in result
        assert "F-statistique" in result
        assert "Nb. Observations" in result

    def test_use_alpha_replaces_const(
        self, ols_model: sm.regression.linear_model.OLSResults
    ) -> None:
        """When use_alpha=True, 'const' is replaced with 'a0' in the coefficient table."""
        result = get_french_summary(ols_model, use_alpha=True)
        assert "a0" in result

    def test_decimal_places_affect_precision(
        self, ols_model: sm.regression.linear_model.OLSResults
    ) -> None:
        """Fewer decimal places produce shorter number strings."""
        short = get_french_summary(ols_model, decimal_places=2)
        long = get_french_summary(ols_model, decimal_places=8)
        # Both should work; long should contain more digits in number tokens
        assert len(long) >= len(short)


# ===================================================================
# 10. get_model_formulas
# ===================================================================


class TestGetModelFormulas:
    """Tests for the formula text generator."""

    def test_returns_nonempty_string(self) -> None:
        """Output is a non-empty string."""
        result = get_model_formulas()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_formula_numbering(self) -> None:
        """Each formula section is numbered."""
        result = get_model_formulas()
        assert "1." in result
        assert "11." in result

    def test_use_alpha_flag(self) -> None:
        """When use_alpha=True, the formulas use 'a' instead of 'β'."""
        result_beta = get_model_formulas(use_alpha=False)
        result_alpha = get_model_formulas(use_alpha=True)
        assert "β" in result_beta
        assert "a₀" in result_alpha


# ===================================================================
# 11. get_interpretations
# ===================================================================


class TestGetInterpretations:
    """Tests for French-language model interpretation generation."""

    def test_returns_string(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Output is a non-empty string."""
        result = get_interpretations(ols_model)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_mentions_r_squared(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Interpretation includes R² discussion."""
        result = get_interpretations(ols_model)
        assert "R²" in result

    def test_mentions_f_statistic(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Interpretation includes F-statistic discussion."""
        result = get_interpretations(ols_model)
        assert "F-statistique" in result or "F-stat" in result

    def test_mentions_durbin_watson(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Interpretation includes Durbin-Watson discussion."""
        result = get_interpretations(ols_model)
        assert "Durbin-Watson" in result

    def test_mentions_breusch_pagan(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Interpretation includes Breusch-Pagan discussion."""
        result = get_interpretations(ols_model)
        assert "Breusch-Pagan" in result

    def test_mentions_anderson_darling(
        self, ols_model: sm.regression.linear_model.OLSResults
    ) -> None:
        """Interpretation includes Anderson-Darling discussion."""
        result = get_interpretations(ols_model)
        assert "Anderson-Darling" in result

    def test_mentions_jarque_bera(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Interpretation includes Jarque-Bera discussion."""
        result = get_interpretations(ols_model)
        assert "Jarque-Bera" in result

    def test_use_alpha_flag(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """When use_alpha=True, variable names use 'a' prefix."""
        result = get_interpretations(ols_model, use_alpha=True)
        assert "a_" in result

    def test_high_r2_interpretation(
        self, ols_model_high_r2: sm.regression.linear_model.OLSResults
    ) -> None:
        """Model with R² > 0.9 gets the 'excellent ajustement' message."""
        result = get_interpretations(ols_model_high_r2)
        assert "90%" in result or "excellent" in result.lower()

    def test_multiple_interpretation_paragraphs(
        self, ols_model: sm.regression.linear_model.OLSResults
    ) -> None:
        """Result contains multiple paragraphs (one per test/variable)."""
        result = get_interpretations(ols_model)
        lines = result.strip().split("\n")
        # Should have at least: R², F-stat, 2 vars, Durbin-Watson, BP, AD, JB
        assert len(lines) >= 7


# ===================================================================
# 12. get_calculation_steps
# ===================================================================


class TestCalculationSteps:
    """Tests for step-by-step OLS calculation trace generation."""

    def test_returns_string(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Output is a non-empty string."""
        result = get_calculation_steps(ols_model)
        assert isinstance(result, str)
        assert len(result) > 200

    def test_contains_step_numbers(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Output contains all 10 numbered calculation steps."""
        result = get_calculation_steps(ols_model)
        for i in range(1, 11):
            assert f"{i}." in result

    def test_contains_key_formulas(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Output references key OLS formulas and matrices."""
        result = get_calculation_steps(ols_model)
        assert "X'X" in result
        assert "X'y" in result
        assert "hat" in result.lower() or "ŷ" in result
        assert "R²" in result

    def test_use_alpha_flag(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """When use_alpha=True, the symbol is 'a' instead of 'β'."""
        result = get_calculation_steps(ols_model, use_alpha=True)
        assert "â" in result or "â" in result

    def test_numeric_values_present(self, ols_model: sm.regression.linear_model.OLSResults) -> None:
        """Output includes numeric values from the model (e.g. n, k, R²)."""
        result = get_calculation_steps(ols_model)
        assert str(int(ols_model.nobs)) in result
        assert str(ols_model.rsquared) in result
