"""Tests for the statistical tables used in the econometrics application.

Validates the Student t, chi-squared, and Fisher-Snedecor tables for correct
dimensions, column structure, and known reference values.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Student t-distribution table
# ---------------------------------------------------------------------------


class TestStudentTableStructure:
    """Structural tests for the Student t-distribution table."""

    def test_row_count(self, student_data: list[tuple[Any, ...]]) -> None:
        """Table should have 101 rows: degrees of freedom 1-100 plus infinity."""
        assert len(student_data) == 101

    def test_last_row_is_infinity(self, student_data: list[tuple[Any, ...]]) -> None:
        """The last row must represent the infinity (normal) reference row."""
        assert student_data[-1][0] == "∞"

    def test_first_row_dl_is_one(self, student_data: list[tuple[Any, ...]]) -> None:
        """The first row corresponds to 1 degree of freedom."""
        assert student_data[0][0] == 1

    def test_row_tuples_have_correct_length(self, student_data: list[tuple[Any, ...]]) -> None:
        """Every row must contain exactly 7 values (dl + 6 alpha columns)."""
        for row in student_data:
            assert len(row) == 7

    def test_column_names(self, student_columns: list[str]) -> None:
        """Expected column headers match the reference specification."""
        expected = ["dl", "0.20", "0.10", "0.05", "0.025", "0.01", "0.005"]
        assert student_columns == expected

    def test_numeric_rows_are_monotone_decreasing_for_0_05(
        self, student_data: list[tuple[Any, ...]]
    ) -> None:
        """For alpha=0.05 (index 3), critical values should decrease as dl increases."""
        values = [row[3] for row in student_data if isinstance(row[0], int)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1], (
                f"Non-monotone at dl={student_data[i][0]}: {values[i - 1]} < {values[i]}"
            )


class TestStudentTableValues:
    """Spot-check known values from the Student t-distribution table."""

    @pytest.mark.parametrize(
        "dl, alpha_col, expected",
        [
            (1, "0.05", 6.314),
            (10, "0.05", 1.812),
            (30, "0.025", 2.042),  # bilateral α=0.05 → upper-tail 0.025
            (30, "0.01", 2.457),
            (100, "0.05", 1.652),
            (100, "0.005", 2.626),
        ],
    )
    def test_known_values(
        self,
        student_data: list[tuple[Any, ...]],
        student_columns: list[str],
        dl: int,
        alpha_col: str,
        expected: float,
    ) -> None:
        """Verify specific critical values against the reference table."""
        col_index = student_columns.index(alpha_col)
        row = next(r for r in student_data if r[0] == dl)
        assert row[col_index] == expected

    def test_infinity_row_normal_approximation(self, student_data: list[tuple[Any, ...]]) -> None:
        """The infinity row should give standard normal z-critical values."""
        inf_row = student_data[-1]
        assert inf_row[0] == "∞"
        assert inf_row[3] == 1.645  # alpha=0.05
        assert inf_row[4] == 1.960  # alpha=0.025
        assert inf_row[5] == 2.326  # alpha=0.01


# ---------------------------------------------------------------------------
# Chi-squared distribution table
# ---------------------------------------------------------------------------


class TestChi2TableStructure:
    """Structural tests for the chi-squared table."""

    def test_row_count(self, chi2_data: list[tuple]) -> None:
        """Table should have 30 rows for degrees of freedom 1 through 30."""
        assert len(chi2_data) == 30

    def test_v_values_are_sequential(self, chi2_data: list[tuple]) -> None:
        """Degrees of freedom should be sequential integers from 1 to 30."""
        v_values = [row[0] for row in chi2_data]
        assert v_values == list(range(1, 31))

    def test_column_count(self, chi2_columns: list[str]) -> None:
        """Table should have 14 columns (v + 13 probability levels)."""
        assert len(chi2_columns) == 14

    def test_row_length(self, chi2_data: list[tuple]) -> None:
        """Every row must have exactly 14 values."""
        for row in chi2_data:
            assert len(row) == 14


class TestChi2TableValues:
    """Spot-check known chi-squared critical values."""

    @pytest.mark.parametrize(
        "v, alpha_col, expected",
        [
            (1, "0.05", 3.84),
            (1, "0.01", 6.63),
            (5, "0.05", 11.07),
            (10, "0.05", 18.31),
            (20, "0.05", 31.41),
            (30, "0.05", 43.77),
            (30, "0.001", 59.70),
        ],
    )
    def test_known_values(
        self,
        chi2_data: list[tuple],
        chi2_columns: list[str],
        v: int,
        alpha_col: str,
        expected: float,
    ) -> None:
        """Verify specific chi-squared critical values against the reference."""
        col_index = chi2_columns.index(alpha_col)
        row = next(r for r in chi2_data if r[0] == v)
        assert row[col_index] == expected


# ---------------------------------------------------------------------------
# Fisher-Snedecor F-distribution tables
# ---------------------------------------------------------------------------

EXPECTED_V1_KEYS: list[Any] = [1, 2, 3, 4, 5, 6, 8, 12, 24, "∞"]


class TestFisherTableStructure:
    """Structural tests for the Fisher-Snedecor tables."""

    def test_v1_keys(self, fisher_tables: dict[Any, list[tuple[Any, float, float]]]) -> None:
        """Fisher tables must cover the specified v1 values."""
        assert list(fisher_tables.keys()) == EXPECTED_V1_KEYS

    @pytest.mark.parametrize("v1", EXPECTED_V1_KEYS)
    def test_each_v1_has_34_rows(
        self,
        fisher_tables: dict[Any, list[tuple[Any, float, float]]],
        v1: Any,
    ) -> None:
        """Each v1 sub-table should have 34 rows (v2 values)."""
        assert len(fisher_tables[v1]) == 34

    @pytest.mark.parametrize("v1", EXPECTED_V1_KEYS)
    def test_row_length(
        self,
        fisher_tables: dict[Any, list[tuple[Any, float, float]]],
        v1: Any,
    ) -> None:
        """Each row must be a 3-tuple: (v2, P=0.05, P=0.01)."""
        for row in fisher_tables[v1]:
            assert len(row) == 3

    @pytest.mark.parametrize("v1", EXPECTED_V1_KEYS)
    def test_last_row_is_infinity(
        self,
        fisher_tables: dict[Any, list[tuple[Any, float, float]]],
        v1: Any,
    ) -> None:
        """The last v2 entry in every sub-table should be infinity."""
        assert fisher_tables[v1][-1][0] == "∞"


class TestFisherTableValues:
    """Spot-check known Fisher-Snedecor F critical values."""

    def test_fisher_v1_1_v2_30_p0_05(
        self, fisher_tables: dict[Any, list[tuple[Any, float, float]]]
    ) -> None:
        """F(1, 30) at P=0.05 should be 4.17."""
        row = next(r for r in fisher_tables[1] if r[0] == 30)
        assert row[1] == 4.17

    def test_fisher_v1_1_v2_30_p0_01(
        self, fisher_tables: dict[Any, list[tuple[Any, float, float]]]
    ) -> None:
        """F(1, 30) at P=0.01 should be 7.56."""
        row = next(r for r in fisher_tables[1] if r[0] == 30)
        assert row[2] == 7.56

    def test_fisher_v1_inf_v2_inf(
        self, fisher_tables: dict[Any, list[tuple[Any, float, float]]]
    ) -> None:
        """F(∞, ∞) should be 1.00 at both probability levels."""
        row = fisher_tables["∞"][-1]
        assert row == ("∞", 1.00, 1.00)

    def test_fisher_v1_5_v2_10_p0_05(
        self, fisher_tables: dict[Any, list[tuple[Any, float, float]]]
    ) -> None:
        """F(5, 10) at P=0.05 should be 3.33."""
        row = next(r for r in fisher_tables[5] if r[0] == 10)
        assert row[1] == 3.33

    def test_fisher_v1_24_v2_60_p0_01(
        self, fisher_tables: dict[Any, list[tuple[Any, float, float]]]
    ) -> None:
        """F(24, 60) at P=0.01 should be 2.12."""
        row = next(r for r in fisher_tables[24] if r[0] == 60)
        assert row[2] == 2.12

    def test_fisher_first_entry_v1_1(
        self, fisher_tables: dict[Any, list[tuple[Any, float, float]]]
    ) -> None:
        """F(1, 1) at P=0.05 should be 161.4."""
        row = fisher_tables[1][0]
        assert row == (1, 161.4, 4052.0)

    @pytest.mark.parametrize("v1", EXPECTED_V1_KEYS)
    def test_f_values_decrease_with_v2(
        self,
        fisher_tables: dict[Any, list[tuple[Any, float, float]]],
        v1: Any,
    ) -> None:
        """For a fixed v1, F critical values should decrease as v2 increases."""
        p05_values = [row[1] for row in fisher_tables[v1]]
        for i in range(1, len(p05_values)):
            assert p05_values[i] <= p05_values[i - 1], (
                f"Non-monotone at v1={v1}, row {i}: {p05_values[i - 1]} < {p05_values[i]}"
            )
