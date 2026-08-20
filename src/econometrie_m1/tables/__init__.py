"""Tables statistiques de reference (Student, Khi-deux, Fisher-Snedecor)."""

from econometrie_m1.tables.statistical_tables import (
    CHI2_COLUMNS,
    CHI2_DATA,
    FISHER_COLUMNS,
    FISHER_TABLES,
    STUDENT_COLUMNS,
    STUDENT_DATA,
    Chi2Row,
    FisherRow,
    StudentRow,
    get_chi2_data,
    get_chi2_row,
    get_fisher_critical,
    get_fisher_table,
    get_fisher_tables,
    get_student_data,
    get_student_row,
)

__all__ = [
    "CHI2_COLUMNS",
    "CHI2_DATA",
    "FISHER_COLUMNS",
    "FISHER_TABLES",
    "STUDENT_COLUMNS",
    "STUDENT_DATA",
    "Chi2Row",
    "FisherRow",
    "StudentRow",
    "get_chi2_data",
    "get_chi2_row",
    "get_fisher_critical",
    "get_fisher_table",
    "get_fisher_tables",
    "get_student_data",
    "get_student_row",
]
