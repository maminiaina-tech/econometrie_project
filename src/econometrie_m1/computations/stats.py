"""Pure statistical computation functions extracted from econometrie_app.py."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

if TYPE_CHECKING:
    from statsmodels.regression.linear_model import OLSResults


# ---------------------------------------------------------------------------
# Dataclasses for structured results
# ---------------------------------------------------------------------------


@dataclass
class OutlierColumnResult:
    column: str
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    outlier_count: int
    outlier_indices: list[int]


@dataclass
class OutlierDetectionResult:
    columns: list[OutlierColumnResult]
    all_outlier_indices: list[int]


@dataclass
class PartialCorrelationResult:
    var1: str
    var2: str
    control_vars: tuple[str, ...]
    order: int
    correlation: float


@dataclass
class PartialR2Result:
    variable: str
    r2_partial: float


@dataclass
class PartialCorrelationsFullResult:
    partial_correlations: list[PartialCorrelationResult]
    partial_r2: list[PartialR2Result]


@dataclass
class KleinTestResult:
    r_squared: float
    pair_results: dict[tuple[str, str], tuple[float, float]]
    multicollinear_detected: bool


@dataclass
class FarrarGlobalResult:
    determinant: float
    chi2: float
    df: int
    p_value: float


@dataclass
class FarrarFTestResult:
    variable: str
    f_value: float
    f_pvalue: float


@dataclass
class FarrarTTestResult:
    var1: str
    var2: str
    other_vars: list[str]
    t_value: float
    p_value: float


@dataclass
class FarrarGlauberResult:
    global_test: FarrarGlobalResult
    f_tests: list[FarrarFTestResult]
    t_tests: list[FarrarTTestResult]


@dataclass
class OptimalMixResult:
    elasticities: dict[str, float]
    optimal_shares: dict[str, float]


@dataclass
class MulticollinearityVIFResult:
    variable: str
    vif: float
    r_squared_aux: float | None = None


@dataclass
class MulticollinearityResult:
    vif_data: list[MulticollinearityVIFResult]
    eigenvalues: list[float]
    condition_indices: list[float]
    correlation_matrix: pd.DataFrame
    max_condition_index: float


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def format_number(num: int | float | Any, decimal_places: int = 6) -> str:
    """Format a number using fixed-point notation.

    Args:
        num: The number to format.
        decimal_places: Number of decimal places.

    Returns:
        Formatted string.
    """
    if isinstance(num, (int, float)):
        return f"{num:.{decimal_places}f}"
    return str(num)


# ---------------------------------------------------------------------------
# Outlier detection / correction
# ---------------------------------------------------------------------------


def detect_outliers(data: pd.DataFrame) -> OutlierDetectionResult:
    """Detect outliers using the IQR method on every numeric column.

    Args:
        data: Input dataframe.

    Returns:
        Structured detection result with per-column statistics.
    """
    columns: list[OutlierColumnResult] = []
    all_indices: list[int] = []

    for col in data.select_dtypes(include=[np.number]).columns:
        q1 = float(data[col].quantile(0.25))
        q3 = float(data[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (data[col] < lower) | (data[col] > upper)
        outlier_idx = data.index[mask].tolist()

        columns.append(
            OutlierColumnResult(
                column=col,
                q1=q1,
                q3=q3,
                iqr=iqr,
                lower_bound=lower,
                upper_bound=upper,
                outlier_count=len(outlier_idx),
                outlier_indices=outlier_idx,
            )
        )
        all_indices.extend(outlier_idx)

    return OutlierDetectionResult(
        columns=columns,
        all_outlier_indices=sorted(set(all_indices)),
    )


def correct_outliers(data: pd.DataFrame, outlier_indices: list[int]) -> pd.DataFrame:
    """Replace outliers with the per-column median.

    Args:
        data: Original dataframe.
        outlier_indices: Row indices to replace.

    Returns:
        Corrected dataframe (copy).
    """
    corrected = data.copy()

    for col in corrected.select_dtypes(include=[np.number]).columns:
        q1 = float(corrected[col].quantile(0.25))
        q3 = float(corrected[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        median_val = corrected[col].median()
        corrected.loc[(corrected[col] < lower) | (corrected[col] > upper), col] = median_val

    return corrected


# ---------------------------------------------------------------------------
# Partial correlations & partial R²
# ---------------------------------------------------------------------------


def _partial_corr(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    control_vars: list[str],
) -> float:
    """Compute partial correlation between *var1* and *var2* controlling for *control_vars*."""
    if len(control_vars) > 0:
        res1 = sm.OLS(df[var1], sm.add_constant(df[control_vars])).fit()
        res2 = sm.OLS(df[var2], sm.add_constant(df[control_vars])).fit()
        return float(np.corrcoef(res1.resid, res2.resid)[0, 1])
    return float(df[[var1, var2]].corr().iloc[0, 1])


def calculate_partial_correlations(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
) -> PartialCorrelationsFullResult:
    """Calculate systematic partial correlations and partial R².

    Args:
        data: Input dataframe.
        y_var: Dependent variable name.
        x_vars: List of independent variable names (>= 2).

    Returns:
        Full result with all partial correlations and partial R².
    """
    df = data.copy()

    partial_corrs: list[PartialCorrelationResult] = []

    for order in range(1, len(x_vars)):
        for i, var1 in enumerate(x_vars):
            for var2 in x_vars[i + 1 :]:
                all_controls = [v for v in x_vars if v not in [var1, var2]]
                if len(all_controls) >= order:
                    for control_vars in combinations(all_controls, order):
                        corr = _partial_corr(df, var1, var2, list(control_vars))
                        partial_corrs.append(
                            PartialCorrelationResult(
                                var1=var1,
                                var2=var2,
                                control_vars=tuple(control_vars),
                                order=order,
                                correlation=corr,
                            )
                        )

    # Partial R²
    partial_r2_list: list[PartialR2Result] = []
    full_model = sm.OLS(df[y_var], sm.add_constant(df[x_vars])).fit()
    for var in x_vars:
        reduced_vars = [v for v in x_vars if v != var]
        reduced_model = sm.OLS(df[y_var], sm.add_constant(df[reduced_vars])).fit()
        r2_partial = (full_model.ess - reduced_model.ess) / (1 - reduced_model.rsquared)
        partial_r2_list.append(PartialR2Result(variable=var, r2_partial=r2_partial))

    return PartialCorrelationsFullResult(
        partial_correlations=partial_corrs,
        partial_r2=partial_r2_list,
    )


# ---------------------------------------------------------------------------
# Klein test
# ---------------------------------------------------------------------------


def klein_test(
    r_squared: float,
    data: pd.DataFrame,
    x_vars: list[str],
) -> KleinTestResult:
    """Perform the Klein test for multicollinearity.

    For each pair (Xi, Xj) the test compares R² of the full model to r²(Xi, Xj).
    If R²_model < r²_pair, multicollinearity is flagged.

    Args:
        r_squared: R² of the full model.
        data: Input dataframe containing the x_vars.
        x_vars: Independent variable names.

    Returns:
        Structured result.
    """
    corr_matrix = data[x_vars].corr()
    pair_results: dict[tuple[str, str], tuple[float, float]] = {}
    multicollinear = False

    for i in range(len(x_vars)):
        for j in range(i + 1, len(x_vars)):
            v1, v2 = x_vars[i], x_vars[j]
            corr_sq = float(corr_matrix.loc[v1, v2] ** 2)
            pair_results[(v1, v2)] = (r_squared, corr_sq)
            if r_squared < corr_sq:
                multicollinear = True

    return KleinTestResult(
        r_squared=r_squared,
        pair_results=pair_results,
        multicollinear_detected=multicollinear,
    )


# ---------------------------------------------------------------------------
# Farrar-Glauber test
# ---------------------------------------------------------------------------


def farrar_glauber_test(
    data: pd.DataFrame,
    x_vars: list[str],
) -> FarrarGlauberResult:
    """Perform the Farrar-Glauber three-step multicollinearity test.

    Args:
        data: Input dataframe containing the x_vars.
        x_vars: Independent variable names.

    Returns:
        Structured result with global χ², F-tests, and t-tests.
    """
    X = data[x_vars]
    corr_matrix = X.corr()
    n = len(X)
    p = len(x_vars)

    # 1. Global χ² test
    det = float(np.linalg.det(corr_matrix))
    chi2 = -((n - 1) - (2 * p + 5) / 6) * np.log(det)
    df = p * (p - 1) // 2
    p_value = float(1 - stats.chi2.cdf(chi2, df))
    global_test = FarrarGlobalResult(
        determinant=det,
        chi2=chi2,
        df=df,
        p_value=p_value,
    )

    # 2. F-tests on auxiliary regressions
    f_tests: list[FarrarFTestResult] = []
    for var in x_vars:
        other_vars = [v for v in x_vars if v != var]
        model = sm.OLS(data[var], sm.add_constant(data[other_vars])).fit()
        f_tests.append(
            FarrarFTestResult(
                variable=var,
                f_value=float(model.fvalue),
                f_pvalue=float(model.f_pvalue),
            )
        )

    # 3. t-tests on partial correlations
    t_tests: list[FarrarTTestResult] = []
    for i in range(len(x_vars)):
        for j in range(i + 1, len(x_vars)):
            v1, v2 = x_vars[i], x_vars[j]
            other_vars = [v for v in x_vars if v not in [v1, v2]]
            if len(other_vars) > 0:
                res1 = sm.OLS(data[v1], sm.add_constant(data[other_vars])).fit()
                res2 = sm.OLS(data[v2], sm.add_constant(data[other_vars])).fit()
                corr, pv = stats.pearsonr(res1.resid, res2.resid)
                n_res = len(res1.resid)
                t_val = corr * np.sqrt((n_res - 2) / (1 - corr**2))
                t_tests.append(
                    FarrarTTestResult(
                        var1=v1,
                        var2=v2,
                        other_vars=other_vars,
                        t_value=float(t_val),
                        p_value=float(pv),
                    )
                )

    return FarrarGlauberResult(
        global_test=global_test,
        f_tests=f_tests,
        t_tests=t_tests,
    )


# ---------------------------------------------------------------------------
# Optimal mix via elasticities
# ---------------------------------------------------------------------------


def determine_optimal_mix(
    model: OLSResults,
    data: pd.DataFrame,
    x_vars: list[str],
) -> OptimalMixResult:
    """Determine the optimal mix proportions based on variable elasticities.

    Args:
        model: A fitted OLS model (statsmodels).
        data: Original input dataframe.
        x_vars: Independent variable names.

    Returns:
        Elasticities and optimal share percentages.
    """
    y_mean = float(np.mean(model.model.endog))
    x_means = [float(data[var].mean()) for var in x_vars]
    coefficients = [float(model.params[var]) for var in x_vars]

    elasticities: dict[str, float] = {}
    for coef, x_mean, var in zip(coefficients, x_means, x_vars):
        elasticities[var] = coef * (x_mean / y_mean)

    total = sum(abs(e) for e in elasticities.values())
    optimal_shares = {var: abs(e) / total for var, e in elasticities.items()}

    return OptimalMixResult(
        elasticities=elasticities,
        optimal_shares=optimal_shares,
    )


# ---------------------------------------------------------------------------
# Multicollinearity analysis (VIF, condition indices, correlation matrix)
# ---------------------------------------------------------------------------


def analyze_multicollinearity(
    X: pd.DataFrame,
) -> MulticollinearityResult:
    """Compute VIFs, eigenvalues, condition indices, and the correlation matrix.

    Args:
        X: Design matrix **including** a 'const' column (will be dropped for
           eigenvalue / condition index / correlation calculations).

    Returns:
        Structured multicollinearity diagnostics.
    """
    # VIF
    vif_data: list[MulticollinearityVIFResult] = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif_val = float(variance_inflation_factor(X.values, i))
        vif_data.append(MulticollinearityVIFResult(variable=col, vif=vif_val))

    vif_data.sort(key=lambda v: v.vif, reverse=True)

    # Eigenvalues & condition indices
    X_no_const = X.drop(columns=["const"]) if "const" in X.columns else X
    X_centered = X_no_const - X_no_const.mean()
    eigvals = np.linalg.eigvals(X_centered.T @ X_centered)
    eigvals_list = sorted(eigvals.real.tolist(), reverse=True)
    max_eig = max(eigvals_list)
    condition_indices = [math.sqrt(max_eig / ev) if ev > 0 else float("inf") for ev in eigvals_list]

    # Correlation matrix
    corr_matrix = X_no_const.corr()

    return MulticollinearityResult(
        vif_data=vif_data,
        eigenvalues=eigvals_list,
        condition_indices=condition_indices,
        correlation_matrix=corr_matrix,
        max_condition_index=max(condition_indices),
    )


# ---------------------------------------------------------------------------
# Text generation helpers
# ---------------------------------------------------------------------------


def get_french_summary(
    model: OLSResults,
    decimal_places: int = 6,
    use_alpha: bool = False,
) -> str:
    """Return a French-language OLS summary string.

    Args:
        model: A fitted OLS model.
        decimal_places: Decimal places for number formatting.
        use_alpha: Use ``alpha`` instead of ``beta`` in the coefficient table.

    Returns:
        Summary string.
    """

    summary_tables = model.summary().tables
    coef_table: str = ""
    if len(summary_tables) > 1:
        coef_table = summary_tables[1].as_text()
        if use_alpha:
            coef_table = coef_table.replace("beta", "alpha").replace("const", "a0")

    now = datetime.now()
    date_str = now.strftime("%a, %d %b %Y")
    time_str = now.strftime("%H:%M:%S")

    return (
        f"Variable Dépendante:           {model.model.endog_names}   R-carré:                       {format_number(model.rsquared, decimal_places)}\n"
        f"Modèle:                       OLS   R-carré ajusté:              {format_number(model.rsquared_adj, decimal_places)}\n"
        f"Méthode:          Moindres Carrés   F-statistique:               {format_number(model.fvalue, decimal_places)}\n"
        f"Date:             {date_str}   Prob (F-statistique):          {format_number(model.f_pvalue, decimal_places)}\n"
        f"Heure:                 {time_str}   Log-vraisemblance:            {format_number(model.llf, decimal_places)}\n"
        f"Nb. Observations:      {model.nobs}   AIC:                          {format_number(model.aic, decimal_places)}\n"
        f"Df Résidus:          {model.df_resid}   BIC:                          {format_number(model.bic, decimal_places)}\n"
        f"Df Modèle:            {model.df_model}                                         \n"
        "Covariance Type:   nonrobuste                                         \n"
        "==============================================================================\n"
        f"{coef_table}\n"
        "==============================================================================\n"
    )


def get_model_formulas(use_alpha: bool = False) -> str:
    """Return the formatted formulas text.

    Args:
        use_alpha: Use ``alpha`` instead of ``beta``.

    Returns:
        Multi-line formula string.
    """
    s = "a" if use_alpha else "β"
    return (
        f"1. Modèle de régression:\n"
        f"   Y = {s}₀ + {s}₁X₁ + {s}₂X₂ + ... + {s}ₖXₖ + ε\n\n"
        f"2. Estimateur MCO:\n"
        f"   {s}̂ = (X'X)⁻¹X'y\n\n"
        "3. Matrice de projection (hat matrix):\n"
        "   H = X(X'X)⁻¹X'\n"
        "   ŷ = Hy = Xβ̂ (valeurs prédites)\n"
        "   h_ii = éléments diagonaux de H (levier)\n\n"
        "4. Variance des résidus:\n"
        f"   σ̂² = (y - X{s}̂)'(y - X{s}̂) / (n - k) = ε̂'ε̂ / (n - k)\n\n"
        f"5. Matrice variance-covariance:\n"
        f"   Var({s}̂) = σ̂²(X'X)⁻¹\n\n"
        f"6. Statistique t:\n"
        f"   t = {s}̂ᵢ / se({s}̂ᵢ)\n\n"
        "7. Statistique F:\n"
        "   F = [(SCE₀ - SCE₁)/q] / [SCE₁/(n - k)]\n"
        "   où SCE = somme des carrés des résidus\n\n"
        "8. R² et R² ajusté:\n"
        "   R² = 1 - SCE/SCT\n"
        "   R²_adj = 1 - (1-R²)(n-1)/(n-k-1)\n\n"
        "9. Corrélation partielle:\n"
        "   ρ(X1,X2|Z) = corr(e1, e2)\n"
        "   où e1 = résidus de X1~Z, e2 = résidus de X2~Z\n\n"
        "10. Test de Klein (multicolinéarité):\n"
        "    Si R²_y < r²_xi,xj ⇒ multicolinéarité\n\n"
        "11. Test de Farrar-Glauber (multicolinéarité):\n"
        "    a) Test χ² global\n"
        "    b) Tests F sur régressions auxiliaires\n"
        "    c) Tests t sur corrélations partielles\n"
    )


def get_interpretations(
    model: OLSResults,
    decimal_places: int = 6,
    use_alpha: bool = False,
) -> str:
    """Generate French-language interpretation paragraphs for the model.

    Args:
        model: A fitted OLS model.
        decimal_places: Decimals for formatting.
        use_alpha: Use ``alpha`` in variable names.

    Returns:
        Joined interpretation text.
    """
    s = "a" if use_alpha else "β"
    interpretations: list[str] = []

    rsq = model.rsquared
    if rsq > 0.9:
        interpretations.append(
            f"Le R² de {format_number(rsq, decimal_places)} indique que le modèle explique plus de 90% "
            "de la variabilité de Y, ce qui suggère un excellent ajustement."
        )
    elif rsq > 0.7:
        interpretations.append(
            f"Le R² de {format_number(rsq, decimal_places)} indique que le modèle explique une grande partie "
            "de la variabilité de Y."
        )
    elif rsq > 0.5:
        interpretations.append(
            f"Le R² de {format_number(rsq, decimal_places)} indique que le modèle explique environ la moitié "
            "de la variabilité de Y."
        )
    else:
        interpretations.append(
            f"Le R² de {format_number(rsq, decimal_places)} est relativement faible, ce qui suggère que le "
            "modèle explique peu de la variabilité de Y."
        )

    f_pval = model.f_pvalue
    if f_pval < 0.05:
        interpretations.append(
            f"La F-statistique significative (p-value = {format_number(f_pval, decimal_places)}) "
            "indique que le modèle dans son ensemble est statistiquement significatif."
        )
    else:
        interpretations.append(
            f"La F-statistique non significative (p-value = {format_number(f_pval, decimal_places)}) "
            "suggère que le modèle n'est pas meilleur qu'un modèle sans variables explicatives."
        )

    for i, var in enumerate(model.params.index):
        if var == "const":
            continue
        coef = float(model.params.iloc[i])
        pval = float(model.pvalues.iloc[i])
        var_name = f"{s}_{var}"
        if pval < 0.05:
            interpretations.append(
                f"La variable {var_name} est significative (p-value = {format_number(pval, decimal_places)}). "
                f"Une augmentation d'une unité de {var} est associée à une variation de "
                f"{format_number(coef, decimal_places)} unités de Y, toutes choses égales par ailleurs."
            )
        else:
            interpretations.append(
                f"La variable {var_name} n'est pas significative (p-value = {format_number(pval, decimal_places)}), "
                "ce qui suggère qu'elle pourrait ne pas avoir d'impact sur Y dans ce modèle."
            )

    # Durbin-Watson
    from statsmodels.stats.stattools import durbin_watson

    dw = durbin_watson(model.resid)
    if dw < 1.5:
        interpretations.append(
            "La valeur du test de Durbin-Watson suggère une possible autocorrélation positive des résidus, "
            "ce qui peut indiquer une spécification incorrecte du modèle."
        )
    elif dw > 2.5:
        interpretations.append(
            "La valeur du test de Durbin-Watson suggère une possible autocorrélation négative des résidus."
        )
    else:
        interpretations.append(
            "Le test de Durbin-Watson ne détecte pas d'autocorrélation significative des résidus."
        )

    # Breusch-Pagan
    from statsmodels.stats.diagnostic import het_breuschpagan

    bp = het_breuschpagan(model.resid, model.model.exog)
    if bp[1] < 0.05:
        interpretations.append(
            "Le test de Breusch-Pagan rejette l'hypothèse d'homoscédasticité (p-value < 0.05), "
            "ce qui suggère la présence d'hétéroscédasticité."
        )
    else:
        interpretations.append(
            "Le test de Breusch-Pagan ne rejette pas l'hypothèse d'homoscédasticité, ce qui est une bonne "
            "nouvelle pour les propriétés des estimateurs MCO."
        )

    # Anderson-Darling
    from statsmodels.stats._adnorm import normal_ad

    ad = normal_ad(model.resid)
    if ad[1] < 0.05:
        interpretations.append(
            "Le test d'Anderson-Darling rejette l'hypothèse de normalité des résidus (p-value < 0.05), "
            "ce qui peut affecter la validité des tests d'hypothèse."
        )
    else:
        interpretations.append(
            "Le test d'Anderson-Darling ne rejette pas l'hypothèse de normalité des résidus, ce qui est "
            "favorable pour l'inférence statistique."
        )

    # Jarque-Bera
    from statsmodels.stats.stattools import jarque_bera

    jb = jarque_bera(model.resid)
    if jb[1] < 0.05:
        interpretations.append(
            "Le test de Jarque-Bera rejette l'hypothèse de normalité des résidus (p-value < 0.05), "
            "confirmant la non-normalité."
        )
    else:
        interpretations.append(
            "Le test de Jarque-Bera ne rejette pas l'hypothèse de normalité des résidus."
        )

    return "\n".join(interpretations)


def get_calculation_steps(
    model: OLSResults,
    use_alpha: bool = False,
) -> str:
    """Generate a step-by-step calculation trace for the OLS model.

    Args:
        model: A fitted OLS model.
        use_alpha: Use ``alpha`` instead of ``beta``.

    Returns:
        Multi-line string with numbered calculation steps.
    """
    s = "a" if use_alpha else "β"
    X = model.model.exog
    y = model.model.endog

    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    XtX_inv = np.linalg.inv(XtX)
    H = np.dot(X, np.dot(XtX_inv, X.T))
    h_diag = np.diag(H)

    return (
        f"1. Estimation des coefficients ({s}):\n"
        f"   {s}̂ = (X'X)⁻¹X'y\n"
        f"   X'X = \n{XtX}\n\n"
        f"   X'y = \n{Xty}\n\n"
        f"   (X'X)⁻¹ = \n{XtX_inv}\n\n"
        f"   {s}̂ = \n{model.params}\n\n"
        f"2. Calcul des valeurs prédites (ŷ):\n"
        f"   ŷ = X{s}̂\n"
        f"   Exemple pour la première observation: \n{X[0]} * {s}̂ = {np.dot(X[0], model.params)}\n\n"
        "3. Calcul des résidus (e):\n"
        "   e = y - ŷ\n"
        f"   Exemple pour la première observation: {y[0]} - {model.fittedvalues[0]} = {model.resid[0]}\n\n"
        "4. Calcul de la variance des résidus (σ²):\n"
        "   σ̂² = e'e / (n - k)\n"
        f"   e'e = {np.dot(model.resid, model.resid)}\n"
        f"   n = {model.nobs}, k = {len(model.params)}\n"
        f"   σ̂² = {model.mse_resid}\n"
        f"   σ = {np.sqrt(model.mse_resid)}\n\n"
        f"5. Calcul des écarts-types des coefficients:\n"
        f"   Ω = Var({s}̂) = σ̂²(X'X)⁻¹\n"
        f"   se({s}̂_j) = √(Var({s}̂)_jj)\n"
        f"   Matrice Var({s}̂): \n{model.cov_params()}\n"
        f"   Écarts-Type: \n{model.bse}\n\n"
        f"6. Calcul des statistiques t:\n"
        f"   t_j = {s}̂_j / se({s}̂_j)\n"
        f"   Statistiques t: \n{model.tvalues}\n\n"
        "7. Calcul du R²:\n"
        "   SCT = Σ(y_i - ȳ)²\n"
        "   SCE = Σ(y_i - ŷ_i)²\n"
        "   R² = 1 - SCE/SCT\n"
        f"   SCT = {np.sum((y - np.mean(y)) ** 2)}\n"
        f"   SCE = {np.sum(model.resid**2)}\n"
        f"   R² = {model.rsquared}\n\n"
        "8. Calcul du R² ajusté:\n"
        "   R²_adj = 1 - (1-R²)(n-1)/(n-k-1)\n"
        f"   R²_adj = {model.rsquared_adj}\n\n"
        "9. Calcul de la F-statistique:\n"
        "   F = [(SCT - SCE)/k] / [SCE/(n - k - 1)]\n"
        f"   F = {model.fvalue}\n\n"
        "10. Calcul de la matrice de projection (hat matrix H):\n"
        "   H = X(X'X)⁻¹X'\n"
        f"   Dimensions de H: {H.shape}\n"
        "   Exemple de sous-matrice de H (5 premières lignes/colonnes):\n"
        f"{H[:5, :5]}\n"
        "   Éléments diagonaux (leviers) h_ii (5 premiers):\n"
        f"{h_diag[:5]}\n"
        "   Trace de H (somme des leviers): tr(H) = k = \n"
        f"{np.trace(H)} (nombre de variables explicatives + constante)\n\n"
    )
