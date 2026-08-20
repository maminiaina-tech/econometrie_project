# Econometrie

Outil d'analyse econometrique avancee pour le Master 1 en Sciences de Données et Intelligence Artificielle. Application de bureau Python offrant une interface graphique complete pour l'estimation et le diagnostic de modeles de regression lineaire.

## Fonctionnalites

### Estimation de modeles
- Regression par moindres carrés ordinaires (MCO/OLS)
- Estimation des coefficients avec erreurs standards, statistiques t et p-values
- R², R² ajusté, F-statistique, AIC, BIC, log-vraisemblance

### Analyse complete
- Matrices X, X'X, X'Y, (X'X)⁻¹, variance-covariance
- Tableau des resultats avec valeurs observees, predites et residus
- Steps de calcul detailles et formules mathematiques

### Tests de diagnostic
- **Durbin-Watson** : autocorrelation des residus
- **Breusch-Pagan** : heteroscedasticite
- **Anderson-Darling** : normalite
- **Jarque-Bera** : normalite

### Multicollinearite
- Facteurs d'Inflation de la Variance (VIF)
- Indices de condition et valeurs propres
- Test de Klein
- Test de Farrar-Glauber (Chi², tests F, tests t)

### Correlation partielle
- Correlations partielles de tous ordres
- R² partiels (contribution marginale de chaque variable)

### Tests statistiques
- Test de significativite individuelle (test t)
- Test de significativite globale (test F)
- Determination du mix optimal par elasticites

### Import/Export
- Import Excel (avec apercu interactif) et CSV
- Export Excel (multi-feuilles), Word (DOCX), PDF (paysage)
- Export de graphiques (PNG)

### Graphiques de diagnostic
- Residus vs valeurs ajustees
- QQ Plot
- Leverage
- Histogramme des residus
- Matrice de correlation
- Autocorrelogramme (ACF/PACF)

## Installation

### Depuis les sources

```bash
git clone https://github.com/maminiaina-tech/econometrie_project
cd econometrie_project
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

### Dependancees principales

- Python >= 3.10
- numpy, pandas, statsmodels, scipy
- matplotlib, seaborn
- fpdf2, python-docx, openpyxl
- sympy

## Utilisation

### Lancer l'application principale

```bash
python -m econometrie_m1.app
```

### Depuis Python

```python
from econometrie_m1.app import main
main()
```

### Lancer les tests

```bash
pytest tests/ -v
```

### Linter le code

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Structure du projet

```
econometrie_m1/
├── src/
│   └── econometrie_m1/
│       ├── __init__.py
│       ├── app.py                    # Application principale (IHM)
│       ├── tables/
│       │   ├── __init__.py
│       │   └── statistical_tables.py # Tables de Student, Chi², Fisher
│       ├── computations/
│       │   ├── __init__.py
│       │   └── stats.py              # Calculs econometriques pures
│       └── export/
│           ├── __init__.py
│           └── exporters.py          # Export Excel, Word, PDF
├── tests/
│   ├── conftest.py
│   └── test_statistical_tables.py
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
└── README.md
```

## Configuration

L'application propose un menu de configuration permettant de :
- Modifier le nombre de decimales affichees
- Ajuster le seuil de significativite (alpha)
- Afficher/masquer les formules, interpretations et etapes de calcul
- Changer la notation (α ou β)

## License

MIT
