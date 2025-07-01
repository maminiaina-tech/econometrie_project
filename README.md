# Résumé Automatique de Documents

Cette application est une interface de bureau développée en Python avec Tkinter, permettant de générer des résumés automatiques de documents (PDF, DOCX, TXT) 
en utilisant des modèles de traitement du langage naturel (NLP). Elle inclut également une fonctionnalité de traduction, ainsi que la possibilité de télécharger les résumés en formats DOCX et PDF.

## Fonctionnalités

- **Extraction de texte** : Supporte les formats PDF, DOCX, et TXT.
- **Génération de résumés** : Utilise un modèle de résumé NLP pour résumer le contenu extrait.
- **Traduction** : Traduction automatique des résumés entre l'anglais et le français.
- **Téléchargement** : Permet de télécharger les résumés en fichiers PDF et DOCX.
- **Historique** : Enregistre et affiche l'historique des résumés générés et traduits.

## Prérequis

Avant de démarrer, assurez-vous d'avoir installé les dépendances suivantes :

- Python 3.x
- Tkinter (inclus avec la plupart des installations Python)
- `transformers` pour les modèles NLP
- `pdfplumber` pour extraire le texte des fichiers PDF
- `python-docx` pour lire et créer des fichiers DOCX
- `fpdf` pour créer des fichiers PDF

### Installation des dépendances

Pour installer les bibliothèques nécessaires, exécutez les commandes suivantes :

```bash
pip install transformers pdfplumber python-docx fpdf


