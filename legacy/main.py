import tkinter as tk
from tkinter import filedialog, messagebox, Text
from transformers import pipeline
import pdfplumber
import docx

# Initialiser le modèle de résumé
summarizer = pipeline("summarization")

# Chemin du fichier global
file_path = ""

# Fonction pour extraire le texte d'un fichier
def extract_text(file_path):
    text = ""
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        messagebox.showerror("Erreur", "Format de fichier non pris en charge.")
    return text

# Fonction pour générer un résumé
def summarize_text():
    if not file_path:
        messagebox.showerror("Erreur", "Veuillez sélectionner un fichier.")
        return
    text = extract_text(file_path)
    if text:
        try:
            # Limite pour le modèle de résumé (500 tokens par lot)
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, summary[0]['summary_text'])
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du résumé : {str(e)}")

# Fonction pour sélectionner un fichier
def open_file():
    global file_path
    file_path = filedialog.askopenfilename(
        title="Sélectionner un fichier",
        filetypes=(("Fichiers PDF", "*.pdf"), ("Fichiers Word", "*.docx"), ("Fichiers TXT", "*.txt"))
    )
    if file_path:
        file_label.config(text=f"Fichier sélectionné : {file_path}")

# Interface graphique avec Tkinter
root = tk.Tk()
root.title("Résumé Automatique de Documents")
root.geometry("600x400")
root.config(bg="#f2f2f2")

title_label = tk.Label(root, text="Résumé Automatique de Documents", font=("Arial", 16), bg="#f2f2f2")
title_label.pack(pady=10)

file_label = tk.Label(root, text="Aucun fichier sélectionné", bg="#f2f2f2", fg="#555")
file_label.pack()

# Bouton pour sélectionner un fichier
open_file_button = tk.Button(root, text="Sélectionner un Fichier", command=open_file, bg="#4CAF50", fg="white")
open_file_button.pack(pady=10)

# Bouton pour résumer
summarize_button = tk.Button(root, text="Générer le Résumé", command=summarize_text, bg="#2196F3", fg="white")
summarize_button.pack(pady=10)

# Zone de texte pour afficher le résumé
result_text = Text(root, wrap=tk.WORD, height=10, width=70, padx=10, pady=10)
result_text.pack(pady=10)

root.mainloop()
