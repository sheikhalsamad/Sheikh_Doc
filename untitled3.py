# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:07:06 2024

@author: cheik
"""

# Installation de la bibliothèque prince (si ce n'est pas déjà fait)


# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier Excel
file_path = r'C:\Users\cheik\Downloads\TEST.xlsx'  # Mettre le chemin correct
df = pd.read_excel(file_path)

# Remplacer les valeurs comme 'NO GENDER' par 'Unknown'
df['Gender'].replace('NO GENDER', 'Unknown', inplace=True)

# 1. Créer un tableau croisé dynamique (pivot table) pour compter les occurrences
# de chaque Gender dans chaque catégorie de voiture
pivot_table = pd.crosstab(df['Car_Category'], df['Gender'])

# Afficher la table de fréquence
print(pivot_table)

# 2. Visualisation des résultats avec un graphique en barres empilées
pivot_table.plot(kind='bar', stacked=True, figsize=(10, 6))

# Configuration du graphique
plt.title("Répartition des catégories de voitures selon le genre")
plt.xlabel("Catégories de voitures")
plt.ylabel("Nombre de personnes")
plt.legend(title="Gender", loc='upper right')
plt.show()

# Calculer les proportions (en pourcentages) de chaque genre dans chaque catégorie de voiture
pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0)

# Tracer le graphique en pourcentages empilés
pivot_table_percentage.plot(kind='bar', stacked=True, figsize=(10, 6))

# Configuration du graphique
plt.title("Répartition en pourcentage des catégories de voitures selon le genre")
plt.xlabel("Catégories de voitures")
plt.ylabel("Pourcentage")
plt.legend(title="Gender", loc='upper right')
plt.show()

# 1. Visualiser la répartition du Genre par rapport à 'Subject_Car_Make'
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Subject_Car_Make', hue='Gender')
plt.title('Répartition du Genre en fonction de Subject Car Make')
plt.xlabel('Subject Car Make')
plt.ylabel('Nombre')
plt.xticks(rotation=45)
plt.legend(title='Genre')
plt.show()

# 2. Visualiser la répartition du Genre par rapport à 'Subject_Car_Colour'
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Subject_Car_Colour', hue='Gender')
plt.title('Répartition du Genre en fonction de Subject Car Colour')
plt.xlabel('Subject Car Colour')
plt.ylabel('Nombre')
plt.xticks(rotation=45)
plt.legend(title='Genre')
plt.show()

# 3. Visualiser la répartition du Genre par rapport à 'LGA_Name'
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='LGA_Name', hue='Gender')
plt.title('Répartition du Genre en fonction de LGA Name')
plt.xlabel('LGA Name')
plt.ylabel('Nombre')
plt.xticks(rotation=90)
plt.legend(title='Genre')
plt.show()

# 4. Visualiser la répartition du Genre par rapport à 'State'
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='State', hue='Gender')
plt.title('Répartition du Genre en fonction de State')
plt.xlabel('State')
plt.ylabel('Nombre')
plt.xticks(rotation=45)
plt.legend(title='Genre')
plt.show()
