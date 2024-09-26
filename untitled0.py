# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:32:14 2024

@author: cheik
"""

# Importer les bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from scipy import stats
import statsmodels.api as sm

# Chemin corrigé (copié manuellement depuis l'explorateur)
file_path = r'C:/Users/cheik/Downloads/StudentPerformanceFactors.csv'

# Lecture du fichier CSV
df = pd.read_csv(file_path)

# Afficher les premières lignes pour vérifier le chargement
df.head()

# Statistiques descriptives
print(df.describe())

# Visualisation de la répartition des variables principales
plt.figure(figsize=(12, 6))
sns.histplot(df['Hours_Studied'], kde=True, bins=10, color='blue')
plt.title('Distribution des heures étudiées')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['Sleep_Hours'], kde=True, bins=10, color='green')
plt.title('Distribution des heures de sommeil')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Parental_Involvement', y='Exam_Score', data=df)
plt.title('Répartition des notes en fonction de l\'implication des parents')
plt.show()

# Matrice de corrélation
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice de corrélation')
plt.show()

# Corrélation entre heures étudiées et note finale
corr_hours_exam = df['Hours_Studied'].corr(df['Exam_Score'])
print(f"Corrélation entre heures étudiées et note finale : {corr_hours_exam}")

# Régression linéaire pour prédire les notes en fonction des heures d'étude, du sommeil, etc.
X = df[['Hours_Studied', 'Sleep_Hours', 'Attendance']]
y = df['Exam_Score']

# Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle de régression linéaire
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Prédictions
y_pred = lin_reg.predict(X_test)

# Résultats
print(f"Coefficient de régression : {lin_reg.coef_}")
print(f"Intercept : {lin_reg.intercept_}")


# Création d'une variable binaire pour classifier réussite/échec
df['Passed'] = df['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)

# Séparation des variables explicatives et de la variable cible
X_class = df[['Hours_Studied', 'Sleep_Hours', 'Attendance', 'Parental_Involvement']]
y_class = df['Passed']

# Conversion des variables catégorielles en numériques avec get_dummies
X_class = pd.get_dummies(X_class, drop_first=True)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

# Modèle de régression logistique
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Prédictions
y_pred_class = log_reg.predict(X_test)

# Rapport de classification
print(classification_report(y_test, y_pred_class))

# Interactions entre les heures d'étude et la participation des parents
plt.figure(figsize=(10, 6))
sns.lmplot(x='Hours_Studied', y='Exam_Score', hue='Parental_Involvement', data=df)
plt.title('Interaction entre heures d\'études et implication parentale sur la note finale')
plt.show()

# Appliquer K-Means pour segmenter les étudiants
X_cluster = df[['Hours_Studied', 'Sleep_Hours', 'Attendance']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hours_Studied', y='Exam_Score', hue='Cluster', palette='viridis', data=df)
plt.title('Clustering des étudiants en fonction des heures étudiées et des scores')
plt.show()

# Analyse ANOVA pour vérifier si la note varie significativement selon l'accès à Internet
anova_result = stats.f_oneway(df[df['Internet_Access'] == 'Yes']['Exam_Score'],
                              df[df['Internet_Access'] == 'No']['Exam_Score'])

print(f"Résultat de l'ANOVA : F-statistique = {anova_result.statistic}, p-value = {anova_result.pvalue}")


# Conversion des variables catégorielles en numériques avec get_dummies
X_class = pd.get_dummies(X_class, drop_first=True)

# Normaliser les données
scaler = StandardScaler()
X_class = scaler.fit_transform(X_class)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)



