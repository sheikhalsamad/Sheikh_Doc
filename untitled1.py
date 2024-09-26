# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:01:03 2024

@author: cheik
"""

# Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pmdarima as pm

# Chargement de la base de données (ajuste le chemin du fichier si nécessaire)
try:
    df = pd.read_excel(r'C:/Users/cheik/Downloads/MP.xlsx', sheet_name='monkeypox')
except FileNotFoundError:
    print("Le fichier spécifié est introuvable. Veuillez vérifier le chemin.")

# Vérification des données
if df is not None:
    print(df.head())

    # Statistiques descriptives générales
    print(df.describe())

    # Vérifier les colonnes pour éviter les erreurs
    if 'date' in df.columns and 'total_cases' in df.columns:
        
        # Visualisation de l'évolution des cas au fil du temps
        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(df['date']), df['total_cases'], label='Total Cases')
        plt.xlabel('Date')
        plt.ylabel('Total Cases')
        plt.title('Évolution des cas de Monkeypox au fil du temps')
        plt.legend()
        plt.show()

       # Vérifie et ordonne les dates
df_arima = df[['date', 'total_cases']].dropna()

# Convertir la colonne 'date' en datetime
df_arima['date'] = pd.to_datetime(df_arima['date'])

# Tri des dates pour garantir l'ordre croissant
df_arima = df_arima.sort_values(by='date')

# Vérifier les doublons dans les dates
duplicate_dates = df_arima[df_arima.duplicated(subset='date', keep=False)]
if not duplicate_dates.empty:
    print("Doublons trouvés dans les dates :")
    print(duplicate_dates)
    
    # Si vous trouvez des doublons, vous devez les gérer. Par exemple, vous pouvez agréger les données.
    df_arima = df_arima.groupby('date').agg({'total_cases': 'mean'}).reset_index()

# Définir la date comme index
df_arima.set_index('date', inplace=True)

# Ajout explicite de la fréquence journalière et gestion des dates manquantes
df_arima = df_arima.asfreq('D', method='pad')  # Utilise la dernière valeur pour remplir les trous dans les dates

# Division en train et test (80% train, 20% test)
train = df_arima[:int(0.8 * len(df_arima))]
test = df_arima[int(0.8 * len(df_arima)):]

# Modélisation ARIMA avec les données réindexées
model = ARIMA(train['total_cases'], order=(5, 1, 0))
arima_model = model.fit()

# Prévision sur la période de test
forecast = arima_model.forecast(steps=len(test))

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['total_cases'], label='Entraînement')
plt.plot(test.index, test['total_cases'], label='Test', color='orange')
plt.plot(test.index, forecast, label='Prévisions ARIMA', color='green')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.title('Prédiction ARIMA des cas futurs')
plt.legend()
plt.show()

# Calcul de l'erreur quadratique moyenne
mse = mean_squared_error(test['total_cases'], forecast)
print(f"Mean Squared Error: {mse}")
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ajustement du modèle SARIMA
# SARIMA(p, d, q)(P, D, Q, s)
# Ici on suppose une périodicité saisonnière de 12 mois (s=12), à ajuster selon les données
model = SARIMAX(train['total_cases'], 
                order=(1, 1, 1),  # p, d, q : Non-saisonnier
                seasonal_order=(1, 1, 1, 12))  # P, D, Q, s : Saisonnier
sarima_model = model.fit()

# Faire des prévisions
forecast = sarima_model.forecast(steps=len(test))

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['total_cases'], label='Entraînement')
plt.plot(test.index, test['total_cases'], label='Test', color='orange')
plt.plot(test.index, forecast, label='Prévisions SARIMA', color='green')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.title('Prédiction SARIMA des cas futurs')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error

# Vérification que les tailles des prévisions et des données de test correspondent
if len(test) == len(forecast):
    # Calcul de l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(test['total_cases'], forecast)
    
    # Calcul de l'erreur quadratique moyenne racine (RMSE)
    rmse = np.sqrt(mse)
    
    # Calcul de l'erreur absolue moyenne (MAE)
    mae = mean_absolute_error(test['total_cases'], forecast)

    # Affichage des résultats arrondis à 4 chiffres significatifs
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
else:
    print("Erreur : Les tailles des données de test et des prévisions ne correspondent pas.")

from statsmodels.graphics.tsaplots import plot_acf

# Vérifier la saisonnalité en visualisant l'Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(train['total_cases'], lags=24)
plt.title('Fonction d\'autocorrélation (ACF) des cas de Monkeypox')
plt.show()

# Appliquer la différenciation (d=1)
df_diff = train['total_cases'].diff().dropna()

# Vérifier l'autocorrélation après différenciation
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(10, 6))
plot_acf(df_diff, lags=24)
plt.title("Autocorrelation Function (ACF) après différenciation (d=1)")
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ajustement du modèle SARIMA (p, d, q) (P, D, Q, s) avec une saisonnalité de 6 mois
model_sarima = SARIMAX(train['total_cases'], 
                       order=(1, 1, 1),  # Composant non-saisonnier
                       seasonal_order=(1, 1, 1, 6))  # Composant saisonnier avec s=6
sarima_model = model_sarima.fit()

# Faire des prévisions
forecast_sarima = sarima_model.forecast(steps=len(test))

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['total_cases'], label='Entraînement')
plt.plot(test.index, test['total_cases'], label='Test', color='orange')
plt.plot(test.index, forecast_sarima, label='Prévisions SARIMA', color='green')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.title('Prédiction SARIMA des cas futurs avec saisonnalité de 6 mois')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Charger les données
# On suppose que df contient une colonne 'total_cases' et une colonne 'date'
# Si ce n'est pas le cas, assure-toi que 'df' est correctement défini
# Ex: df = pd.read_csv("your_data.csv") ou df = pd.read_excel("your_data.xlsx")

# Création de décalages (lags)
df_lag = df.copy()
df_lag['lag1'] = df_lag['total_cases'].shift(1)
df_lag['lag2'] = df_lag['total_cases'].shift(2)
df_lag['lag3'] = df_lag['total_cases'].shift(3)
df_lag = df_lag.dropna()  # Retirer les lignes avec des valeurs manquantes causées par les décalages

# Séparer les variables explicatives (X) et la variable cible (y)
X = df_lag[['lag1', 'lag2', 'lag3']]
y = df_lag['total_cases']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

# Prédictions
rf_predictions = rf_model.predict(X_test)

# Vérification que les indices des prédictions sont alignés avec ceux des données de test
rf_predictions_series = pd.Series(rf_predictions, index=y_test.index)

# Calcul du MSE
mse_rf = mean_squared_error(y_test, rf_predictions)
print(f"MSE pour Random Forest: {mse_rf}")

# Visualisation des prévisions avec des dates correctes
plt.figure(figsize=(10, 6))

# Tracer les données réelles
plt.plot(y_test.index, y_test, label='Données réelles', color='blue')

# Tracer les prévisions avec les indices de test (dates alignées)
plt.plot(rf_predictions_series.index, rf_predictions_series, label='Prédictions Random Forest', color='red', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.title('Prévisions Random Forest vs Données réelles')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
conda install tensorflow
import tensorflow as tf
print(tf.__version__)

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['total_cases']])

# Fonction pour créer des séquences de données pour LSTM
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Paramètres de séquence
sequence_length = 10

# Créer les séquences pour LSTM
X, y = create_sequences(scaled_data, sequence_length)

# Diviser les données en train et test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape des données pour le modèle LSTM [samples, time_steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Construction du modèle LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Prédictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Inverser la normalisation

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Données réelles', color='blue')
plt.plot(df.index[-len(predictions):], predictions, label='Prédictions LSTM', color='red', linestyle='--')
plt.xlabel('Date

