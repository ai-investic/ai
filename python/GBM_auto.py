#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import datetime
from tensorflow.keras import utils
from keras.layers import Dense, Activation, Flatten, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[2]:


data_2021 = pd.read_csv("données/valeursfoncieres-2022-s1.csv", sep="|")


# In[3]:


data_2021


# In[4]:


data_2021_c = data_2021[['Nature mutation', 'Valeur fonciere', 'No voie', 'Code postal',
                         'Code departement', 'Code type local', 'Code voie', 'Code type local', 'Surface reelle bati', 'Surface terrain']]


# In[5]:


data_2021_c.head()


# In[6]:


data_2021_c["Code departement"].unique()


# In[7]:


data_2021_c.dropna(axis=0, inplace=True)
data_2021_c= data_2021_c.fillna(0)
data_2021_c = data_2021_c[data_2021_c['Nature mutation'] == 'Vente']
data_2021_c = data_2021_c.drop(columns='Nature mutation')


# In[8]:


data_2021_c['Code postal'] = data_2021_c['Code postal'].astype(int)
data_2021_c['No voie'] = data_2021_c['No voie'].astype(int)
data_2021_c['Code postal'] = data_2021_c['Code postal'].astype(str)
data_2021_c['No voie'] = data_2021_c['No voie'].astype(str)
data_2021_c['Surface reelle bati'] = data_2021_c['Surface reelle bati'].astype(str)
data_2021_c["ID_Location"] = data_2021_c['Code voie'].astype(str) +"-"+ data_2021_c["Code postal"]+"-"+ data_2021_c["No voie"]+"-"+ data_2021_c["Surface reelle bati"]
data_2021_c['Code postal'] = data_2021_c['Code postal'].astype(float)
data_2021_c['No voie'] = data_2021_c['No voie'].astype(float)
data_2021_c['Code departement'] = data_2021_c['Code departement'].astype(str)
data_2021_c['Code departement'] = data_2021_c['Code departement'].str.replace('2A', '222')
data_2021_c['Code departement'] = data_2021_c['Code departement'].str.replace('2B', '223')
data_2021_c = data_2021_c.drop(columns='Code voie')


# In[9]:


data_2021_c["Code departement"].unique()


# In[10]:


data_2021_c['Valeur fonciere'] = data_2021_c['Valeur fonciere'].str.replace(',', '.')
data_2021_c['Valeur fonciere'] = data_2021_c['Valeur fonciere'].astype(float)
data_2021_c['Valeur fonciere'] = data_2021_c['Valeur fonciere'].astype(int)

data_2021_c['No voie'] = data_2021_c['No voie'].astype(int)
data_2021_c['Code postal'] = data_2021_c['Code postal'].astype(int)
data_2021_c['Code type local'] = data_2021_c['Code type local'].astype(int)
#data_2021_c['Surface terrain'] = data_2021_c['Surface terrain'].astype(int)
data_2021_c['Surface reelle bati'] = data_2021_c['Surface reelle bati'].str.replace(',', '.')
data_2021_c['Surface reelle bati'] = data_2021_c['Surface reelle bati'].astype(float)
data_2021_c['Surface reelle bati'] = data_2021_c['Surface reelle bati'].astype(int)


# In[11]:


data_2021_c.set_index('ID_Location', inplace=True)


# In[12]:


data_2021_c


# In[13]:


data_2021_c.sort_values(by=['ID_Location'], ascending=True, inplace=True)


# In[14]:


data_2021_c.dropna(axis=0, inplace=True)
data_2021_c= data_2021_c.fillna(0)
data_2021_c = data_2021_c.drop_duplicates()
data_2021_c


# In[15]:


# Séparer les données en variables d'entrée (X) et variable cible (y)
X = data_2021_c.drop("Valeur fonciere", axis=1)
y = data_2021_c["Valeur fonciere"]


# In[16]:


#data = data_2021_c.groupby(['ID_Location', 'Date mutation'])


# In[17]:


#print(data.head(5))


# In[ ]:





# In[18]:


# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Définir une liste d'hyperparamètres à essayer
hyperparameters = [
    {'n_estimators': 3000, 'learning_rate': 0.05, 'max_depth': 7, 'random_state': 50},
    {'n_estimators': 3500, 'learning_rate': 0.05, 'max_depth': 7, 'random_state': 50},
    {'n_estimators': 4000, 'learning_rate': 0.05, 'max_depth': 7, 'random_state': 50},
    {'n_estimators': 4500, 'learning_rate': 0.05, 'max_depth': 7, 'random_state': 50},
    {'n_estimators': 5000, 'learning_rate': 0.05, 'max_depth': 7, 'random_state': 50},
    {'n_estimators': 5500, 'learning_rate': 0.05, 'max_depth': 7, 'random_state': 50},
    {'n_estimators': 6000, 'learning_rate': 0.05, 'max_depth': 7, 'random_state': 50}
]

best_rmse = float('inf')
best_model = None


# In[20]:


# Boucle sur les hyperparamètres
for params in hyperparameters:
    # Créer et entraîner le modèle GBM avec les hyperparamètres actuels
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calculer la racine carrée de l'erreur quadratique moyenne (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Vérifier si le modèle actuel est le meilleur jusqu'à présent
    if rmse < best_rmse:
        best_rmse = rmse
        best_mae = mae
        best_r2 = r2
        best_model = model

    # Afficher les résultats de l'itération actuelle
    print("Hyperparameters:", params)
    print("RMSE:", rmse)
    # Calcul de la MAE
    print("Mean Absolute Error (MAE):", mae)
    # Calcul du R²
    print("R-squared (R²):", r2)
    print()


# In[21]:


# Afficher les meilleurs résultats
print("Meilleur modèle:")
print("Hyperparameters:", best_model.get_params())
print("RMSE:", best_rmse)
print("Mean Absolute Error (MAE):", best_mae)
print("R-squared (R²):", best_r2)


# In[ ]:




