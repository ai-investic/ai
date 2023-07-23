#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import joblib
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


data_2017 = pd.read_csv("données/valeursfoncieres-2017-s2.csv", sep="|")
data_2018 = pd.read_csv("données/valeursfoncieres-2018.csv", sep="|")
data_2019 = pd.read_csv("données/valeursfoncieres-2019.csv", sep="|")
data_2020 = pd.read_csv("données/valeursfoncieres-2020.csv", sep="|")
data_2021 = pd.read_csv("données/valeursfoncieres-2021.csv", sep="|")
data_2022 = pd.read_csv("données/valeursfoncieres-2022-s1.csv", sep="|")

dataframes = []


# In[3]:


dataframes.append(data_2017)
dataframes.append(data_2018)
dataframes.append(data_2019)
dataframes.append(data_2020)
dataframes.append(data_2021)
dataframes.append(data_2022)


# In[4]:


data = pd.concat(dataframes, ignore_index=True)


# In[5]:


data


# In[6]:


data_c = data[['Nature mutation', 'Valeur fonciere', 'No voie', 'Code postal',
                         'Code departement', 'Code type local', 'Code voie', 'Nombre pieces principales', 'Nombre de lots', 'Date mutation', 'Surface reelle bati', 'Surface terrain']]


# In[7]:


data_c.head()


# In[8]:


data_c["Code departement"].unique()


# In[9]:


data_c.dropna(axis=0, inplace=True)
data_c= data_c.fillna(0)
data_c = data_c[data_c['Nature mutation'] == 'Vente']
data_c = data_c.drop(columns='Nature mutation')


# In[10]:


data_c['Code postal'] = data_c['Code postal'].astype(int)
data_c['No voie'] = data_c['No voie'].astype(int)
data_c['Code postal'] = data_c['Code postal'].astype(str)
data_c['No voie'] = data_c['No voie'].astype(str)
data_c['Surface reelle bati'] = data_c['Surface reelle bati'].astype(str)
data_c["ID_Location"] = data_c['Code voie'].astype(str) +"-"+ data_c["Code postal"]+"-"+ data_c["No voie"]+"-"+ data_c["Surface reelle bati"]
data_c['Code postal'] = data_c['Code postal'].astype(float)
data_c['No voie'] = data_c['No voie'].astype(float)
data_c['Code departement'] = data_c['Code departement'].astype(str)
data_c['Code departement'] = data_c['Code departement'].str.replace('2A', '222')
data_c['Code departement'] = data_c['Code departement'].str.replace('2B', '223')
data_c = data_c.drop(columns='Code voie')


# In[11]:


data_c["Code departement"].unique()


# In[12]:


data_c['Nombre pieces principales'] = data_c['Nombre pieces principales'].astype(int)
data_c['Nombre de lots'] = data_c['Nombre de lots'].astype(int)
data_c['Valeur fonciere'] = data_c['Valeur fonciere'].str.replace(',', '.')
data_c['Valeur fonciere'] = data_c['Valeur fonciere'].astype(float)
data_c['Valeur fonciere'] = data_c['Valeur fonciere'].astype(int)
data_c = data_c[data_c['Valeur fonciere'] < 5000000]
data_c['Code type local'] = data_c['Code type local'].astype(int)
data_c['No voie'] = data_c['No voie'].astype(int)
data_c['Code postal'] = data_c['Code postal'].astype(int)
data_c['Code type local'] = data_c['Code type local'].astype(int)
#data_2021_c['Surface terrain'] = data_2021_c['Surface terrain'].astype(int)
data_c['Surface reelle bati'] = data_c['Surface reelle bati'].str.replace(',', '.')
data_c['Surface reelle bati'] = data_c['Surface reelle bati'].astype(float)
data_c['Surface reelle bati'] = data_c['Surface reelle bati'].astype(int)
data_c['Date mutation'] = pd.to_datetime(data_c['Date mutation'], format='%d/%m/%Y')


# In[13]:


data_c['Year'] = data_c['Date mutation'].dt.year
data_c['Month'] = data_c['Date mutation'].dt.month
data_c['Day'] = data_c['Date mutation'].dt.day
data_c.drop('Date mutation', axis=1, inplace=True)


# In[14]:


data_c.set_index('ID_Location', inplace=True)


# In[15]:


data_c


# In[16]:


data_c.sort_values(by=['ID_Location'], ascending=True, inplace=True)


# In[17]:


data_c.dropna(axis=0, inplace=True)
data_c= data_c.fillna(0)
data_c = data_c.drop_duplicates()
data_c


# In[18]:


# Séparer les données en variables d'entrée (X) et variable cible (y)
X = data_c.drop("Valeur fonciere", axis=1)
y = data_c["Valeur fonciere"]


# In[19]:


#data = data_2021_c.groupby(['ID_Location', 'Date mutation'])


# In[20]:


#print(data.head(5))


# In[ ]:





# In[21]:


# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


#len(train_generator)


# In[23]:


# Créer et entraîner le modèle GBM avec des hyperparamètres personnalisés
model = GradientBoostingRegressor(
    n_estimators=9000,  # Nombre d'estimateurs (arbres) dans l'ensemble
    learning_rate=0.05,  # Taux d'apprentissage
    max_depth=7,  # Profondeur maximale des arbres
    random_state=50  # Graine aléatoire pour la reproductibilité
)


# In[24]:


# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)


# In[25]:


y_pred = model.predict(X_test)


# In[26]:


# Calculer l'erreur quadratique moyenne (RMSE) sur l'ensemble de test
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)


# In[27]:


print(y_pred)


# In[29]:


# Enregistrer le modèle
joblib.dump(model, 'GBM_Model_date.pkl')


# In[ ]:




