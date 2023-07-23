#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import joblib

# Charger le modèle à partir du fichier enregistré
model = joblib.load('GBM_Model_date.pkl')

def make_prediction(input_data):
#Input_data = No voie, Code postal Code departement, code type local,
# nombre pieces principales, nombre de lots, surface reelle bati, surface terrain, année, mois, jour

    # Faire la prédiction
    prediction = model.predict(input_data)

    return prediction


# In[ ]:


if __name__ == "__main__":
    input_values = [float(arg) for arg in sys.argv[1:]]

    # Faire la prédiction en utilisant les valeurs d'entrée passées en paramètre
    result = make_prediction([input_values])

    # Afficher la prédiction
    print("Résultat de la prédiction :", result)

