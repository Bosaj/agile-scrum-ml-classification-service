import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Fonction pour charger le modèle
def load_model():
    try:
        with open('best_classification_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        # Créer un modèle par défaut si aucun n'existe
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # On peut pré-entraîner le modèle ici si nécessaire
        return model

# Titre de l'application
st.title("Prédictions de Compatibilité des Traitements")

# Description
st.write("Cette application prédit la compatibilité des traitements pour un patient en fonction de ses caractéristiques.")

# Collecte des entrées utilisateur
st.header("Entrer les informations du patient")

age = st.number_input("Âge", min_value=0, max_value=120, value=30)
sexe = st.selectbox("Sexe", options=["M", "F"])
poids = st.number_input("Poids (kg)", min_value=0.0, max_value=200.0, value=70.0)
taille = st.number_input("Taille (cm)", min_value=50.0, max_value=250.0, value=170.0)
fumeur = st.selectbox("Fumeur", options=["Oui", "Non"])
alcool = st.selectbox("Consommation d'alcool", options=["Oui", "Non"])
activite_physique = st.selectbox("Niveau d'activité physique", 
                                options=["Faible", "Moyenne", "Élevée"])
antecedents = st.selectbox("Antécédents médicaux", 
                          options=["Aucun", "Cardiovasculaire", "Chronique", "Autre"])
allergies = st.selectbox("Type d'allergies", 
                        options=["Aucune", "Médicamenteuse", "Autre"])

# Calculer l'IMC
imc = poids / ((taille / 100) ** 2)

# Fonction pour encoder les variables catégorielles
def encode_categorical(data):
    # Encodage one-hot pour les variables catégorielles
    activite_mapping = {"Faible": 0, "Moyenne": 1, "Élevée": 2}
    antecedents_mapping = {"Aucun": 0, "Cardiovasculaire": 1, "Chronique": 2, "Autre": 3}
    allergies_mapping = {"Aucune": 0, "Médicamenteuse": 1, "Autre": 2}
    
    data['Sexe'] = 1 if data['Sexe'] == "M" else 0
    data['Fumeur'] = 1 if data['Fumeur'] == "Oui" else 0
    data['Alcool'] = 1 if data['Alcool'] == "Oui" else 0
    data['Activité_physique'] = activite_mapping[data['Activité_physique']]
    data['Antécédents_médicaux'] = antecedents_mapping[data['Antécédents_médicaux']]
    data['Allergies'] = allergies_mapping[data['Allergies']]
    
    return data

# Préparer les données pour la prédiction
def prepare_data(input_values):
    input_dict = {
        'Age': age,
        'Sexe': sexe,
        'Poids': poids,
        'Taille': taille,
        'IMC': imc,
        'Fumeur': fumeur,
        'Alcool': alcool,
        'Activité_physique': activite_physique,
        'Antécédents_médicaux': antecedents,
        'Allergies': allergies
    }
    
    # Encoder les données
    encoded_dict = encode_categorical(input_dict)
    
    # Convertir en DataFrame
    return pd.DataFrame([encoded_dict])

# Bouton de prédiction
if st.button("Prédire la compatibilité"):
    # Charger le modèle
    model = load_model()
    
    # Préparer les données
    input_data = prepare_data({
        'Age': age,
        'Sexe': sexe,
        'Poids': poids,
        'Taille': taille,
        'IMC': imc,
        'Fumeur': fumeur,
        'Alcool': alcool,
        'Activité_physique': activite_physique,
        'Antécédents_médicaux': antecedents,
        'Allergies': allergies
    })
    
    try:
        # Effectuer la prédiction
        prediction = model.predict(input_data)[0]
        
        # Afficher le résultat avec un niveau de confiance simulé
        confidence = model.predict_proba(input_data)[0]
        max_confidence = max(confidence) * 100
        
        if prediction == 1:
            st.success(f"Le traitement est compatible avec le patient. (Confiance: {max_confidence:.1f}%)")
        else:
            st.error(f"Le traitement n'est PAS compatible avec le patient. (Confiance: {max_confidence:.1f}%)")
            
        # Afficher des informations supplémentaires
        st.info("""
        Facteurs pris en compte dans la prédiction:
        - Données démographiques (âge, sexe)
        - Paramètres physiques (poids, taille, IMC)
        - Mode de vie (tabagisme, alcool, activité physique)
        - Historique médical (antécédents, allergies)
        """)
        
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la prédiction: {str(e)}")
        st.warning("Veuillez vérifier que le modèle est correctement entraîné et que toutes les entrées sont valides.")