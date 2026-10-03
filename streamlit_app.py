import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prédiction de Compatibilité Médicale · ENIAD",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).parent
MODEL_CANDIDATES = [
    ROOT_DIR / "test" / "scikit_classification_model.joblib",
    ROOT_DIR / "scikit_classification_model.joblib",
    ROOT_DIR / "test" / "best_classification_model.pkl",
]


@st.cache_resource
def load_model():
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            try:
                return joblib.load(candidate)
            except Exception:
                pass
    return None


st.title("💊 Prédictions de Compatibilité des Traitements")
st.caption("École Nationale d'Intelligence Artificielle et du Digital (ENIAD) · Service ML Médical")

st.markdown("""
Cette application clinique évalue la compatibilité d'un protocole de traitement pour un patient
en fonction de ses indicateurs biométriques, de son mode de vie et de son historique médical.
""")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Données Biométriques & Démographiques")
    age = st.number_input("Âge du patient", min_value=1, max_value=120, value=45)
    sexe = st.selectbox("Sexe biologique", options=["M", "F"])
    poids = st.number_input("Poids corporel (kg)", min_value=20.0, max_value=220.0, value=74.5)
    taille = st.number_input("Taille (cm)", min_value=80.0, max_value=240.0, value=172.0)

    # Calcul dynamique de l'IMC
    imc = poids / ((taille / 100) ** 2)
    st.metric("Indice de Masse Corporelle (IMC)", f"{imc:.1f} kg/m²")

with col_right:
    st.subheader("🩺 Mode de Vie & Antécédents")
    fumeur = st.selectbox("Statut tabagique", options=["Non", "Oui"])
    alcool = st.selectbox("Consommation d'alcool", options=["Non", "Oui"])
    activite_physique = st.selectbox(
        "Niveau d'activité physique", options=["Faible", "Moyenne", "Élevée"]
    )
    antecedents = st.selectbox(
        "Antécédents médicaux majeurs",
        options=["Aucun", "Cardiovasculaire", "Chronique", "Autre"],
    )
    allergies = st.selectbox(
        "Type d'allergies déclarées",
        options=["Aucune", "Médicamenteuse", "Autre"],
    )


def encode_categorical(data):
    activite_mapping = {"Faible": 0, "Moyenne": 1, "Élevée": 2}
    antecedents_mapping = {
        "Aucun": 0,
        "Cardiovasculaire": 1,
        "Chronique": 2,
        "Autre": 3,
    }
    allergies_mapping = {"Aucune": 0, "Médicamenteuse": 1, "Autre": 2}

    encoded = dict(data)
    encoded["Sexe"] = 1 if data["Sexe"] == "M" else 0
    encoded["Fumeur"] = 1 if data["Fumeur"] == "Oui" else 0
    encoded["Alcool"] = 1 if data["Alcool"] == "Oui" else 0
    encoded["Activité_physique"] = activite_mapping[data["Activité_physique"]]
    encoded["Antécédents_médicaux"] = antecedents_mapping[data["Antécédents_médicaux"]]
    encoded["Allergies"] = allergies_mapping[data["Allergies"]]
    return encoded


st.divider()

if st.button("🚀 Analyser la Compatibilité Clinique", type="primary", use_container_width=True):
    model = load_model()

    if model is None:
        st.error("Le modèle de classification n'a pas pu être chargé.")
    else:
        input_dict = {
            "Age": age,
            "Sexe": sexe,
            "Poids": poids,
            "Taille": taille,
            "IMC": imc,
            "Fumeur": fumeur,
            "Alcool": alcool,
            "Activité_physique": activite_physique,
            "Antécédents_médicaux": antecedents,
            "Allergies": allergies,
        }

        encoded = encode_categorical(input_dict)
        df_input = pd.DataFrame([encoded])

        try:
            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0]
            confidence = proba[pred] * 100

            st.subheader("Diagnostic de Compatibilité :")

            if pred == 1:
                st.success(f"✅ **Traitement Compatible avec le profil patient** (Niveau de confiance : {confidence:.1f}%)")
            else:
                st.error(f"❌ **Risque d'Incompatibilité Détecté** (Niveau de confiance : {confidence:.1f}%)")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Probabilité de Compatibilité", f"{proba[1]*100:.1f}%")
            with res_col2:
                st.metric("Probabilité de Contre-indication", f"{proba[0]*100:.1f}%")

        except Exception as e:
            st.error(f"Erreur d'inférence : {e}")
