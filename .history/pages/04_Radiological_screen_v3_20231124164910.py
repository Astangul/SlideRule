import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openpyxl import load_workbook
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(
    page_title="Slide-Rule",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)




st.title("Slide Rule")

st.write(
    """My first Streamlit app
    """
)



# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_excel("./DB/All-at-once_DB.xlsx")
    data["Absolute Uncertainty"] = (data["1s uncertainty"] / 100) * data["Dose (Gy)"]
    data['Filter Combo'] = data.apply(lambda row: f"{row['Particle']}_{row['Screen']}_{row['Code']}_{row['Case']}_{row['Thickness (cm)']}", axis=1)
    return data

data = load_data()

# Sélection du cas de référence
st.header("Sélectionner un cas de référence")
ref_particle = st.selectbox('Particule (Référence):', options=list(data['Particle'].unique()))
ref_screen = st.selectbox('Écran (Référence):', options=list(data['Screen'].unique()))
ref_code = st.selectbox('Code (Référence):', options=list(data['Code'].unique()))
ref_case = st.selectbox('Cas (Référence):', options=list(data['Case'].unique()))
ref_thickness = st.selectbox('Épaisseur (Référence):', options=list(data['Thickness (cm)'].unique()))

# Sélection des séries à comparer
st.header("Sélectionner des séries à comparer avec le cas de référence")
compare_particles = st.multiselect('Particule:', options=list(data['Particle'].unique()), default=list(data['Particle'].unique())[0])
compare_screens = st.multiselect('Écran:', options=list(data['Screen'].unique()), default=list(data['Screen'].unique())[0])
compare_codes = st.multiselect('Code:', options=list(data['Code'].unique()), default=list(data['Code'].unique())[0])
compare_cases = st.multiselect('Cas:', options=list(data['Case'].unique()), default=list(data['Case'].unique())[0])
compare_thicknesses = st.multiselect('Épaisseur:', options=list(data['Thickness (cm)'].unique()), default=list(data['Thickness (cm)'].unique())[0])

# Filtrage des données pour le cas de référence et les séries à comparer
ref_data = data[(data['Particle'] == ref_particle) & 
                (data['Screen'] == ref_screen) &
                (data['Code'] == ref_code) &
                (data['Case'] == ref_case) &
                (data['Thickness (cm)'] == ref_thickness)]

compare_data = data[data['Particle'].isin(compare_particles) & 
                    data['Screen'].isin(compare_screens) &
                    data['Code'].isin(compare_codes) &
                    data['Case'].isin(compare_cases) &
                    data['Thickness (cm)'].isin(compare_thicknesses)]

# Création du graphique
fig = go.Figure()

# Ajout du cas de référence
fig.add_trace(go.Scatter(x=ref_data["Distance (m)"], y=ref_data["Dose (Gy)"],
                         mode='lines+markers', name="Référence",
                         line=dict(dash='dash')))

# Ajout des séries à comparer et calcul du ratio
for combo in compare_data['Filter Combo'].unique():
    df_subset = compare_data[compare_data['Filter Combo'] == combo]
    # Calcul du ratio des doses
    df_subset = df_subset.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
    df_subset['Dose Ratio'] = df_subset['Dose (Gy)'] / df_subset['Dose (Gy)_ref']
    
    fig.add_trace(go.Scatter(x=df_subset["Distance (m)"], y=df_subset["Dose Ratio"],
                             mode='lines+markers', name=combo))

fig.update_layout(
    title="Ratio des Doses par Rapport au Cas de Référence",
    xaxis_title="Distance (m) [Log]",
    yaxis_title="Ratio des Doses",
    xaxis={'type': 'log'}
)

st.plotly_chart(fig)
