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



# Charger vos données
@st.cache_data
def load_data():
    data = pd.read_excel("./DB/All-at-once_DB.xlsx")
    data["Absolute Uncertainty"] = (data["1s uncertainty"] / 100) * data["Dose (Gy)"]
    data['Filter Combo'] = data.apply(lambda row: f"{row['Particle']}_{row['Screen']}_{row['Code']}_{row['Case']}_{row['Thickness (cm)']}", axis=1)
    return data

data = load_data()

# Sélection des paramètres
st.header("Sélection des Paramètres")

# Créer deux colonnes pour les filtres
col1, col2 = st.columns(2)

# Filtres pour le cas de référence
with col1:
    st.subheader("Cas de Référence")
    ref_particle = st.selectbox('Particule (Référence):', options=list(data['Particle'].unique()))
    ref_screen = st.selectbox('Écran (Référence):', options=list(data['Screen'].unique()))
    ref_code = st.selectbox('Code (Référence):', options=list(data['Code'].unique()))
    ref_case = st.selectbox('Cas (Référence):', options=list(data['Case'].unique()))
    ref_thickness = st.selectbox('Épaisseur (Référence):', options=list(data['Thickness (cm)'].unique()))

# Filtres pour les séries à comparer
with col2:
    st.subheader("Séries à Comparer")
    compare_particles = st.multiselect('Particule:', options=list(data['Particle'].unique()), default=list(data['Particle'].unique())[0])
    compare_screens = st.multiselect('Écran:', options=list(data['Screen'].unique()), default=list(data['Screen'].unique())[0])
    compare_codes = st.multiselect('Code:', options=list(data['Code'].unique()), default=list(data['Code'].unique())[0])
    compare_cases = st.multiselect('Cas:', options=list(data['Case'].unique()), default=list(data['Case'].unique())[0])
    compare_thicknesses = st.multiselect('Épaisseur:', options=list(data['Thickness (cm)'].unique()), default=list(data['Thickness (cm)'].unique())[0])
# Préparation des données
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

# Fusionner les données de référence et les données comparatives
combined_data = pd.concat([ref_data, compare_data]).drop_duplicates()

# Création des onglets
tab1, tab2, tab3 = st.tabs(["🔢 Tableau des Données", "📈 Graphe des Doses", "Graphe des Ratios"])

with tab1:
    st.subheader("Données Combinées du Cas de Référence et Séries Sélectionnées")
    st.dataframe(combined_data)


with tab2:
    st.subheader("Graphe des Doses")
    fig1 = go.Figure()

    # Nom pour la légende du cas de référence
    ref_label = f"{ref_particle}_{ref_screen}_{ref_code}_{ref_case}_{ref_thickness} (Référence)"

    # Ajout du cas de référence avec une étiquette personnalisée
    fig1.add_trace(go.Scatter(x=ref_data["Distance (m)"], y=ref_data["Dose (Gy)"],
                              mode='lines+markers', name=ref_label))

    # Ajout des séries à comparer
    for combo in compare_data['Filter Combo'].unique():
        df_subset = compare_data[compare_data['Filter Combo'] == combo]
        fig1.add_trace(go.Scatter(x=df_subset["Distance (m)"], 
                                  y=df_subset["Dose (Gy)"],
                                  mode='lines+markers', 
                                  marker_symbol='star-diamond-dot',
                                  line=dict(dash='dash'),
                                  name=combo,
                                  error_y=dict(type='data', array=df_subset["Absolute Uncertainty"], visible=True)
                                  ))

    fig1.update_layout(
        # xaxis_title="Distance (m) [Log10]",
        # yaxis_title="Dose (Gy) [Log10]",
        # xaxis={'type': 'log', 'showgrid':True},
        # yaxis={'type': 'log', 'showgrid':True},
        # height=600, width=900,
        title="Dose en fonction de la Distance (échelles logarithmiques)",
        legend_title="Combinaison des Filtres",
        legend=dict(
            orientation="h",  # Horizontal
            xanchor="center",  # Ancre au centre
            x=0.5,  # Positionner au centre en x
            y=-0.3  # Position en dessous du graphique
        ))
    fig1.update_xaxes(type='log', showgrid=True, title="Distance (m) [Log]")
    fig1.update_yaxes(type='log', showgrid=True, title="Dose (Gy)[Log]", tickformat='.2e' )
    fig1.update_xaxes(minor = dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))

    st.plotly_chart(fig1, use_container_width=True)


with tab3:
    st.subheader("Graphe des Ratios")
    fig2 = go.Figure()
    for combo in compare_data['Filter Combo'].unique():
        df_subset = compare_data[compare_data['Filter Combo'] == combo]
        df_subset = df_subset.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
        df_subset['Dose Ratio'] = df_subset['Dose (Gy)'] / df_subset['Dose (Gy)_ref']
        fig2.add_trace(go.Scatter(x=df_subset["Distance (m)"], y=df_subset["Dose Ratio"],
                                  mode='lines+markers', name=combo))
    fig2.update_layout(xaxis_title="Distance (m) [Log]", yaxis_title="Ratio des Doses",
                       xaxis={'type': 'log'})
    st.plotly_chart(fig2)
