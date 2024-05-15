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
@st.cache
def load_data():
    data = pd.read_excel("./DB/All-at-once_DB.xlsx")
    data["Absolute Uncertainty"] = (data["1s uncertainty"] / 100) * data["Dose (Gy)"]
    data['Filter Combo'] = data.apply(lambda row: f"{row['Particle']}_{row['Screen']}_{row['Code']}_{row['Case']}_{row['Thickness (cm)']}", axis=1)
    return data

data = load_data()

# Widgets pour la sélection des filtres
particle_types = st.multiselect('Particule:', options=list(data['Particle'].unique()), default=list(data['Particle'].unique())[0])
screen_types = st.multiselect('Écran:', options=list(data['Screen'].unique()), default=list(data['Screen'].unique())[0])
code_types = st.multiselect('Code:', options=list(data['Code'].unique()), default=list(data['Code'].unique())[0])
case_types = st.multiselect('Cas:', options=list(data['Case'].unique()), default=list(data['Case'].unique())[0])
thickness_types = st.multiselect('Épaisseur:', options=list(data['Thickness (cm)'].unique()), default=list(data['Thickness (cm)'].unique())[0])

# Filtrage des données et création du graphique
filtered_data = data[data['Particle'].isin(particle_types) & 
                     data['Screen'].isin(screen_types) &
                     data['Code'].isin(code_types) &
                     data['Case'].isin(case_types) &
                     data['Thickness (cm)'].isin(thickness_types)]

fig = go.Figure()
for combo in filtered_data['Filter Combo'].unique():
    df_subset = filtered_data[filtered_data['Filter Combo'] == combo]
    fig.add_trace(go.Scatter(x=df_subset["Distance (m)"], y=df_subset["Dose (Gy)"],
                             mode='lines+markers', name=combo,
                             line=dict(dash='dash'),
                             error_y=dict(type='data', array=df_subset["Absolute Uncertainty"],
                                          visible=True)))

fig.update_layout(
    title="Dose en fonction de la Distance (échelles logarithmiques)",
    xaxis_title="Distance (m) [Log]",
    yaxis_title="Dose (Gy) [Log]",
    legend_title="Combinaison des Filtres",
    xaxis={'type': 'log'},
    yaxis={'type': 'log'}
)

st.plotly_chart(fig)
