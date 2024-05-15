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

# # Methode 1
# # Ajouter un slider pour le nombre de fissions
# fissions = st.slider("Nombre de fissions", 
#                      min_value=1.0e16, 
#                      max_value=1.0e20, 
#                      value=1.0e17, 
#                      step=1.0e16, 
#                      format='%e')

# # Methode 2
# # Afficher la plage des valeurs du slider en notation scientifique
# st.write("Nombre de fissions (notation scientifique) : de 1E16 à 1E20")

# # Définir le slider pour sélectionner l'exposant
# exposant = st.slider("Sélectionnez l'exposant pour le nombre de fissions :", min_value=16, max_value=20, value=17, step=1)

# # Calculer le nombre de fissions basé sur l'exposant
# fissions = 10 ** exposant

# # Afficher la valeur sélectionnée en notation scientifique
# st.write(f"Nombre de fissions sélectionné: {fissions:.1e}")

# Slider pour les multiples de 1E15 fissions
# Par exemple, une valeur de 550 représente 5.5E17
slider_value = st.slider("Nombre de fissions (x1E15):", min_value=1, max_value=2000, value=1100, step=1, format='%e')

# Convertir la valeur du slider en fissions réelles
fissions = slider_value * 1e15

# Affichage de la valeur sélectionnée en notation scientifique
st.write(f"Nombre de fissions sélectionné: {fissions:.1e} fissions")

# Calculer le facteur de multiplication des doses
dose_multiplier = fissions / 1e17

# Charger vos données
@st.cache_data
def load_data():
    data = pd.read_excel("./DB/All-at-once_DB.xlsx")
    data["Absolute Uncertainty"] = (data["1s uncertainty"]) * data["Dose (Gy)"]
    data['Filter Combo'] = data.apply(lambda row: f"{row['Particle']}_{row['Screen']}_{row['Code']}_{row['Case']}_{row['Thickness (cm)']}", axis=1)
    return data


data = load_data()


# Définir une liste de couleurs
# colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

def hex_to_rgba(hex_color, alpha=0.3):
    """Convertir une couleur hexadécimale en une couleur rgba avec transparence."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return 'rgba(' + ', '.join(str(int(hex_color[i:i + lv // 3], 16)) for i in range(0, lv, lv // 3)) + f', {alpha})'



# Sélection des paramètres
st.header("Sélection des Paramètres")


with st.expander("↳", expanded=True):
    # Créer deux colonnes pour les filtres
    col1, col2 = st.columns(2)
    
    # Filtres pour les séries à comparer
    with col1:
        st.subheader("Séries à Comparer (multiple choice)")
        compare_cases = st.multiselect('Case:', options=list(data['Case'].unique()), default=list(data['Case'].unique())[0])
        compare_codes = st.multiselect('Code:', options=list(data['Code'].unique()), default=list(data['Code'].unique())[0])
        compare_particles = st.multiselect('Particle:', options=list(data['Particle'].unique()), default=list(data['Particle'].unique())[0])
        compare_screens = st.multiselect('Screen:', options=list(data['Screen'].unique()), default=list(data['Screen'].unique())[0])
        compare_thicknesses = st.multiselect('Thickness (cm):', options=list(data['Thickness (cm)'].unique()), default=list(data['Thickness (cm)'].unique())[0])
    
    # Filtres pour le cas de référence
    with col2:
        st.subheader("Reference case")
        ref_case = st.selectbox('Case:', options=list(data['Case'].unique()))
        ref_code = st.selectbox('Code:', options=list(data['Code'].unique()))
        ref_particle = st.selectbox('Particle:', options=list(data['Particle'].unique()))
        ref_screen = st.selectbox('Screen:', options=list(data['Screen'].unique()))
        ref_thickness = st.selectbox('Thickness (cm):', options=list(data['Thickness (cm)'].unique()))


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

# Exclure le cas de référence des données de comparaison
compare_data = compare_data[~((compare_data['Particle'] == ref_particle) & 
                              (compare_data['Screen'] == ref_screen) &
                              (compare_data['Code'] == ref_code) &
                              (compare_data['Case'] == ref_case) &
                              (compare_data['Thickness (cm)'] == ref_thickness))]

# Fusionner les données de référence et les données comparatives
combined_data = pd.concat([ref_data, compare_data]).drop_duplicates()
# Formatage des colonnes spécifiques
formatted_data = combined_data.style.format({
    "1s uncertainty": "{:.2%}",  # Format en pourcentage avec 2 décimales
    "Dose (Gy)": "{:.2e}",       # Format scientifique avec 2 décimales
    "Absolute Uncertainty": "{:.2e}"  # Format scientifique avec 2 décimales
})
# Création des onglets
tab1, tab2, tab3 = st.tabs(["📈 Graph", "🆚 Comparison", "🔢 Dataframe"])



with tab3:
    # st.subheader("Données Combinées du Cas de Référence et Séries Sélectionnées")
    st.dataframe(formatted_data)


with tab1:
    # st.subheader("Graphe des Doses")
    
    log_x = st.checkbox("Échelle logarithmique pour l'axe X", value=True)
    log_y = st.checkbox("Échelle logarithmique pour l'axe Y", value=True)
    fig1 = go.Figure()

    # Nom pour la légende du cas de référence
    ref_label = f"{ref_particle}_{ref_screen}_{ref_code}_{ref_case}_{ref_thickness} (reference)"

    # Ajout du cas de référence avec une étiquette personnalisée
    fig1.add_trace(go.Scatter(x=ref_data["Distance (m)"],
                              y=ref_data["Dose (Gy)"],
                              mode='lines+markers', 
                              name=ref_label,
                              error_y=dict(type='data', array=2*ref_data["Absolute Uncertainty"], visible=True)
                              ))

    # Ajout des séries à comparer
    for combo in compare_data['Filter Combo'].unique():
        df_subset = compare_data[compare_data['Filter Combo'] == combo]
        fig1.add_trace(go.Scatter(x=df_subset["Distance (m)"], 
                                  y=df_subset["Dose (Gy)"],
                                  mode='lines+markers', 
                                  marker_symbol='circle-dot', marker_size=8,
                                  line=dict(dash='dash'),
                                  name=combo,
                                  error_y=dict(type='data', array=2*df_subset["Absolute Uncertainty"], visible=True)
                                  ))

    fig1.update_layout(
        hovermode='x',
        showlegend=True,
        xaxis={'showgrid':True},
        yaxis={'showgrid':True},
        height=700,  #width=900,
        # title="Dose en fonction de la Distance (échelles logarithmiques)",
        legend_title="Click on legends below to hide/show:",
        legend=dict(
            orientation="h",  # Horizontal
            xanchor="center",  # Ancre au centre
            x=0.5,  # Positionner au centre en x
            y=-0.3  # Position en dessous du graphique
        ))
    fig1.update_xaxes(minor = dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))
    fig1.update_yaxes(minor = dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))

# Mise à jour conditionnelle des axes
    if log_x:
        fig1.update_xaxes(type='log', title="Distance (m) [Log10]")
    else:
        fig1.update_xaxes(type='linear', title="Distance (m)")
    
    if log_y:
        fig1.update_yaxes(type='log', title="Dose (Gy) ± 2σ [Log10]", tickformat='.2e')
    else:
        fig1.update_yaxes(type='linear', title="Dose (Gy) ± 2σ",  tickformat='.2e')

    #st.plotly_chart(fig1)
    st.plotly_chart(fig1, use_container_width=True) #le graphe occupe toute la page


with tab2:
    #st.subheader("Graphe des Ratios")
    fig2 = go.Figure()
    for index, combo in enumerate(compare_data['Filter Combo'].unique()):
        # Choisissez une couleur du tableau des couleurs
        color = colors[index % len(colors)]
        rgba_color = hex_to_rgba(color, alpha=0.3)
                       
        df_subset = compare_data[compare_data['Filter Combo'] == combo]
        df_subset = df_subset.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
        df_subset['Dose Ratio'] = df_subset['Dose (Gy)'] / df_subset['Dose (Gy)_ref']
         # Calcul de la nouvelle colonne 'Combined Uncertainty'
        df_subset['Combined Uncertainty'] = np.sqrt(
            np.square(df_subset['1s uncertainty']) + 
            np.square(df_subset['1s uncertainty_ref'])
        )
        # Calcul de l'incertitude absolue
        df_subset['Absolute Combined Uncertainty'] = df_subset['Combined Uncertainty'] * df_subset['Dose Ratio']
        
        #___________
        # Calcul des limites supérieure et inférieure pour les bandes d'erreur
        upper_bound = df_subset['Dose Ratio'] + 1*df_subset['Absolute Combined Uncertainty']
        lower_bound = df_subset['Dose Ratio'] - 1*df_subset['Absolute Combined Uncertainty']

         # Tracé de la ligne principale
        fig2.add_trace(go.Scatter(
            x=df_subset["Distance (m)"], 
            y=df_subset["Dose Ratio"],
            error_y=dict(type='data', array=1*df_subset['Absolute Combined Uncertainty'], visible=True),
            mode='lines+markers',
            marker_symbol='circle-open-dot', marker_size=8,
            line=dict(dash='dash', color=color),
            name=combo
        ))
        
        # Tracé des bandes d'erreur
        fig2.add_trace(go.Scatter(
            x=df_subset["Distance (m)"], 
            y=upper_bound,
            mode='lines', 
            line=dict(width=0),
            hoverinfo='none',
            showlegend=False
        ))
        fig2.add_trace(go.Scatter(
            x=df_subset["Distance (m)"], 
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',  # Remplissage entre cette ligne et la ligne précédente
            fillcolor=rgba_color,
            hoverinfo='none',
            showlegend=False,
            #name='Upper Bound'
        ))
                        
    fig2.update_layout(
    height=600,
    legend_title="Click on legends below to hide/show:",
    showlegend=True,
    hovermode='x',
    xaxis={
        'type': 'log',
        'showgrid': True,
        'title': "Distance (m) [Log10]",
        # Autres personnalisations de l'axe X si nécessaire
    },
    yaxis={
        'showgrid': True,
        'title': "Ratio des Doses",
        # 'range': [0.5, 1.5],  # Limites centrées autour de 1
        'tickmode': 'auto',  # Ou 'array' si vous voulez spécifier les ticks manuellement
         #'nticks': 20,  # Nombre de ticks sur l'axe Y, ajustez selon vos besoins
        'minor': {
            'ticks': "inside",
            'ticklen': 6,
            'griddash': 'dot',
            'showgrid': True
        }
    }
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(df_subset)
