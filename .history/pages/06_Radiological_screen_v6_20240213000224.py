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

# Add a slider for the number of fissions
#  Method 1 : 
# fissions = st.slider("Select the number of fissions:", 
#                      min_value=1e16, max_value=1e20, 
#                      value=1e17, step=1e15, 
#                      format="%.1e")

# st.write(f"Number of fissions selected: {fissions}")


# Method 2
# Generate a list of values on a logarithmic scale with steps of 0.1
values = []
for exponent in range(13, 21):  # For 1.0E13 to 1.0E20
    base = 10 ** exponent
    values.extend([base * i for i in np.arange(1, 10, 0.1)])  # 1.0, 1.1, ..., 9.9

# Convertir les valeurs en chaînes de caractères pour l'affichage
options = [f"{v:.1e}" for v in values]

# S'assurer que la valeur par défaut est formatée de la même manière que les options
default_value = f"{1e17:.1e}"

# Créer un select_slider pour permettre à l'utilisateur de choisir une valeur
selected_value = st.select_slider(
    'Select the number of fissions:',
    options=options,
    value=default_value  # Utilisation de la valeur par défaut correctement formatée
)

# Convertir la valeur sélectionnée en float pour les calculs
fissions = float(selected_value)

st.write(f"Number of fissions selected: {fissions:.1e} ")

# # Calculate le facteur de multiplication des doses
dose_multiplier = fissions / 1e17


# Charger vos données
@st.cache_data
def load_data():
    data = pd.read_excel("./DB/All-at-once_DB.xlsx")
    data["Absolute Uncertainty"] = data["1s uncertainty"] * data["Dose (Gy)"]
    data['Filter Combo'] = data.apply(lambda row: f"{row['Particle']}_{row['Screen']}_{row['Code']}_{row['Case']}_{row['Thickness (cm)']}", axis=1)
    return data


data = load_data()


# Définir une liste de couleurs
# colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
colors = [
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#e377c2',  # raspberry yogurt pink
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#d62728',  # brick red
    '#1f77b4',  # muted blue
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
st.header("Parameter Selection")


with st.expander("↳ Click to expand/collapse", expanded=True):
    # Créer deux colonnes pour les filtres
    col1, col2 = st.columns(2)
    
    # Filtres pour les séries à comparer
    with col1:
        st.subheader("Series to Compare (multiple choice)")
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
    # st.subheader("Données Combinées du cas de Référence et Séries Sélectionnées")
    st.dataframe(formatted_data)


with tab1:
    # st.subheader("Graphe des Doses")
    log_x = st.checkbox("X-axis log scale", value=True)
    log_y = st.checkbox("Y-axis log scale", value=True)
    fig1 = go.Figure()

    # Name for the legend of the reference case
    ref_label = f"{ref_particle}_{ref_screen}_{ref_code}_{ref_case}_{ref_thickness} (reference)"

    # Adding the reference case with a custom label
    fig1.add_trace(go.Scatter(x=ref_data["Distance (m)"],
                              y=ref_data["Dose (Gy)"]*dose_multiplier,
                              mode='lines+markers', 
                              marker_symbol='diamond', marker_size=8,
                              name=ref_label,
                              error_y=dict(type='data', array=2*ref_data["Absolute Uncertainty"]*dose_multiplier, visible=True)
                              ))

    # Adding series to compare
    # for combo in compare_data['Filter Combo'].unique():
    for index, combo in enumerate(compare_data['Filter Combo'].unique()):
        color = colors[index % len(colors)]
        
        df_subset = compare_data[compare_data['Filter Combo'] == combo]
        fig1.add_trace(go.Scatter(x=df_subset["Distance (m)"], 
                                  y=df_subset["Dose (Gy)"]*dose_multiplier,
                                  mode='lines+markers', 
                                  marker_symbol='circle-dot', marker_size=8,
                                  line=dict(dash='dash', color=color), 
                                  name=combo,
                                  error_y=dict(type='data', array=2*df_subset["Absolute Uncertainty"]*dose_multiplier, visible=True)
                                  ))

    fig1.update_layout(
        hovermode='x',
        showlegend=True,
        xaxis={'showgrid':True},
        yaxis={'showgrid':True},
        height=700,  #width=900,
        # title="Dose en fonction de la Distance (échelles logarithmiques)",
        legend_title="Click on legends below to hide/show:",
        # legend=dict(
        #     orientation="h",  # Horizontal
        #     xanchor="center",  # Ancre au centre
        #     x=0.5,  # Positionner au centre en x
        #     y=-0.3  # Position en dessous du graphique
        # )
    )
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
    log_x_fig2 = st.checkbox("X-axis log scale", value=True, key="log_x_fig2")
    #st.subheader("Graphe des Ratios")
    fig2 = go.Figure()
    for index, combo in enumerate(compare_data['Filter Combo'].unique()):
        # Choose a color from the color chart
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
        # Calculation of the absolute uncertainty
        df_subset['Absolute Combined Uncertainty'] = df_subset['Combined Uncertainty'] * df_subset['Dose Ratio']
        
        #___________
        # Calcul des limites supérieure et inférieure pour les bandes d'erreur
        upper_bound = df_subset['Dose Ratio'] + 1*df_subset['Absolute Combined Uncertainty']
        lower_bound = df_subset['Dose Ratio'] - 1*df_subset['Absolute Combined Uncertainty']

        legend_group = f"group_{index}"  # Unique name for each plot group
        # Define the hover template to format values in scientific notation
        hovertemplate = '%{y:.3f} ± %{customdata:.2e}'

        # Plotting the main line
        fig2.add_trace(go.Scatter(
            x=df_subset["Distance (m)"], 
            y=df_subset["Dose Ratio"],
            customdata=1*df_subset['Absolute Combined Uncertainty'],
            error_y=dict(type='data', array=1*df_subset['Absolute Combined Uncertainty'], visible=True),
            mode='lines+markers',
            marker_symbol='circle-open-dot', marker_size=8,
            line=dict(dash='dash', color=color),
            name=combo,
            legendgroup=legend_group,  # Associer au groupe
            hovertemplate=hovertemplate
        ))
        
        # Tracés des bandes d'erreur liés à la ligne principale par le groupe
        fig2.add_trace(go.Scatter(
            x=df_subset["Distance (m)"],
            y=upper_bound,
            mode='lines', 
            line=dict(width=0),
            hoverinfo='none',
            showlegend=False,
            legendgroup=legend_group,  # Même groupe que la ligne principale
            fillcolor=rgba_color,
            #name='Upper Bound'
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
            legendgroup=legend_group, # Même groupe
            #name='Lower Bound'
        ))

    fig2.update_layout(
        hovermode='x ',
        height=600,
        showlegend=True,
        legend_title="Click on legends below to hide/show:",
        xaxis={
            'showgrid': True,
            'title': "Distance (m) [Log10]",
        # Autres personnalisations de l'axe X si nécessaire
        },
        yaxis={
            'showgrid': True,
            'title': "Ratio des Doses",
            # 'range': [0.5, 1.5],  # Limits centered around 1
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

# Mise à jour conditionnelle des axes
    if log_x_fig2:
        fig2.update_xaxes(type='log', title="Distance (m) [Log10]")
    else:
        fig2.update_xaxes(type='linear', title="Distance (m)")
    
    st.plotly_chart(fig2, use_container_width=True)
    #st.dataframe(df_subset)