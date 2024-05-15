import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.st_filter_dataframe import df_multiselect_filters, df_selectbox_filters
from utils.plot_functions import dose_ratio_scatter_plot, dose_ratio_bar_chart

# Configuration de la page Streamlit
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

st.write("My first Streamlit app")

# Chargement des données avec mise en cache
@st.cache_data
def load_data(sheet_name):
    return pd.read_excel("./DB/All-at-once_DB.xlsx", sheet_name=sheet_name)

data = load_data('screen')

# Création des valeurs pour le select_slider
values = []
for exponent in range(13, 21):  # For 1.0E13 to 1.0E20
    base = 10 ** exponent
    values.extend([base * i for i in np.arange(1, 10, 0.1)])  # 1.0, 1.1, ..., 9.9

# Conversion des valeurs en chaînes de caractères pour l'affichage
options = [f"{v:.1e}" for v in values]

# S'assurer que la valeur par défaut est formatée de la même manière que les options
default_value = f"{1e17:.1e}"

# Initialisation de l'état de la session
if 'fission_slider' not in st.session_state:
    st.session_state['fission_slider'] = default_value

if 'fission_input' not in st.session_state:
    st.session_state['fission_input'] = 1e17

# Callbacks pour synchroniser les valeurs
def update_fission_slider():
    st.session_state.fission_slider = f"{st.session_state.fission_input:.1e}"

def update_fission_input():
    st.session_state.fission_input = float(st.session_state.fission_slider)

# Synchronisation des valeurs avant la création des widgets
if f"{st.session_state.fission_input:.1e}" != st.session_state.fission_slider:
    st.session_state.fission_slider = f"{st.session_state.fission_input:.1e}"

# Widgets pour sélectionner le nombre de fissions
selected_value = st.sidebar.select_slider(
    'Select the number of fissions:',
    options=options,
    key="fission_slider",
    on_change=update_fission_input
)

# Convertir la valeur sélectionnée en float pour les calculs
fissions_number_slider = float(selected_value)

# Créer un number_input pour permettre à l'utilisateur d'entrer manuellement la valeur
fissions_number_input = st.sidebar.number_input(
    "OR enter the number of fissions",
    min_value=1.0e+13,
    max_value=1.0e+20,
    value=fissions_number_slider,
    step=None,
    format="%.1e",
    key="fission_input",
    on_change=update_fission_slider
)

# Calcul du facteur de multiplication des doses
dose_multiplier = fissions_number_input / 1e17

# Mise à jour des données en fonction du facteur de multiplication
data["Absolute Uncertainty"] =  data["Dose (Gy)"] * data["1s uncertainty"] * dose_multiplier
data["Dose (Gy)"] = data["Dose (Gy)"] * dose_multiplier

# Définition des couleurs
# colors = ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#B2828D', '#6C7C32', '#778AAE', '#862A16', '#A777F1', '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038']
# colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
colors = ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72']


st.divider()
# Sélection des paramètres
st.header("Parameter selection")

with st.expander("Click to expand/collapse", expanded=False):
    # Créer deux colonnes pour les filtres
    col1, col2 = st.columns(2)
    with col1:
        # Utiliser df_multiselect_filters pour filtrer les séries à comparer
        st.subheader("Series to compare")
        compare_data, compare_filters = df_multiselect_filters(data, default_columns=['Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)'], key="compare_series")
    with col2:
        # Utiliser df_selectbox_filters pour sélectionner le cas de référence
        st.subheader("Reference case")
        ref_data, ref_filters = df_selectbox_filters(data, default_columns=['Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)'], key="reference_case")

# Exclure le cas de référence des données de comparaison
# Par exemple, si ref_filters est un dictionnaire contenant les filtres appliqués:
ref_criteria = {col: ref_data.iloc[0][col] for col in ref_filters.keys()}

# Construire une condition pour filtrer le cas de référence hors des séries à comparer
condition = np.logical_and.reduce([compare_data[col].eq(value) for col, value in ref_criteria.items()])

# Exclure le cas de référence des séries à comparer
compare_data = compare_data[~condition]

# Fusionner les données de référence et les données comparatives
combined_data = pd.concat([ref_data, compare_data]).drop_duplicates()

# Créer un libellé pour le cas de référence
ref_label_parts = [str(value[0]) for key, value in ref_filters.items() if value]

# Joindre les parties du libellé avec des underscores pour créer le libellé final sans crochets
ref_label = "_".join(ref_label_parts) + " (reference)"


compare_data['unique_key'] = compare_data.apply(lambda row: '_'.join([str(row[col]) for col in compare_filters.keys()]), axis=1)

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
    st.dataframe(formatted_data, hide_index=True)


with tab1:
#     # st.subheader("Graphe des Doses")
#     log_x = st.toggle("X-axis log scale", value=True, key="log_x_fig1")
#     log_y = st.toggle("Y-axis log scale", value=True, key="log_y_fig1")
#     st.write(f"Estimated prompt dose based on total fissions: {fissions_number_input:.2e}")
#     fig1 = go.Figure()

#     # Adding the reference case with a custom label
#     fig1.add_trace(go.Scatter(x=ref_data["Distance (m)"],
#                               y=ref_data["Dose (Gy)"],
#                               mode='lines+markers', 
#                               marker_symbol='diamond', marker_size=8,
#                               name=ref_label,
#                               error_y=dict(type='data', array=2*ref_data["Absolute Uncertainty"], visible=True)
#                               ))

#     # Adding series to compare
#     for index, (key, group) in enumerate(compare_data.groupby('unique_key')):
#         fig1.add_trace(go.Scatter(x=group["Distance (m)"], 
#                                   y=group["Dose (Gy)"],
#                                   mode='lines+markers', 
#                                   marker_symbol='circle-dot', marker_size=8,
#                                   line=dict(dash='dash', color=colors[index % len(colors)]), 
#                                   name=f'{key}',
#                                   error_y=dict(type='data', array=2*group["Absolute Uncertainty"], visible=True)
#                                   ))

#     fig1.update_layout(
#         hovermode='x',
#         showlegend=True,
#         xaxis={'showgrid':True},
#         yaxis={'showgrid':True},
#         height=700,  #width=900,
#         legend_title="Click on legends below to hide/show:",
#         #title="Dose en fonction de la Distance (échelles logarithmiques)",
#         # legend=dict(
#         #     orientation="h",  # Horizontal
#         #     xanchor="center",  # Ancre au centre
#         #     x=0.5,  # Positionner au centre en x
#         #     y=-0.3  # Position en dessous du graphique
#         # )
#     )
#     fig1.update_xaxes(minor = dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))
#     fig1.update_yaxes(minor = dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))

# # Mise à jour conditionnelle des axes
#     if log_x:
#         fig1.update_xaxes(type='log', title="Distance (m) [Log10]")
#     else:
#         fig1.update_xaxes(type='linear', title="Distance (m)")
    
#     if log_y:
#         fig1.update_yaxes(type='log', title="Dose (Gy) ± 2σ [Log10]", tickformat='.2e')
#     else:
#         fig1.update_yaxes(type='linear', title="Dose (Gy) ± 2σ",  tickformat='.2e')

#     #st.plotly_chart(fig1)
#     st.plotly_chart(fig1, use_container_width=True) # le graphe occupe toute la page



with tab2:
    plot_type = st.radio("Select plot type:", ["Scatter Plot", "Bar Chart"])
    
    if plot_type == "Scatter Plot":
        dose_ratio_scatter_plot(compare_data, ref_data, colors)
    elif plot_type == "Bar Chart":
        dose_ratio_bar_chart(compare_data, ref_data, colors)



@st.cache_data
def calculate_significant_discrepancies(input_data, grouping_columns):
    """
    Calculate and return a DataFrame of significant discrepancies between codes in the dataset.

    The function groups data by specified columns and computes differences in doses between codes. 
    A discrepancy is deemed significant if the absolute difference exceeds three times the combined 
    standard deviations of the measurements.

    Args:
        input_data (DataFrame): The DataFrame with data to be analyzed.
        grouping_columns (list): Columns to group data by.

    Returns:
        DataFrame: A DataFrame containing the significant discrepancies, including group details, 
        involved codes, and measurement differences.
    """    
    grouped_data = input_data.groupby(grouping_columns)
    significant_differences = []

    for name, local_group in grouped_data:
        codes = local_group['Code'].unique()
        for i, code1 in enumerate(codes):
            for code2 in codes[i + 1:]:
                dose_code1 = local_group[local_group['Code'] == code1]['Dose (Gy)'].iloc[0]
                dose_code2 = local_group[local_group['Code'] == code2]['Dose (Gy)'].iloc[0]
                ecart_relatif = abs(dose_code1 - dose_code2) / max(dose_code1, dose_code2)
                
                if ecart_relatif > 0.1:
                    # Construction du dictionnaire pour le nouveau DataFrame
                    new_row_data = {
                        'Code1': code1,
                        'Code2': code2,
                        'Dose_Code1': f"{dose_code1:.2e}",
                        'Dose_Code2': f"{dose_code2:.2e}",
                        'Relative_Difference': f"{ecart_relatif:.2%}"
                    }
                    
                    # Ajouter les détails du groupe en premier dans le dictionnaire
                    group_data = {}
                    if isinstance(name, tuple):  # Si le groupe est par plusieurs colonnes
                        for col, val in zip(grouping_columns, name):
                            group_data[col] = val
                    else:  # Si le groupe est par une seule colonne
                        group_data[grouping_columns[0]] = name
                    
                    # Fusion des dictionnaires de groupes et de résultats
                    full_row_data = {**group_data, **new_row_data}
                    
                    # Ajout de la ligne à la liste de données
                    significant_differences.append(full_row_data)

    # Création du DataFrame final à partir de la liste de dictionnaires
    return pd.DataFrame(significant_differences)

st.divider()
st.header("Error Handling")
tab4, tab5, tab6 = st.tabs(["❔ Missing values", "❗ High uncertainty (1σ>10%)", "🚨 Discrepancies"])

with tab4:
    empty_values_df = data[data.isnull().any(axis=1)]
    st.dataframe(empty_values_df, hide_index = True)

with tab5:
    high_uncertainty_df = data[data['1s uncertainty'] > 0.1]
    high_uncertainty_df = high_uncertainty_df.style.format({
    "1s uncertainty": "{:.2%}",  # Format en pourcentage avec 2 décimales
    "Dose (Gy)": "{:.2e}",       # Format scientifique avec 2 décimales
    "Absolute Uncertainty": "{:.2e}"  # Format scientifique avec 2 décimales
})
    st.dataframe(high_uncertainty_df, hide_index = True)


# Supposons que 'data' est déjà chargé
codes_uniques = data['Code'].unique()

# Grouper les données par les paramètres communs pour la comparaison
columns_to_group_by = ['Fissile', 'Case', 'Library', 'Flux-to-dose conversion factor', 'Screen', 'Thickness (cm)', 'Particle', 'Distance (m)']

# Suppose que 'data' et 'columns_to_group_by' sont déjà définis
ecarts_significatifs = calculate_significant_discrepancies(data, columns_to_group_by)

with tab6:
    if not ecarts_significatifs.empty:
        st.write("Relative difference exceeding 10%:")
        st.dataframe(ecarts_significatifs, hide_index=True)
    else:
        st.write("No significant discrepancies found between codes.")
