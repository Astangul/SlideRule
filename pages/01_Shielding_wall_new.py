import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.utils_func_st import df_multiselect_filters, df_selectbox_filters, normalize_filters, generate_filter_combinations, load_data_with_filters
from utils.plot_func_st import dose_scatter_plot_2, dose_ratio_scatter_plot_2, dose_ratio_bar_chart_2, generate_analogous_colors

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
    max_value=1.0e+21,
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

# Création des onglets
tab1, tab2, tab3 = st.tabs(["📈 Visualize", "🆚 Compare", "🔢 Anomalies"])

with tab1:
    with st.expander("Choose series to plot (click to expand/collapse)", expanded=False):
        visu_data, visu_filters = df_multiselect_filters(data, default_columns=['Fissile', 'Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)'], key="visu_series")
    # Formatage des colonnes spécifiques
    formatted_data = visu_data.style.format({
        "1s uncertainty": "{:.2%}",  # Format en pourcentage avec 2 décimales
        "Dose (Gy)": "{:.2e}",       
        "Absolute Uncertainty": "{:.2e}"  
        })
    st.write(f"Estimated prompt dose based on total fissions: {fissions_number_input:.2e}")
    dose_scatter_plot_2(visu_data, visu_filters, colors)
    st.dataframe(formatted_data, hide_index=True)

# Initialisation de l'état de la session pour le suivi des séries
if 'num_series' not in st.session_state:
    st.session_state['num_series'] = 1  # Commence avec une série

with tab2:
    with st.container():
        btn_col = st.columns([1, 1, 25])
        with btn_col[0]:
            if st.button("➕", key="btn_add"):
                st.session_state['num_series'] += 1
        with btn_col[1]:
            if st.button("➖", key="btn_remove") and st.session_state['num_series'] > 1:
                st.session_state['num_series'] -= 1
    
    global_fig = go.Figure()  # Créer une seule figure pour tout le traçage
    
    plot_type = st.radio("Select the plot type:", ["Scatter Plot", "Bar Chart"], key="global_plot_type")
    
    for i in range(1, st.session_state['num_series'] + 1):
        with st.expander(f"Serie {i} (click to expand/collapse)", expanded=False):
            col3, col4 = st.columns(2)
            with col3:
                st.caption("Comparison case(s)")
                compare_data, compare_filters = df_multiselect_filters(
                    data, default_columns=['Fissile', 'Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)'], key=f"compare_series_{i}")   
            with col4:
                st.caption("Reference case")
                ref_data, ref_filters = df_selectbox_filters(
                    data, default_columns=['Fissile', 'Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)'], key=f"reference_case_{i}")
            
            # Normaliser ref_filters pour qu'il soit comparable
            normalized_ref_filters = normalize_filters(ref_filters)
            
            # Générer toutes les combinaisons possibles de compare_filters
            all_compare_combinations = generate_filter_combinations(compare_filters)

            # Filtrer les combinaisons identiques à ref_filters
            filtered_combinations = [comb for comb in all_compare_combinations if comb != normalized_ref_filters]

            if filtered_combinations:
                # Définir la couleur de base pour chaque série
                base_color = colors[i % len(colors) - 1]
                # Générer les couleurs analogues pour les différents cas d'une même série
                analogous_colors = generate_analogous_colors(base_color, num_colors=len(filtered_combinations), spread=0.3)

                for index, valid_comb in enumerate(filtered_combinations):
                    # Recharger les données avec les filtres validés
                    filtered_compare_data = load_data_with_filters(data, valid_comb)
                    
                    # Couleur analogue pour chaque cas
                    adjusted_color = analogous_colors[index]
                    
                    # Traiter tous les cas comme une seule série
                    if plot_type == "Scatter Plot":
                        dose_ratio_scatter_plot_2(filtered_compare_data, valid_comb, ref_data, ref_filters, adjusted_color, global_fig, i)
                    elif plot_type == "Bar Chart":
                        dose_ratio_bar_chart_2(filtered_compare_data, valid_comb, ref_data, ref_filters, adjusted_color, global_fig, i)
            else:
                st.write(f"Comparison and reference cases for Series {i} are identical. No plot will be generated.")
    
    # Toggle pour l'échelle logarithmique sur l'axe X
    if plot_type == "Scatter Plot":
        log_x_fig = st.toggle("X-axis log scale", value=True, key="log_x_global_fig")
        # Définir l'échelle de l'axe X
        if log_x_fig:
            global_fig.update_xaxes(type='log', title="Distance (m) [Log10]")
        else:
            global_fig.update_xaxes(type='linear', title="Distance (m)")
    
    # Afficher la figure globale avec toutes les séries dans un seul graphique
    st.plotly_chart(global_fig, use_container_width=True)

with tab3:
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

    tab4, tab5, tab6 = st.tabs(["❔ Missing values", "❗ High uncertainty", "🚨 Discrepancies"])

    with tab4:
        st.write("Missing dose values in the dataset:")
        empty_values_df = data[data.isnull().any(axis=1)]
        st.dataframe(empty_values_df, hide_index = True)

    with tab5:
        st.write("Relative uncertainty (1σ) exceeding 10%:")
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
