import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.utils_func_st import df_multiselect_filters, df_selectbox_filters, normalize_filters, generate_filter_combinations, load_data_with_filters
from utils.plot_func_st import dose_scatter_plot_2, dose_ratio_scatter_plot_2, dose_ratio_bar_chart_2, generate_analogous_colors

# ______________________________________________________________________________________________________________________

# Chargement des données avec mise en cache
@st.cache_data
def load_data(sheet_name):
    return pd.read_excel("./Database/All-at-once_DB.xlsx", sheet_name=sheet_name)

fissions_number_input = 1E17

selected_data = st.sidebar.selectbox("Select the data to load", ("Shielded configurations", "Bare configurations", "Sensitivity studies", "Delayed fission gamma"), index=0)
match selected_data:
    case "Delayed fission gamma":
        st.warning('Data not implemented yet. Coming soon...')
        st.stop()
    case "Bare configurations":
        data = load_data('bare')
        # Valeurs par défaut pour bare configurations
        visu_series_default_columns = ['Fissile', 'Case', 'Code', 'Particle']
        visu_series_default_values = {
            "Fissile": ["U"],
            "Case": "__all__",
            "Code": ["MCNP 6.1"],
            "Particle": ["N"]
        }
        compare_series_default_columns = visu_series_default_columns
        compare_series_default_values = {
            "Fissile": ["U"],
            "Case": ["C1 [U(4.95)O2F2 (H/235U = 410)]"],
            "Code": ["SCALE 6.2", "COG 11.2"],
            "Particle": ["N"]
        }
        reference_case_default_columns = ['Fissile', 'Case', 'Code', 'Particle']
        reference_case_default_values = {
            "Fissile": "U",
            "Case": "C1 [U(4.95)O2F2 (H/235U = 410)]",
            "Code": "MCNP 6.1",
            "Particle": "N"
        }
        my_columns_to_group_by = ['Fissile', 'Case', 'Library', 'Flux-to-dose conversion factor', 'Particle', 'Distance (m)']
    case "Shielded configurations":
        selection = st.sidebar.pills("Please choose one", ["Wall", "Wall position"], selection_mode="single", default = "Wall", key="shielded_configurations")
        match selection:
            case None:
                st.warning('Please select a shielded configuration')
                st.stop()
            case "Wall position":
                st.warning('Data not implemented yet. Coming soon...')
                st.stop()
            case "Wall":
                data = load_data('wall')
                # Valeurs par défaut
                visu_series_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)']
                visu_series_default_values = {
                    "Fissile": ["U"],
                    "Case": ["C1 [U(4.95)O2F2 (H/235U = 410)]"],
                    "Code": ["MCNP 6.1"],
                    "Particle": ["N"],
                    "Screen": ["None", "Concrete"], 
                    "Thickness (cm)": "__all__"
                }
                compare_series_default_columns = visu_series_default_columns
                compare_series_default_values = {
                    "Screen": ["None"],  
                    "Code": "__all__",  
                }
                reference_case_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)']
                reference_case_default_values = {
                    "Screen": "None"
                }
                my_columns_to_group_by = ['Fissile', 'Case', 'Library', 'Flux-to-dose conversion factor', 'Screen', 'Thickness (cm)', 'Particle', 'Distance (m)']
    case "Sensitivity studies":
        selection = st.sidebar.pills("Please choose one", ["Shyshine", "Humidity", "Ground"], selection_mode="single", default = "Shyshine", key="sensitivity_studies")
        match selection:
            case None:
                st.warning('Please select a sensitivity study')
                st.stop()
            case "Shyshine":
                data = load_data('skyshine')
                visu_series_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Skyshine']
                visu_series_default_values = {
                    "Skyshine": ["YES", "NO"] 
                }
                compare_series_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Skyshine']
                compare_series_default_values = {
                    "Skyshine": ["YES", "NO"] 
                }
                reference_case_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Skyshine']
                reference_case_default_values = {
                    "Skyshine": "YES"
                }
                my_columns_to_group_by = ['Fissile', 'Case', 'Library', 'Flux-to-dose conversion factor', 'Skyshine', 'Particle', 'Distance (m)']
            case "Humidity":
                data = load_data('humidity')
                visu_series_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Humidity']
                visu_series_default_values = {
                    "Humidity": (0.0, 0.1, 1.0) 
                }
                compare_series_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Humidity']
                compare_series_default_values = {
                    "Humidity": (0.0, 0.1, 1.0) 
                }                
                reference_case_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Humidity']
                reference_case_default_values = {
                    "Humidity": (0.0)
                }
                my_columns_to_group_by = ['Fissile', 'Case', 'Library', 'Flux-to-dose conversion factor', 'Humidity', 'Particle', 'Distance (m)']
            case "Ground":
                data = load_data('ground')
                visu_series_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Ground']
                visu_series_default_values = {
                    "Ground": ["Concrete", "Soil"] 
                }
                compare_series_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Ground']
                compare_series_default_values = {
                    "Ground": ["Concrete", "Soil"] 
                }
                reference_case_default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Ground']
                reference_case_default_values = {
                    "Ground": "Concrete"
                }
                my_columns_to_group_by = ['Fissile', 'Case', 'Library', 'Flux-to-dose conversion factor', 'Ground', 'Particle', 'Distance (m)']
                

# # Mise à jour des données en fonction du facteur de multiplication
data["Absolute Uncertainty"] =  data["Dose (Gy)"] * data["1s uncertainty"] 
# data["Dose (Gy)"] = data["Dose (Gy)"] * dose_multiplier


# Définition des couleurs par défaut
default_colors = ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72']

# Initialisation de la palette de couleurs personnalisée dans session_state
if 'custom_colors' not in st.session_state:
    st.session_state.custom_colors = default_colors.copy()

# Initialisation du multiplicateur de sigma pour les barres d'erreur
if 'sigma_multiplier' not in st.session_state:
    st.session_state.sigma_multiplier = 2

# Utiliser la palette personnalisée
colors = st.session_state.custom_colors

# Création des onglets
tab1, tab2, tab3 = st.tabs(["📈 Visualize", "🆚 Compare", "🔍 Anomalies"])

with tab1:
    with st.expander("Choose series to plot (click to expand/collapse)", expanded=False):
        visu_data, visu_filters = df_multiselect_filters(data, default_columns=visu_series_default_columns, default_values=visu_series_default_values, key="visu_series")
    
    # Expander pour personnaliser les couleurs
    with st.expander("🎨 Customize color palette (click to expand/collapse)", expanded=False):
        st.write("Customize the colors used in the plots. Changes will apply to all graphs on this page.")
        
        # Créer des colonnes pour organiser les color pickers
        num_colors_to_show = 12  # Afficher les 12 premières couleurs (suffisant pour la plupart des cas)
        cols = st.columns(6)
        
        for i in range(num_colors_to_show):
            with cols[i % 6]:
                new_color = st.color_picker(
                    f"Color {i+1}",
                    value=st.session_state.custom_colors[i],
                    key=f"color_picker_{i}"
                )
                st.session_state.custom_colors[i] = new_color
        
        # Bouton pour réinitialiser les couleurs
        col_reset1, col_reset2, col_reset3 = st.columns([1, 1, 4])
        with col_reset1:
            if st.button("🔄 Reset colors", help="Reset all colors to default values"):
                st.session_state.custom_colors = default_colors.copy()
                st.rerun()
        with col_reset2:
            if st.button("🎲 Randomize", help="Generate random colors"):
                import random
                st.session_state.custom_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(default_colors))]
                st.rerun()
    
    # Formatage des colonnes spécifiques
    formatted_data = visu_data.style.format({
        "1s uncertainty": "{:.2%}",  # Format en pourcentage avec 2 décimales
        "Dose (Gy)": "{:.2e}",       
        "Absolute Uncertainty": "{:.2e}"  
        })
    st.write(f"Estimated prompt dose based on total fissions: {fissions_number_input:.1e}")
    
    # Contrôle des barres d'incertitude
    st.number_input(
        "Error bars (σ):",
        min_value=1,
        max_value=3,
        value=2,
        step=1,
        key="sigma_multiplier",
        help="Adjust the uncertainty multiplier for error bars (1σ, 2σ, or 3σ)"
    )
    
    dose_scatter_plot_2(visu_data, visu_filters, colors, st.session_state.sigma_multiplier)
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
                    data, default_columns=compare_series_default_columns, default_values=compare_series_default_values, key=f"compare_series_{i}")   
            with col4:
                st.caption("Reference case")
                ref_data, ref_filters = df_selectbox_filters(
                    data, default_columns=reference_case_default_columns, default_values=reference_case_default_values, key=f"reference_case_{i}")
            
            # Normaliser ref_filters pour qu'il soit comparable
            normalized_ref_filters = normalize_filters(ref_filters)
            
            # Générer toutes les combinaisons possibles de compare_filters
            all_compare_combinations = generate_filter_combinations(compare_filters)

            # Filtrer les combinaisons identiques à ref_filters
            filtered_combinations = [comb for comb in all_compare_combinations if comb != normalized_ref_filters]

            if filtered_combinations:
                # Utiliser directement les couleurs de la palette personnalisée
                # Calculer l'index de départ pour cette série dans la palette
                color_start_index = sum([len(generate_filter_combinations(st.session_state.get(f'compare_series_{j}', {}))) - 1 
                                        for j in range(1, i) if len(generate_filter_combinations(st.session_state.get(f'compare_series_{j}', {}))) > 1])

                for index, valid_comb in enumerate(filtered_combinations):
                    # Recharger les données avec les filtres validés
                    filtered_compare_data = load_data_with_filters(data, valid_comb)
                    
                    # Utiliser la couleur de la palette personnalisée
                    color_index = (color_start_index + index) % len(colors)
                    trace_color = colors[color_index]
                    
                    # Traiter tous les cas comme une seule série
                    if plot_type == "Scatter Plot":
                        dose_ratio_scatter_plot_2(filtered_compare_data, valid_comb, ref_data, ref_filters, trace_color, global_fig, i, st.session_state.sigma_multiplier)
                    elif plot_type == "Bar Chart":
                        dose_ratio_bar_chart_2(filtered_compare_data, valid_comb, ref_data, ref_filters, trace_color, global_fig, i, st.session_state.sigma_multiplier)
            else:
                st.write(f"Comparison and reference cases for Series {i} are identical. No plot will be generated.")
    
    # Expander pour personnaliser les couleurs
    with st.expander("🎨 Customize color palette (click to expand/collapse)", expanded=False):
        st.write("Customize the colors used in the plots. Changes will apply to all graphs on this page.")
        
        # Créer des colonnes pour organiser les color pickers
        num_colors_to_show = 12  # Afficher les 12 premières couleurs (suffisant pour la plupart des cas)
        cols = st.columns(6)
        
        for i in range(num_colors_to_show):
            with cols[i % 6]:
                new_color = st.color_picker(
                    f"Color {i+1}",
                    value=st.session_state.custom_colors[i],
                    key=f"color_picker_compare_{i}"
                )
                st.session_state.custom_colors[i] = new_color
        
        # Bouton pour réinitialiser les couleurs
        col_reset1, col_reset2, col_reset3 = st.columns([1, 1, 4])
        with col_reset1:
            if st.button("🔄 Reset colors", help="Reset all colors to default values", key="reset_compare"):
                st.session_state.custom_colors = default_colors.copy()
                st.rerun()
        with col_reset2:
            if st.button("🎲 Randomize", help="Generate random colors", key="randomize_compare"):
                import random
                st.session_state.custom_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(default_colors))]
                st.rerun()
    
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

                    # if ecart_relatif > 0.1:
                    # # Construction du dictionnaire pour le nouveau DataFrame
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
        # Ajout du range slider pour définir la plage d'incertitude
        uncertainty_range = st.slider(
        "Relative uncertainty (1σ) range [%]:", 
        min_value=0.0, 
        max_value=100.0, 
        value=(10.0, 100.0),  # Valeurs par défaut
        step=1.0
        )
        high_uncertainty_df = data[
            (data['1s uncertainty'] >= (uncertainty_range[0] / 100)) & 
            (data['1s uncertainty'] <= (uncertainty_range[1] / 100))
        ]
        high_uncertainty_df = high_uncertainty_df.style.format({
            "1s uncertainty": "{:.2%}",  # Format en pourcentage avec 2 décimales
            "Dose (Gy)": "{:.2e}",       # Format scientifique avec 2 décimales
            "Absolute Uncertainty": "{:.2e}"  # Format scientifique avec 2 décimales
        })
        st.dataframe(high_uncertainty_df, hide_index = True)

    with tab6:
        # Grouper les données par les paramètres communs pour la comparaison
        columns_to_group_by = my_columns_to_group_by

        # Suppose que 'data' et 'columns_to_group_by' sont déjà définis
        ecarts_significatifs = calculate_significant_discrepancies(data, columns_to_group_by)
        
        discrepancy_range = st.slider(
            "Relative difference range [%]:", 
            min_value=0.0, 
            max_value=100.0, 
            value=(10.0, 100.0),  
            step=1.0)
        ecarts_significatifs_filtered = ecarts_significatifs[
            (ecarts_significatifs['Relative_Difference'].str.rstrip('%').astype(float) >= discrepancy_range[0]) & 
            (ecarts_significatifs['Relative_Difference'].str.rstrip('%').astype(float) <= discrepancy_range[1])
        ]

        if not ecarts_significatifs_filtered.empty:
            st.dataframe(ecarts_significatifs_filtered, hide_index=True)
        else:
            st.write("No significant discrepancies found within the selected range.")