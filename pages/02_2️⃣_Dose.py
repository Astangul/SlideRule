import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d


from utils.utils_func_st import df_multiselect_filters, df_selectbox_filters, normalize_filters, generate_filter_combinations, load_data_with_filters
from utils.plot_func_st import dose_scatter_plot_3, dose_ratio_scatter_plot_2, dose_ratio_bar_chart_2, generate_analogous_colors

# ______________________________________________________________________________________________________________________
# Configuration de la page Streamlit
st.set_page_config(
    page_title="Slide-Rule",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': "https://gitlab.extra.irsn.fr/snc/SlideRule/-/issues",
        'About': "https://ncsp.llnl.gov/analytical-methods/criticality-slide-rule"
    }
)
sidebar_logo_path = "./icons/Slide-Rule_orange.png"
main_body_logo_path = "./icons/Slide-Rule_DallE-1.png"
st.logo(image = sidebar_logo_path, size="large", icon_image = sidebar_logo_path)
st.warning('Section in development (WIP)')
# ______________________________________________________________________________________________________________________

# Chargement des données avec mise en cache
@st.cache_data
def load_data(sheet_name):
    return pd.read_excel("./DB/All-at-once_DB.xlsx", sheet_name=sheet_name)

data = load_data('final')

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

# **Calculer dynamiquement le pas**
current_value = st.session_state.get('fission_input', 1e17)
exponent = np.floor(np.log10(current_value))
step = (10 ** exponent) * 0.1

# Créer un number_input pour permettre à l'utilisateur d'entrer manuellement la valeur
fissions_number_input = st.sidebar.number_input(
    "OR enter the number of fissions",
    min_value=1.0e+13,
    max_value=1.0e+21,
    value=fissions_number_slider,
    step=step,
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
tab1, tab2 = st.tabs(["📈 Visualize", "🔢 Data"])
with tab1:
    with st.expander("Choose series to plot (click to expand/collapse)", expanded=False):

        # Fixer les filtres disponibles (l'utilisateur ne peut pas en ajouter d'autres)
        fixed_filters = ['Fissile', 'Case', 'Screen']

        # Valeurs par défaut (imposées pour Fissile et Case)
        fixed_default_values = {
            "Fissile": ['U'],  # Une seule valeur possible
            "Case": ['C1 [U(4.95)O2F2 (H/235U = 410)]'],  # Une seule valeur possible
            "Screen": ["None", "Concrete"]  # "None" obligatoire + 1 choix max
        }

        # Sélection unique pour "Fissile"
        selected_fissile = st.selectbox("Select fissile material", options=data["Fissile"].unique(), index=0)

        # Sélection unique pour "Case"
        selected_case = st.selectbox("Select case", options=data["Case"].unique(), index=0)

        # Sélection personnalisée pour "Screen" : "None" obligatoire + un seul autre écran possible
        available_screens = [s for s in data["Screen"].unique() if s != "None"]  # Exclut "None" temporairement
        selected_screens = st.multiselect("Select screen material", 
                                          options=available_screens, 
                                          default=["Concrete"], 
                                          max_selections=1)  # Limite à 1 choix

        # Assurer que "None" est toujours inclus
        selected_screens.insert(0, "None")

        # Construire le dictionnaire des filtres avec ces valeurs fixes
        visu_filters = {
            "Fissile": [selected_fissile],  # Convertir en liste pour compatibilité
            "Case": [selected_case],
            "Screen": selected_screens
        }

        # Filtrer les données avec les sélections
        visu_data = data[
            (data["Fissile"].isin(visu_filters["Fissile"])) &
            (data["Case"].isin(visu_filters["Case"])) &
            (data["Screen"].isin(visu_filters["Screen"]))
        ]

    # ______________________________________________________________________________________________________________________
    # st.tabs(["📈 Visualize"])
    # with st.expander("Choose series to plot (click to expand/collapse)", expanded=False):
    #     final_series_default_columns = ['Fissile', 'Case', 'Screen']
    #     final_series_default_values = {
    #         "Case": ['C1 [U(4.95)O2F2 (H/235U = 410)]'],
    #         "Screen": ["None", "Concrete"],   
    #     }
    #     visu_data, visu_filters = df_multiselect_filters(data, default_columns=final_series_default_columns, default_values=final_series_default_values, key="final_visu_series")
    # ______________________________________________________________________________________________________________________
    # Appel de la fonction pour obtenir la figure
    st.write(f"Estimated prompt dose based on total fissions: {fissions_number_input:.1e}")
    fig = dose_scatter_plot_3(visu_data, visu_filters, colors)

    # ______________________________________________________________________________________________________________________
    df_curve_fit = load_data("curve_fit")

    def filter_curve_fit_data(data, filters):
        """
        Filtre les données de l'onglet 'curve_fit' en fonction des filtres spécifiés.

        Args:
            data (pd.DataFrame): Les données à filtrer.
            filters (dict): Les filtres à appliquer (ex. {'Fissile': 'U', 'Screen': ['Concrete', 'Steel']}).

        Returns:
            pd.DataFrame: Les données filtrées.
        """
        # Appliquer les filtres
        for column, value in filters.items():
            if value != "__all__":
                if isinstance(value, list):  # Si le filtre contient plusieurs valeurs
                    data = data[data[column].isin(value)]
                else:  # Si le filtre est une valeur unique
                    data = data[data[column] == value]

        return data

    # Sélection des filtres pour l'onglet 'curve_fit'
    filtered_curve_fit_data = filter_curve_fit_data(df_curve_fit, visu_filters)
    # st.dataframe(filtered_curve_fit_data, hide_index=False)

    # Fonction pour calculer la dose
    def calculate_interpolated_dose(distance, A, k, b):
        return A * distance**-k * np.exp(-b * distance)

    # Fonction d'interpolation des paramètres pour une particule donnée (N ou P)
    def interpolate_parameters(filtered_data, particle, T_new):
        """
        Interpole les paramètres A, k et b pour une particule donnée et une épaisseur d'écran donnée.

        Args:
            filtered_data (pd.DataFrame): Données filtrées contenant les paramètres fittés.
            particle (str): Type de particule ('N' ou 'P').
            T_new (float): Épaisseur d'écran pour l'interpolation.

        Returns:
            dict: Paramètres interpolés {A, k, b} et leurs incertitudes.
        """
        # Filtrer les données pour la particule choisie
        data_particle = filtered_data[filtered_data["Particle"] == particle]

        if data_particle.empty:
            st.warning(f"No data available for Particle {particle}. Skipping interpolation.")
            return None

        thicknesses = data_particle["Thickness (cm)"].values
        A_values = data_particle["A"].values
        k_values = data_particle["k"].values
        b_values = data_particle["b"].values

        # Gestion des incertitudes
        A_uncertainty_values = data_particle["A_uncertainty"].values
        k_uncertainty_values = data_particle["k_uncertainty"].values
        b_uncertainty_values = data_particle["b_uncertainty"].values

        # Conversion en espace logarithmique
        logA_values = np.log(A_values)
        logk_values = np.log(k_values)
        logb_values = np.log(b_values)
        logA_uncertainty = np.log(A_uncertainty_values + 1e-12)
        logk_uncertainty = np.log(k_uncertainty_values + 1e-12)
        logb_uncertainty = np.log(b_uncertainty_values + 1e-12)

        # Création des interpolateurs
        logA_interp = interp1d(thicknesses, logA_values, fill_value="extrapolate", kind='linear')
        logk_interp = interp1d(thicknesses, logk_values, fill_value="extrapolate", kind='linear')
        logb_interp = interp1d(thicknesses, logb_values, fill_value="extrapolate", kind='linear')

        logA_uncertainty_interp = interp1d(thicknesses, logA_uncertainty, fill_value="extrapolate", kind='linear')
        logk_uncertainty_interp = interp1d(thicknesses, logk_uncertainty, fill_value="extrapolate", kind='linear')
        logb_uncertainty_interp = interp1d(thicknesses, logb_uncertainty, fill_value="extrapolate", kind='linear')

        # Vérifier si T_new est hors plage
        if T_new < min(thicknesses) or T_new > max(thicknesses):
            st.warning(f"Warning: The selected thickness ({T_new} cm) is outside the available range for Particle {particle}!")

        # Calcul des paramètres interpolés
        A_new = np.exp(logA_interp(T_new))
        k_new = np.exp(logk_interp(T_new))
        b_new = np.exp(logb_interp(T_new))

        A_uncertainty_new = np.exp(logA_uncertainty_interp(T_new))
        k_uncertainty_new = np.exp(logk_uncertainty_interp(T_new))
        b_uncertainty_new = np.exp(logb_uncertainty_interp(T_new))

        return {
            "A": A_new, "k": k_new, "b": b_new,
            "A_uncertainty": A_uncertainty_new,
            "k_uncertainty": k_uncertainty_new,
            "b_uncertainty": b_uncertainty_new
        }

    # Récupération de l'épaisseur choisie par l'utilisateur
    st.sidebar.divider()
    T_new = st.sidebar.number_input("Enter screen thickness (cm) for interpolation:", min_value=0.0, step=1.0, value=15.0)
    
    color_N = "#9400D3"   # Violet profond pour la courbe interpolée des Neutrons (N)
    color_P = "#FF4500"  # Orange foncé pour la courbe interpolée des Photons (P)

    # Interpolation pour les Neutrons (N)
    params_N = interpolate_parameters(filtered_curve_fit_data, "N", T_new)
    if params_N:
        x_values = np.logspace(np.log10(1), np.log10(1200), 100)
        y_values_N = calculate_interpolated_dose(x_values, params_N["A"], params_N["k"], params_N["b"])
        y_values_upper_N = calculate_interpolated_dose(x_values, params_N["A"] + params_N["A_uncertainty"], params_N["k"] + params_N["k_uncertainty"], params_N["b"] + params_N["b_uncertainty"])
        y_values_lower_N = calculate_interpolated_dose(x_values, params_N["A"] - params_N["A_uncertainty"], params_N["k"] - params_N["k_uncertainty"], params_N["b"] - params_N["b_uncertainty"])

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_N,
            mode='lines',  # ✅ LIGNE CONTINUE SANS MARQUEUR
            name=f"Interpolated N ({T_new} cm)",
            legendgroup="Interpolated N",  # ✅ Groupe de légende
            line=dict(color=color_N),
            hoverinfo='skip'  # ✅ Désactive l'affichage au survol
        ))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_upper_N,
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip', # ✅ Désactive l'affichage au survol
            showlegend=False,
            legendgroup="Interpolated N",  # ✅ Lie la bande à la légende principale
            fillcolor='rgba(148, 0, 211, 0.2)'
        ))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_lower_N,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            legendgroup="Interpolated N",  # ✅ Lie la bande à la légende principale
            fillcolor='rgba(148, 0, 211, 0.2)',
            hoverinfo='skip', # ✅ Désactive l'affichage au survol
            showlegend=False
        ))


    # Interpolation pour les Photons (P)
    params_P = interpolate_parameters(filtered_curve_fit_data, "P", T_new)
    if params_P:
        y_values_P = calculate_interpolated_dose(x_values, params_P["A"], params_P["k"], params_P["b"])
        y_values_upper_P = calculate_interpolated_dose(x_values, params_P["A"] + params_P["A_uncertainty"], params_P["k"] + params_P["k_uncertainty"], params_P["b"] + params_P["b_uncertainty"])
        y_values_lower_P = calculate_interpolated_dose(x_values, params_P["A"] - params_P["A_uncertainty"], params_P["k"] - params_P["k_uncertainty"], params_P["b"] - params_P["b_uncertainty"])

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_P,
            mode='lines',
            name=f"Interpolated P ({T_new} cm)",
            legendgroup="Interpolated P",  # ✅ Groupe de légende
            line=dict(color=color_P),
            hoverinfo='skip'  # ✅ Désactive l'affichage au survol
        ))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_upper_P,
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip', # ✅ Désactive l'affichage au survol
            showlegend=False,
            legendgroup="Interpolated P",  # ✅ Lie la bande à la légende principale
            fillcolor='rgba(255, 69, 0, 0.2)'
        ))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_lower_P,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            legendgroup="Interpolated P",  # ✅ Lie la bande à la légende principale
            fillcolor='rgba(255, 69, 0, 0.2)',
            hoverinfo='skip', # ✅ Désactive l'affichage au survol
            showlegend=False
        ))


    # 🔹 Permettre à l'utilisateur d'entrer des distances spécifiques pour calculer la dose
    st.sidebar.divider()
    user_distances_input = st.sidebar.text_input("Enter distances (comma-separated, in meters):", "5, 10, 50, 100, 500, 1000")
    
    # 🔸 Convertir les distances entrées en une liste de valeurs numériques
    try:
        user_distances = [float(d.strip()) for d in user_distances_input.split(",") if d.strip()]
        user_distances = [d for d in user_distances if d > 0]  # Filtrer les valeurs négatives
    except ValueError:
        st.sidebar.error("Invalid input. Please enter comma-separated numeric values.")
        user_distances = []
    
    # 🔸 Calculer les doses aux distances spécifiées
    if user_distances:
        doses_N = [calculate_interpolated_dose(d, params_N["A"], params_N["k"], params_N["b"]) for d in user_distances] if params_N else []
        doses_P = [calculate_interpolated_dose(d, params_P["A"], params_P["k"], params_P["b"]) for d in user_distances] if params_P else []
    
        # 🔹 Ajouter les marqueurs 🟢 sur les courbes interpolées
        if params_N:
            fig.add_trace(go.Scatter(
                x=user_distances,
                y=doses_N,
                mode='markers',
                name=f"Calculated N doses",
                marker=dict(symbol='circle', size=10, color=color_N),
                hoverinfo='text',
                text=[f"Distance: {d} m<br> Dose N: {dose:.3e} Gy" for d, dose in zip(user_distances, doses_N)],
                legendgroup="Interpolated N"
            ))
    
        if params_P:
            fig.add_trace(go.Scatter(
                x=user_distances,
                y=doses_P,
                mode='markers',
                name=f"Calculated P doses",
                marker=dict(symbol='circle', size=10, color=color_P),
                hoverinfo='text',
                text=[f"Distance: {d} m<br> Dose P: {dose:.3e} Gy" for d, dose in zip(user_distances, doses_P)],
                legendgroup="Interpolated P"
            ))
    
    # 🔹 Affichage du graphique mis à jour avec les points de doses calculés
    st.plotly_chart(fig, use_container_width=True)


    # Affichage des paramètres interpolés
    if params_N:
        st.write(f"**Interpolated parameters for Neutrons (N):**")
        st.write(f"A = {params_N['A']:.4e} ± {params_N['A_uncertainty']:.4e}, k = {params_N['k']:.4f} ± {params_N['k_uncertainty']:.4f}, b = {params_N['b']:.4e} ± {params_N['b_uncertainty']:.4e}")

    if params_P:
        st.write(f"**Interpolated parameters for Photons (P):**")
        st.write(f"A = {params_P['A']:.4e} ± {params_P['A_uncertainty']:.4e}, k = {params_P['k']:.4f} ± {params_P['k_uncertainty']:.4f}, b = {params_P['b']:.4e} ± {params_P['b_uncertainty']:.4e}")

with tab2:
    # Formatage des colonnes spécifiques
    formatted_data = visu_data.style.format({
         "1s uncertainty": "{:.2%}",  # Format en pourcentage avec 2 décimales
         "Dose (Gy)": "{:.2e}",       
         "Absolute Uncertainty": "{:.2e}"  
         })
    st.dataframe(formatted_data, hide_index=True)