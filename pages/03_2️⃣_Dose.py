import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from utils.plot_func_st import dose_scatter_plot_3, hex_to_rgba, hex_to_complementary_rgba

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
main_body_logo_path = "./icons/Slide-Rule_orange.png"
st.logo(image = sidebar_logo_path, size="large", icon_image = main_body_logo_path)
st.warning('Section in development (WIP)')
# ______________________________________________________________________________________________________________________
# Récupérer le thème courant depuis le session_state
if "themes" in st.session_state:
    current_theme = st.session_state.themes["current_theme"]
    # Récupérer la couleur en hexadécimal (par exemple, le texte)
    hex_color = st.session_state.themes[current_theme]["theme.textColor"]
else:
    # Valeur par défaut si le thème n'est pas défini
    current_theme = "light"
    hex_color  = "#6c1d82"

total_curve_color_rgba=hex_to_complementary_rgba(hex_color, alpha=1.0)
total_fill_color_rgba = hex_to_complementary_rgba(hex_color, alpha=0.2)
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


st.sidebar.divider()
available_screens = [s for s in data["Screen"].unique() if s != "None"]
default_screen_index = available_screens.index("Concrete") if "Concrete" in available_screens else 0
selected_screen = st.sidebar.selectbox(
    "Select screen material",
    options=available_screens,
    index=default_screen_index,
)
T_new = st.sidebar.number_input(
    "Enter screen thickness (cm) for interpolation:",
    min_value=0.0,
    step=1.0,
    value=15.0,
)


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
        }

        # Sélection unique pour "Fissile"
        selected_fissile = st.selectbox("Select fissile material", options=data["Fissile"].unique(), index=0)

        # Sélection unique pour "Case"
        selected_case = st.selectbox("Select case", options=data["Case"].unique(), index=0)

        selected_screens = ["None", selected_screen]

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

    # Option d'affichage de la décomposition des particules
    show_components = st.checkbox("Display N and P components", value=False)

    # Calcul de la dose totale à partir des données calculées
    group_cols = [c for c in visu_data.columns if c not in [
        "Particle",
        "Dose (Gy)",
        "1s uncertainty",
        "Absolute Uncertainty",
    ]]
    # Filter groups to ensure both neutron and photon doses are present before
    # computing the total. Missing one component can lead to an artificial
    # drop in the total dose curve.
    required_particles = {"N", "P"}

    def has_complete_particles(group):
        """Return True if both N and P doses are present and not null."""
        valid = group.dropna(subset=["Dose (Gy)"])
        return required_particles.issubset(set(valid["Particle"]))

    complete_visu_data = (
        visu_data.groupby(group_cols)
        .filter(has_complete_particles)
    )
    total_visu_data = (
        complete_visu_data.groupby(group_cols, as_index=False)
        .agg({
            "Dose (Gy)": "sum",
            "Absolute Uncertainty": lambda x: np.sqrt((x ** 2).sum()),
        })
    )
    total_visu_data["1s uncertainty"] = total_visu_data["Absolute Uncertainty"] / total_visu_data["Dose (Gy)"]
    total_visu_data["Particle"] = "Total"
 
    # 🔹 Préparer les courbes à afficher
    if show_components:
        thickness_df = visu_data
    else:
        thickness_df = total_visu_data

    thicknesses = sorted(thickness_df["Thickness (cm)"].unique())
    highlight = []
    if thicknesses:
        lower = max([t for t in thicknesses if t <= T_new], default=None)
        upper = min([t for t in thicknesses if t >= T_new], default=None)
        if lower is None:
            highlight = [upper]
        elif upper is None:
            highlight = [lower]
        elif lower == upper:
            highlight = [lower]
        else:
            highlight = [lower, upper]

    if highlight:
        plot_df = thickness_df[thickness_df["Thickness (cm)"].isin(highlight)]
    else:
        plot_df = thickness_df

    fig = dose_scatter_plot_3(plot_df, visu_filters, colors)
    # ______________________________________________________________________________________________________________________
    df_curve_fit = load_data("curve_fit")
    data["Screen"] = data["Screen"].fillna("None")

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
        return A * distance**-k * np.exp(-b * distance) * dose_multiplier

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

    color_N = "#9400D3"   # Violet profond pour la courbe interpolée des Neutrons (N)
    color_P = "#FF4500"  # Orange foncé pour la courbe interpolée des Photons (P)
    
    # 🔹 Ajout des courbes interpolées pour Neutrons (N)
    params_N = interpolate_parameters(filtered_curve_fit_data, "N", T_new)
    if params_N:
        x_values = np.logspace(np.log10(1), np.log10(1200), 100)
        y_values_N = calculate_interpolated_dose(x_values, params_N["A"], params_N["k"], params_N["b"])
        y_values_upper_N = calculate_interpolated_dose(x_values, params_N["A"] + params_N["A_uncertainty"], params_N["k"] + params_N["k_uncertainty"], params_N["b"] + params_N["b_uncertainty"])
        y_values_lower_N = calculate_interpolated_dose(x_values, params_N["A"] - params_N["A_uncertainty"], params_N["k"] - params_N["k_uncertainty"], params_N["b"] - params_N["b_uncertainty"])

        if show_components:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values_N,
                mode='lines',  # ✅ LIGNE CONTINUE SANS MARQUEUR
                name=f"",
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

        if show_components:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values_P,
                mode='lines',
                name=f"",
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
    # Vérifier si les interpolations N et P existent avant de créer la somme
    if params_N and params_P:
        y_values_total = y_values_N + y_values_P
        y_values_upper_total = y_values_upper_N + y_values_upper_P
        y_values_lower_total = y_values_lower_N + y_values_lower_P

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_total,
            mode='lines',
            name=f"",
            legendgroup="Total Dose",
            line=dict(color=total_curve_color_rgba, dash="solid"),
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_upper_total,
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip',
            showlegend=False,
            legendgroup="Total Dose",
            fillcolor=total_fill_color_rgba
        ))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_lower_total,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            legendgroup="Total Dose",
            fillcolor=total_fill_color_rgba,
            hoverinfo='skip',
            showlegend=False
        ))


    # 🔹 Permettre à l'utilisateur d'entrer des distances spécifiques pour calculer la dose
    st.sidebar.divider()
    user_distances_input = st.sidebar.text_input("Enter distances (semicolon-separated, in meters):", "10; 50; 100; 500; 1000")    
    # 🔸 Convertir les distances entrées en une liste de valeurs numériques
    try:
        user_distances = [float(d.strip()) for d in user_distances_input.split(";") if d.strip()]
        user_distances = [d for d in user_distances if d > 0]  # Filtrer les valeurs négatives
    except ValueError:
        st.sidebar.error("Invalid input. Please enter semicolon-separated numeric values.")
        user_distances = []
    
    # Range check based on available data
    min_dist = visu_data["Distance (m)"].min()
    max_dist = visu_data["Distance (m)"].max()
    invalid_distances = [d for d in user_distances if d < min_dist or d > max_dist]
    if invalid_distances:
        st.warning(
            f"Distances {invalid_distances} are outside the valid range ({min_dist:.1f} - {max_dist:.1f} m)."
        )

    # 🔸 Calculer les doses aux distances spécifiées
    if user_distances:
        doses_N = [calculate_interpolated_dose(d, params_N["A"], params_N["k"], params_N["b"]) for d in user_distances] if params_N else []
        doses_P = [calculate_interpolated_dose(d, params_P["A"], params_P["k"], params_P["b"]) for d in user_distances] if params_P else []
     
        # 🔹 Ajouter les marqueurs 🟢 sur les courbes interpolées
        if show_components and params_N:

            x_values = np.logspace(np.log10(1), np.log10(1200), 100)
            y_values_N = calculate_interpolated_dose(x_values, params_N["A"], params_N["k"], params_N["b"])

            fig.add_trace(go.Scatter(
                x=user_distances,
                y=doses_N,
                mode='markers',
                name=f"Interpolated N ({T_new} cm)",
                marker=dict(symbol='star-square', size=11, color=color_N),    
                # text=[f"[N] {dose:.3e} Gy" for dose in doses_N],
                # text=[f"[N] {dose:.3e} Gy" for d, dose in zip(user_distances, doses_N)],
                legendgroup="Interpolated N"
            ))
    
        if show_components and params_P:
            y_values_P = calculate_interpolated_dose(x_values, params_P["A"], params_P["k"], params_P["b"])

            fig.add_trace(go.Scatter(
                x=user_distances,
                y=doses_P,
                mode='markers',
                name=f"Interpolated P ({T_new} cm)",
                marker=dict(symbol='star-square', size=11, color=color_P),
                # text=[f"[P] {dose:.3e} Gy" for dose in doses_P],
                # text=[f"[P] {dose:.3e} Gy" for d, dose in zip(user_distances, doses_P)],
                legendgroup="Interpolated P"
            ))
        
        # Calcul de la dose totale à chaque distance
        if user_distances and params_N and params_P:
            doses_total = [doses_N[i] + doses_P[i] for i in range(len(user_distances))]

            # Ajouter les marqueurs noirs pour la dose totale
            fig.add_trace(go.Scatter(
                x=user_distances,
                y=doses_total,
                mode='markers',
                name=f"Total Dose ({T_new} cm)",
                marker=dict(symbol='star-square', size=11, color=total_curve_color_rgba),
                legendgroup="Total Dose"
            ))

    fig.layout.update(hovermode="x")  # ✅ Mode de survol unifié
    # 🔹 Affichage du graphique mis à jour avec les points de doses calculés
    st.plotly_chart(fig, use_container_width=True)
    st.toggle("X-axis log scale", value=st.session_state.get("log_x_fig1", True), key="log_x_fig1")
    st.toggle("Y-axis log scale", value=st.session_state.get("log_y_fig1", True), key="log_y_fig1")
    

with tab2:
    # Formatage des colonnes spécifiques
    formatted_data = visu_data.style.format({
         "1s uncertainty": "{:.2%}",  # Format en pourcentage avec 2 décimales
         "Dose (Gy)": "{:.2e}",
         "Absolute Uncertainty": "{:.2e}"  
         })
    # st.header("Calulated Doses")
    # st.dataframe(formatted_data, hide_index=True)

    if user_distances:
        # Création du DataFrame pour les doses calculées
        df_doses = pd.DataFrame({
            "Distance (m)": user_distances,
            "Dose Neutrons (Gy)": doses_N if params_N else [None] * len(user_distances),
            "Dose Photons (Gy)": doses_P if params_P else [None] * len(user_distances),
            "Total Dose (Gy)": doses_total if (params_N and params_P) else [None] * len(user_distances)
        })

        # Appliquer un format scientifique aux colonnes de dose
        formatted_doses = df_doses.style.format({
            "Distance (m)": "{:.1f}",
            "Dose Neutrons (Gy)": "{:.2e}",
            "Dose Photons (Gy)": "{:.2e}",
            "Total Dose (Gy)": "{:.2e}"
        })

        # Affichage du tableau des doses interpolées
        st.header("Interpolated Doses")
        st.dataframe(formatted_doses, hide_index=True)      

    with st.expander("See explanation"):
        st.subheader("Equation used for interpolated dose calculation")
        st.latex(r"D = \frac{N_{\text{fissions}}}{10^{17}} \frac{A}{d^k} \cdot e^{-b \cdot d} \cdot ")

        # Vérifier si les paramètres interpolés existent
        if params_N or params_P:
            # Création du DataFrame pour les paramètres interpolés
            df_params = pd.DataFrame({
                "Parameter": ["A", "k", "b"],
                "Neutron Value": [
                    f"{params_N['A']:.3e} ± {params_N['A_uncertainty']:.3e}" if params_N else "N/A",
                    f"{params_N['k']:.3f} ± {params_N['k_uncertainty']:.3f}" if params_N else "N/A",
                    f"{params_N['b']:.3e} ± {params_N['b_uncertainty']:.3e}" if params_N else "N/A"
                ],
                "Photon Value": [
                    f"{params_P['A']:.3e} ± {params_P['A_uncertainty']:.3e}" if params_P else "N/A",
                    f"{params_P['k']:.3f} ± {params_P['k_uncertainty']:.3f}" if params_P else "N/A",
                    f"{params_P['b']:.3e} ± {params_P['b_uncertainty']:.3e}" if params_P else "N/A"
                ]
            })

            # Affichage du tableau des paramètres interpolés
            st.subheader("Interpolated parameters")
            st.dataframe(df_params, hide_index=True)
        
    st.divider()

    st.header("Calulated Doses")
    st.dataframe(formatted_data, hide_index=True)
# ______________________________________________________________________________________________________________________
