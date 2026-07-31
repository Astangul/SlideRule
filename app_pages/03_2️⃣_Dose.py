import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from utils.plot_func_st import dose_scatter_plot_3, hex_to_rgba, hex_to_complementary_rgba

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
    df = pd.read_excel("./Database/All-at-once_DB.xlsx", sheet_name=sheet_name)
    if "Dose" in df.columns and "Dose (Gy)" not in df.columns:
        df = df.rename(columns={"Dose": "Dose (Gy)"})
    return df

# ______________________________________________________________________________________________________________________
# Physical Coupled + Skyshine model (alternative to the log-linear A/k/b interpolation)
#
# D(r,T) = A.r^-p . exp(-(mu_air+mu_T.T).r) . exp(-s1.T-s2.T^2)          [direct]
#        + B.r^-n . exp(-mu_sky.r) . exp(-mu_s2.T)                       [skyshine]
#
# All thickness coefficients (mu_T, s1, s2, mu_s2) are bounded >= 0, which guarantees
# the predicted dose decreases monotonically with shield thickness at any distance
# (d(dose)/dT <= 0). Unlike a plain log-linear interpolation of per-thickness fit
# parameters, this makes the model safe to evaluate at a thickness outside the
# range that was actually calculated by the transport code.
PHYSICAL_MODEL_LO = np.array([0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0])
PHYSICAL_MODEL_HI = np.array([np.inf, 3, 0.2, 0.01, 1, 0.05, np.inf, 2.5, 0.05, 1])

def predict_physical_coupled(params, r, T):
    A, p, mu_air, mu_T, s1, s2, B, n_sky, mu_sky, mu_s2 = params
    direct = A * r**(-p) * np.exp(-(mu_air + mu_T*T)*r) * np.exp(-s1*T - s2*T**2)
    sky = B * r**(-n_sky) * np.exp(-mu_sky*r) * np.exp(-mu_s2*T)
    return direct + sky

@st.cache_data
def fit_physical_coupled(fissile, case, screen, particle):
    """
    Fit the Physical Coupled + Skyshine model on the raw (unscaled, i.e. at the
    1e17-fission reference) 'final' data for one (Fissile, Case, Screen, Particle)
    selection, using all available thicknesses (including the bare/None case).
    Cached per selection since the fit does not depend on the fission count
    (the fission multiplier is a pure amplitude scaling applied at prediction time).

    Returns (popt, perr, pcov, R2, n_points, thickness_range) or None if unavailable.
    """
    raw = load_data('final')
    subset = raw[
        (raw["Fissile"] == fissile)
        & (raw["Case"] == case)
        & (raw["Screen"].isin(["None", screen]))
        & (raw["Particle"] == particle)
    ].dropna(subset=["Dose (Gy)", "Distance (m)", "Thickness (cm)"])
    subset = subset[subset["Dose (Gy)"] > 0]

    if len(subset) < 15 or subset["Thickness (cm)"].nunique() < 3:
        return None

    r_data = subset["Distance (m)"].values.astype(float)
    T_data = subset["Thickness (cm)"].values.astype(float)
    D_data = subset["Dose (Gy)"].values.astype(float)
    ln_D = np.log(D_data)

    def log_model(vars, A, p, mu_air, mu_T, s1, s2, B, n_sky, mu_sky, mu_s2):
        r, T = vars
        direct = A * r**(-p) * np.exp(-(mu_air + mu_T*T)*r) * np.exp(-s1*T - s2*T**2)
        sky = B * r**(-n_sky) * np.exp(-mu_sky*r) * np.exp(-mu_s2*T)
        return np.log(np.maximum(direct + sky, 1e-300))

    initial_guess = [1.0, 2.0, 0.03, 0.0, 0.1, 0.001, 1e-3, 1.2, 0.005, 0.02]
    try:
        popt, pcov = curve_fit(log_model, (r_data, T_data), ln_D, p0=initial_guess,
                              bounds=(PHYSICAL_MODEL_LO, PHYSICAL_MODEL_HI), maxfev=100000)
        perr = np.sqrt(np.diag(pcov))
        ln_D_fit = log_model((r_data, T_data), *popt)
        ss_res = np.sum((ln_D - ln_D_fit)**2)
        ss_tot = np.sum((ln_D - np.mean(ln_D))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        thickness_range = (float(T_data.min()), float(T_data.max()))
        return popt, perr, pcov, R2, len(subset), thickness_range
    except Exception:
        return None

def predict_physical_coupled_sigma(popt, pcov, r, T):
    """
    Propagate the fit's parameter covariance to a 1-sigma uncertainty on the
    predicted dose at each (r, T), via the delta method: sigma_D^2 = J . Cov . J^T,
    where J is the Jacobian of the dose w.r.t. the 10 parameters (estimated by
    central finite differences). Unlike naively evaluating the model at
    popt +/- perr (which ignores parameter correlations and is not guaranteed
    to bracket the central prediction), this band is centered on the central
    prediction by construction.
    """
    r = np.asarray(r, dtype=float)
    T = np.asarray(T, dtype=float)
    n_par = len(popt)
    J = np.zeros((len(r), n_par))
    for i in range(n_par):
        step = max(abs(popt[i]), 1e-6) * 1e-4
        p_hi = np.array(popt, dtype=float)
        p_lo = np.array(popt, dtype=float)
        p_hi[i] = min(popt[i] + step, PHYSICAL_MODEL_HI[i])
        p_lo[i] = max(popt[i] - step, PHYSICAL_MODEL_LO[i])
        denom = p_hi[i] - p_lo[i]
        if denom <= 0:
            continue
        J[:, i] = (predict_physical_coupled(p_hi, r, T) - predict_physical_coupled(p_lo, r, T)) / denom
    variance = np.einsum('ij,jk,ik->i', J, pcov, J)
    return np.sqrt(np.maximum(variance, 0.0))

data = load_data('final')

# Création des valeurs pour le select_slider
values = []
for exponent in range(13, 24):  # For 1.0E13 to 9.9E23
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
# persist_state="session" (Streamlit >= 1.59) conserve la valeur du widget
# pour toute la session, y compris lors des changements de page.
st.sidebar.select_slider(
    'Select the number of fissions:',
    options=options,
    key="fission_slider",
    on_change=update_fission_input,
    persist_state="session"
)

# **Calculer dynamiquement le pas**
current_value = st.session_state.get('fission_input', 1e17)
exponent = np.floor(np.log10(current_value))
step = (10 ** exponent) * 0.1

# Créer un number_input pour permettre à l'utilisateur d'entrer manuellement la valeur
# (pas de `value=` ici : la clé "fission_input" a déjà une valeur en session_state,
# lui passer aussi `value=` déclenche l'avertissement "widget created with a default
# value but also had its value set via the Session State API")
fissions_number_input = st.sidebar.number_input(
    "OR enter the number of fissions",
    min_value=1.0e+13,
    max_value=9.9e+23,
    step=step,
    format="%.1e",
    key="fission_input",
    on_change=update_fission_slider,
    persist_state="session"
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

    # 🔹 Permettre à l'utilisateur d'entrer des distances spécifiques pour calculer la dose
    user_distances_input = st.sidebar.text_input("Enter distances (semicolon-separated, in meters):", "10; 50; 100; 500; 1000")
    st.sidebar.divider()
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

    dose_method_label = st.sidebar.radio(
        "Dose calculation method:",
        options=[
            "Log-linear interpolation (A, k, b)",
            "Physical Coupled + Skyshine (model fit)",
        ],
        index=0,
        help=(
            "**Log-linear interpolation** (default): interpolates pre-fitted A, k, b "
            "parameters across the available thicknesses. Simple and fast, but not "
            "guaranteed to behave physically outside the calculated thickness range.\n\n"
            "**Physical Coupled + Skyshine**: fits a single model on all available "
            "thicknesses at once. Guaranteed monotonically decreasing with thickness "
            "at any distance, so it stays physical even when extrapolating to a "
            "thickness beyond the calculated range."
        ),
    )
    dose_method = "interp" if dose_method_label.startswith("Log-linear") else "physical"

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
    show_threshold = st.checkbox("Set total dose threshold", value=False)
    threshold_slider_placeholder = st.empty() if show_threshold else None

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

    dose_min = float(plot_df["Dose (Gy)"].min())
    dose_max = float(plot_df["Dose (Gy)"].max())

    dose_threshold = None
    intersection_distance = None
    intersection_placeholder = None
    if show_threshold:
        default_threshold = dose_min * 50
        dose_threshold = threshold_slider_placeholder.number_input(
            "Dose threshold (Gy)",
            min_value=dose_min,
            max_value=dose_max,
            value=default_threshold,
            step=(dose_max - dose_min) / 100 if dose_max > dose_min else 0.01,
            format="%.2e",
            key="dose_threshold_input",
        )
        intersection_placeholder = st.empty()

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

    def get_dose_predictors(particle, T_new, method):
        """
        Returns (predict, predict_upper, predict_lower, ok, info) for the given
        particle/thickness, where predict* are callables mapping an array of
        distances (m) to dose (Gy) already scaled by the fission multiplier.
        Dispatches between the two available methods.
        """
        if method == "physical":
            fit_result = fit_physical_coupled(selected_fissile, selected_case, selected_screen, particle)
            if fit_result is None:
                st.warning(f"Not enough data to fit the Physical Coupled model for Particle {particle}. Skipping.")
                return None, None, None, False, None

            popt, perr, pcov, R2, n_points, (T_min, T_max) = fit_result
            if T_new < T_min or T_new > T_max:
                st.caption(
                    f"ℹ️ T={T_new:.0f} cm is outside the calculated range ({T_min:.0f}-{T_max:.0f} cm) "
                    f"for Particle {particle}: extrapolating with the physical model "
                    f"(guaranteed monotonically decreasing with thickness)."
                )

            def predict(d):
                d = np.asarray(d, dtype=float)
                return predict_physical_coupled(popt, d, np.full_like(d, T_new)) * dose_multiplier

            def predict_sigma(d):
                # 1-sigma uncertainty on the dose, via delta-method propagation of the
                # full parameter covariance (centered on `predict` by construction).
                d = np.asarray(d, dtype=float)
                return predict_physical_coupled_sigma(popt, pcov, d, np.full_like(d, T_new)) * dose_multiplier

            def predict_upper(d):
                return predict(d) + predict_sigma(d)

            def predict_lower(d):
                center = predict(d)
                # Keep strictly positive (needed for the log-scale plot) even when
                # the 1-sigma uncertainty exceeds the central value.
                return np.maximum(center - predict_sigma(d), center * 1e-6)

            info = {
                "popt": popt, "perr": perr, "pcov": pcov, "R2": R2, "n_points": n_points,
                "T_range": (T_min, T_max), "predict_sigma": predict_sigma,
            }
            return predict, predict_upper, predict_lower, True, info

        params = interpolate_parameters(filtered_curve_fit_data, particle, T_new)
        if not params:
            return None, None, None, False, None
        predict = lambda d: calculate_interpolated_dose(d, params["A"], params["k"], params["b"])
        predict_upper = lambda d: calculate_interpolated_dose(
            d, params["A"] + params["A_uncertainty"], params["k"] + params["k_uncertainty"], params["b"] + params["b_uncertainty"]
        )
        predict_lower = lambda d: calculate_interpolated_dose(
            d, params["A"] - params["A_uncertainty"], params["k"] - params["k_uncertainty"], params["b"] - params["b_uncertainty"]
        )
        return predict, predict_upper, predict_lower, True, params

    color_N = "#9400D3"   # Violet profond pour la courbe interpolée des Neutrons (N)
    color_P = "#FF4500"  # Orange foncé pour la courbe interpolée des Photons (P)

    x_values = np.logspace(np.log10(1), np.log10(1200), 100)

    # 🔹 Ajout des courbes pour Neutrons (N)
    predict_N, predict_upper_N, predict_lower_N, ok_N, info_N = get_dose_predictors("N", T_new, dose_method)
    if ok_N:
        y_values_N = predict_N(x_values)
        y_values_upper_N = predict_upper_N(x_values)
        y_values_lower_N = predict_lower_N(x_values)

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

    # Courbe pour les Photons (P)
    predict_P, predict_upper_P, predict_lower_P, ok_P, info_P = get_dose_predictors("P", T_new, dose_method)
    if ok_P:
        y_values_P = predict_P(x_values)
        y_values_upper_P = predict_upper_P(x_values)
        y_values_lower_P = predict_lower_P(x_values)

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
    # Vérifier si les courbes N et P existent avant de créer la somme
    if ok_N and ok_P:
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

        # ------------------------------------------------------------------
        # Threshold line and intersection with total dose
        if show_threshold and dose_threshold is not None:
            dose_min = float(min(y_values_total))
            dose_max = float(max(y_values_total))
            intersection_distance = None
            if dose_min <= dose_threshold <= dose_max:
                intersection_distance = float(
                    np.interp(
                        dose_threshold,
                        y_values_total[::-1],
                        x_values[::-1],
                    )
                )
                fig.add_vline(
                    x=intersection_distance,
                    line=dict(color="green", dash="dot"),
                )
            fig.add_hline(
                y=dose_threshold,
                line=dict(color="green", dash="dash"),
            )
    
    # ------------------------------------------------------------------
    if intersection_placeholder and intersection_distance is not None:
        intersection_placeholder.write(
            f"Distance at threshold: {intersection_distance:.1f} m"
        )
    # ------------------------------------------------------------------
    # 🔸 Calculer les doses aux distances spécifiées
    if user_distances:
        doses_N = list(predict_N(np.array(user_distances))) if ok_N else []
        doses_P = list(predict_P(np.array(user_distances))) if ok_P else []

        # Incertitude 1-sigma sur la dose (uniquement disponible pour le modèle physique)
        if dose_method == "physical":
            doses_N_sigma = list(info_N["predict_sigma"](np.array(user_distances))) if ok_N else []
            doses_P_sigma = list(info_P["predict_sigma"](np.array(user_distances))) if ok_P else []

        # 🔹 Ajouter les marqueurs 🟢 sur les courbes calculées
        if show_components and ok_N:
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

        if show_components and ok_P:
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
        if user_distances and ok_N and ok_P:
            doses_total = [doses_N[i] + doses_P[i] for i in range(len(user_distances))]
            if dose_method == "physical":
                # Combinaison quadratique : fits N et P indépendants
                doses_total_sigma = list(np.sqrt(np.array(doses_N_sigma)**2 + np.array(doses_P_sigma)**2))

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
    st.plotly_chart(fig, width='stretch')
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
        df_doses_data = {
            "Distance (m)": user_distances,
            "Dose Neutrons (Gy)": doses_N if ok_N else [None] * len(user_distances),
            "Dose Photons (Gy)": doses_P if ok_P else [None] * len(user_distances),
            "Total Dose (Gy)": doses_total if (ok_N and ok_P) else [None] * len(user_distances)
        }
        format_dict = {
            "Distance (m)": "{:.1f}",
            "Dose Neutrons (Gy)": "{:.2e}",
            "Dose Photons (Gy)": "{:.2e}",
            "Total Dose (Gy)": "{:.2e}"
        }
        # Incertitude 1-sigma : uniquement disponible pour le modèle physique
        if dose_method == "physical":
            df_doses_data["± σ Neutrons (Gy)"] = doses_N_sigma if ok_N else [None] * len(user_distances)
            df_doses_data["± σ Photons (Gy)"] = doses_P_sigma if ok_P else [None] * len(user_distances)
            df_doses_data["± σ Total (Gy)"] = doses_total_sigma if (ok_N and ok_P) else [None] * len(user_distances)
            format_dict.update({
                "± σ Neutrons (Gy)": "{:.2e}",
                "± σ Photons (Gy)": "{:.2e}",
                "± σ Total (Gy)": "{:.2e}",
            })

        df_doses = pd.DataFrame(df_doses_data)

        # Appliquer un format scientifique aux colonnes de dose
        formatted_doses = df_doses.style.format(format_dict)

        # Affichage du tableau des doses calculées
        table_title = "Interpolated Doses" if dose_method == "interp" else "Model-Predicted Doses (Physical Coupled)"
        st.header(table_title)
        st.dataframe(formatted_doses, hide_index=True)

    with st.expander("See explanation"):
        if dose_method == "interp":
            st.subheader("Equation used for interpolated dose calculation")
            st.latex(r"D = \frac{N_{\text{fissions}}}{10^{17}} \frac{A}{d^k} \cdot e^{-b \cdot d} \cdot ")
            st.caption(
                "The A, k, b parameters are fitted separately at each calculated thickness, then "
                "log-linearly interpolated (or extrapolated) to the requested thickness T."
            )

            # Vérifier si les paramètres interpolés existent
            if ok_N or ok_P:
                # Création du DataFrame pour les paramètres interpolés
                df_params = pd.DataFrame({
                    "Parameter": ["A", "k", "b"],
                    "Neutron Value": [
                        f"{info_N['A']:.3e} ± {info_N['A_uncertainty']:.3e}" if ok_N else "N/A",
                        f"{info_N['k']:.3f} ± {info_N['k_uncertainty']:.3f}" if ok_N else "N/A",
                        f"{info_N['b']:.3e} ± {info_N['b_uncertainty']:.3e}" if ok_N else "N/A"
                    ],
                    "Photon Value": [
                        f"{info_P['A']:.3e} ± {info_P['A_uncertainty']:.3e}" if ok_P else "N/A",
                        f"{info_P['k']:.3f} ± {info_P['k_uncertainty']:.3f}" if ok_P else "N/A",
                        f"{info_P['b']:.3e} ± {info_P['b_uncertainty']:.3e}" if ok_P else "N/A"
                    ]
                })

                # Affichage du tableau des paramètres interpolés
                st.subheader("Interpolated parameters")
                st.dataframe(df_params, hide_index=True)
        else:
            st.subheader("Equation used for the Physical Coupled + Skyshine model")
            st.latex(
                r"D(r,T) = \frac{N_{\text{fissions}}}{10^{17}} \left["
                r"A\,r^{-p}\,e^{-(\mu_{air}+\mu_T T)\,r}\,e^{-s_1 T - s_2 T^2}"
                r" + B\,r^{-n}\,e^{-\mu_{sky} r}\,e^{-\mu_{s2} T}\right]"
            )
            st.caption(
                "Fitted once on all available thicknesses at once (direct + skyshine components). "
                "The thickness coefficients (μ_T, s₁, s₂, μ_s2) are constrained ≥ 0, which guarantees "
                "the dose decreases monotonically with thickness at any distance — safe to evaluate "
                "even for a thickness T beyond the calculated range."
            )

            if ok_N or ok_P:
                param_names = ["A", "p", "μ_air", "μ_T", "s₁", "s₂", "B", "n_sky", "μ_sky", "μ_s2"]
                df_params = pd.DataFrame({
                    "Parameter": param_names,
                    "Neutron Value": [f"{v:.3e}" for v in info_N["popt"]] if ok_N else ["N/A"] * 10,
                    "Photon Value": [f"{v:.3e}" for v in info_P["popt"]] if ok_P else ["N/A"] * 10,
                })
                st.subheader("Fitted parameters")
                st.dataframe(df_params, hide_index=True)

                col_r2_n, col_r2_p = st.columns(2)
                with col_r2_n:
                    if ok_N:
                        st.metric("Neutron fit: log R²", f"{info_N['R2']:.5f}",
                                 help=f"Fitted on {info_N['n_points']} points, T in "
                                      f"[{info_N['T_range'][0]:.0f}, {info_N['T_range'][1]:.0f}] cm")
                with col_r2_p:
                    if ok_P:
                        st.metric("Photon fit: log R²", f"{info_P['R2']:.5f}",
                                 help=f"Fitted on {info_P['n_points']} points, T in "
                                      f"[{info_P['T_range'][0]:.0f}, {info_P['T_range'][1]:.0f}] cm")

    st.divider()

    st.header("Calulated Doses")
    st.dataframe(formatted_data, hide_index=True)
# ______________________________________________________________________________________________________________________
