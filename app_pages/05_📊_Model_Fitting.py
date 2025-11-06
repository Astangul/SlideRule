import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit

from utils.utils_func_st import df_multiselect_filters, load_data_with_filters
from utils.plot_func_st import generate_analogous_colors, hex_to_rgba

# ______________________________________________________________________________________________________________________
# Page configuration
st.title("📊 Dose Model Fitting")

st.warning("⚠️ Section in development (WIP)")

st.markdown("""
Fit analytical models to Monte Carlo dose calculation results and predict doses for configurations not yet calculated.
Select a model from the sidebar to begin.
""")

# ______________________________________________________________________________________________________________________
# Load data with caching
@st.cache_data
def load_data(sheet_name):
    return pd.read_excel("./Database/All-at-once_DB.xlsx", sheet_name=sheet_name)

# ______________________________________________________________________________________________________________________
# Skyshine Enhanced model function

def fit_skyshine_enhanced(r_data, T_data, D_data):
    """
    Enhanced Skyshine version with polynomial build-up in the shield
    
    Parameters (9):
    - A: Direct component amplitude
    - B: Skyshine component amplitude
    - mu_air: Attenuation coefficient in air (m⁻¹)
    - mu_sky: Skyshine attenuation coefficient (m⁻¹)
    - mu_pb: Attenuation coefficient in shield (cm⁻¹)
    - n_sky: Skyshine geometric exponent
    - c1: Linear build-up (cm⁻¹)
    - c2: Quadratic build-up (cm⁻²)
    - f_atten: Skyshine attenuation factor
    """
    r_data = np.array(r_data, dtype=float)
    T_data = np.array(T_data, dtype=float)
    D_data = np.array(D_data, dtype=float)
    
    valid = (~np.isnan(D_data)) & (D_data > 0) & (r_data > 0)
    r_data, T_data, D_data = r_data[valid], T_data[valid], D_data[valid]
    
    if len(D_data) == 0:
        return None, None, None, None, None, None, None, None, None, 0, [0]*9
    
    ln_D = np.log(D_data)
    
    A_init = 1.0
    B_init = 0.1
    mu_air_init = 0.03
    mu_sky_init = 0.005
    mu_pb_init = 0.1
    n_sky_init = 1.2
    c1_init, c2_init = 0.0, 0.0
    f_atten_init = 2.0
    
    initial_guess = [A_init, B_init, mu_air_init, mu_sky_init, mu_pb_init, 
                     n_sky_init, c1_init, c2_init, f_atten_init]
    
    def log_model(vars, A, B, mu_air, mu_sky, mu_pb, n_sky, c1, c2, f_atten):
        r, T = vars
        buildup_screen = 1 + c1*T + c2*T**2
        buildup_screen = np.maximum(buildup_screen, 0.1)
        
        D_direct = (A/r**2 * np.exp(-mu_air*r) * buildup_screen * np.exp(-mu_pb*T))
        D_skyshine = (B * r**(-n_sky) * np.exp(-mu_sky*r) * np.exp(-mu_pb*T/f_atten))
        
        D_total = D_direct + D_skyshine
        return np.log(np.maximum(D_total, 1e-100))
    
    try:
        popt, pcov = curve_fit(log_model, (r_data, T_data), ln_D,
                              p0=initial_guess, maxfev=100000,
                              bounds=([0, 0, 0, 0, 0, 0.5, -1, -0.5, 1], 
                                     [10, 1, 0.1, 0.05, 1, 2, 1, 0.5, 10]))
        
        A, B, mu_air, mu_sky, mu_pb, n_sky, c1, c2, f_atten = popt
        perr = np.sqrt(np.diag(pcov))
        
        ln_D_fit = log_model((r_data, T_data), *popt)
        ss_res = np.sum((ln_D - ln_D_fit)**2)
        ss_tot = np.sum((ln_D - np.mean(ln_D))**2)
        R2 = 1 - ss_res/ss_tot
        
        return A, B, mu_air, mu_sky, mu_pb, n_sky, c1, c2, f_atten, R2, list(perr)
    except Exception as e:
        st.error(f"Model fitting error: {e}")
        return None, None, None, None, None, None, None, None, None, 0, [0]*9

def predict_skyshine_enhanced(params, r, T):
    """
    Calculate the prediction of the Skyshine Enhanced model
    
    Args:
        params: Tuple of 9 parameters (A, B, mu_air, mu_sky, mu_pb, n_sky, c1, c2, f_atten)
        r: Distance(s) in meters
        T: Shield thickness(es) in cm
    
    Returns:
        Predicted dose in Gy
    """
    A, B, mu_air, mu_sky, mu_pb, n_sky, c1, c2, f_atten = params
    buildup = np.maximum(1 + c1*T + c2*T**2, 0.1)
    D_direct = (A/r**2 * np.exp(-mu_air*r) * buildup * np.exp(-mu_pb*T))
    D_sky = (B * r**(-n_sky) * np.exp(-mu_sky*r) * np.exp(-mu_pb*T/f_atten))
    return D_direct + D_sky

# ______________________________________________________________________________________________________________________
# Super Hybrid model function

def fit_super_hybrid(r_data, T_data, D_data):
    """
    Super Hybrid model combining variable exponent, polynomial build-up and exponential correction
    
    Parameters (9):
    - A: Amplitude
    - k0: Base geometric exponent
    - k1: Thickness-dependent geometric term (cm⁻¹)
    - mu_air: Attenuation coefficient in air (m⁻¹)
    - mu_pb: Attenuation coefficient in shield (cm⁻¹)
    - b1: Linear build-up (cm⁻¹)
    - b2: Quadratic build-up (cm⁻²)
    - eps: Correction amplitude
    - lambda_r: Correction characteristic length (m)
    """
    r_data = np.array(r_data, dtype=float)
    T_data = np.array(T_data, dtype=float)
    D_data = np.array(D_data, dtype=float)
    
    valid = (~np.isnan(D_data)) & (D_data > 0) & (r_data > 0)
    r_data, T_data, D_data = r_data[valid], T_data[valid], D_data[valid]
    
    if len(D_data) == 0:
        return None, None, None, None, None, None, None, None, None, 0, [0]*9
    
    ln_D = np.log(D_data)
    ln_r = np.log(r_data)
    
    # Initial estimate using linear regression
    Y = ln_D + 2*ln_r
    X = np.column_stack([np.ones_like(r_data), r_data, T_data, r_data*T_data])
    Beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    
    A_init = np.exp(Beta[0])
    mu_air_init = max(0.001, -Beta[1])
    mu_pb_init = max(0.01, -Beta[2])
    
    k0_init, k1_init = 1.8, -0.01
    b1_init, b2_init = 0.0, 0.0
    eps_init, lambda_init = -0.5, 100
    
    initial_guess = [A_init, k0_init, k1_init, mu_air_init, mu_pb_init, 
                     b1_init, b2_init, eps_init, lambda_init]
    
    def log_model(vars, A, k0, k1, mu_air, mu_pb, b1, b2, eps, lambda_r):
        r, T = vars
        k = k0 + k1*T
        k = np.clip(k, 0.5, 3)
        
        buildup = 1 + b1*T + b2*T**2
        buildup = np.maximum(buildup, 0.1)
        
        correction = 1 + eps * np.exp(-r/lambda_r)
        
        return (np.log(A) - k*np.log(r) + np.log(buildup)
                - mu_air*r - mu_pb*T + np.log(np.abs(correction)))
    
    try:
        popt, pcov = curve_fit(log_model, (r_data, T_data), ln_D,
                              p0=initial_guess, maxfev=100000,
                              bounds=([0, 0.5, -0.1, 0, 0, -2, -0.5, -5, 10], 
                                     [np.inf, 3, 0.1, 0.5, 1, 2, 0.5, 5, 2000]))
        
        A, k0, k1, mu_air, mu_pb, b1, b2, eps, lambda_r = popt
        perr = np.sqrt(np.diag(pcov))
        
        ln_D_fit = log_model((r_data, T_data), *popt)
        ss_res = np.sum((ln_D - ln_D_fit)**2)
        ss_tot = np.sum((ln_D - np.mean(ln_D))**2)
        R2 = 1 - ss_res/ss_tot
        
        return A, k0, k1, mu_air, mu_pb, b1, b2, eps, lambda_r, R2, list(perr)
    except Exception as e:
        st.error(f"Model fitting error: {e}")
        return None, None, None, None, None, None, None, None, None, 0, [0]*9

def predict_super_hybrid(params, r, T):
    """
    Calculate the prediction of the Super Hybrid model
    
    Args:
        params: Tuple of 9 parameters (A, k0, k1, mu_air, mu_pb, b1, b2, eps, lambda_r)
        r: Distance(s) in meters
        T: Shield thickness(es) in cm
    
    Returns:
        Predicted dose in Gy
    """
    A, k0, k1, mu_air, mu_pb, b1, b2, eps, lambda_r = params
    k = np.clip(k0 + k1*T, 0.5, 3)
    buildup = np.maximum(1 + b1*T + b2*T**2, 0.1)
    correction = 1 + eps * np.exp(-r/lambda_r)
    return A * r**(-k) * np.exp(-mu_air*r) * buildup * np.exp(-mu_pb*T) * correction

# ______________________________________________________________________________________________________________________
# Hybrid Corrected model function

def fit_hybrid_corrected(r_data, T_data, D_data):
    """
    Hybrid Corrected model with exponential correction
    
    Parameters (8):
    - A: Amplitude
    - k: Geometric exponent
    - mu_air: Attenuation coefficient in air (m⁻¹)
    - mu_pb: Attenuation coefficient in shield (cm⁻¹)
    - c1: Linear build-up (cm⁻¹)
    - c2: Quadratic build-up (cm⁻²)
    - epsilon: Correction amplitude
    - lambda_r: Correction characteristic length (m)
    """
    r_data = np.array(r_data, dtype=float)
    T_data = np.array(T_data, dtype=float)
    D_data = np.array(D_data, dtype=float)
    
    valid = (~np.isnan(D_data)) & (D_data > 0) & (r_data > 0)
    r_data, T_data, D_data = r_data[valid], T_data[valid], D_data[valid]
    
    if len(D_data) == 0:
        return None, None, None, None, None, None, None, None, 0, [0]*8
    
    ln_D = np.log(D_data)
    ln_r = np.log(r_data)
    
    # Initial estimate using linear regression
    Y = ln_D + 2*ln_r
    X = np.column_stack([np.ones_like(r_data), r_data, T_data, T_data**2])
    Beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    
    A_init = np.exp(Beta[0])
    mu_air_init = max(0.001, -Beta[1])
    mu_pb_init = max(0.01, -Beta[2])
    
    initial_guess = [A_init, 1.8, mu_air_init, mu_pb_init, 0.1, 0.01, 0.1, 10]
    
    def log_model(vars, A, k, mu_air, mu_pb, c1, c2, epsilon, lambda_r):
        r, T = vars
        buildup = 1 + c1*T + c2*T**2
        buildup = np.maximum(buildup, 0.1)
        correction = 1 + epsilon * np.exp(-r/lambda_r)
        
        return (np.log(A) - k*np.log(r) + np.log(buildup) 
                - mu_air*r - mu_pb*T + np.log(correction))
    
    try:
        popt, pcov = curve_fit(log_model, (r_data, T_data), ln_D,
                              p0=initial_guess, maxfev=50000,
                              bounds=([0, 0.5, 0, 0, -2, -0.5, -1, 1], 
                                     [np.inf, 3, 1, 1, 2, 0.5, 1, 1000]))
        
        A, k, mu_air, mu_pb, c1, c2, eps, lambda_r = popt
        perr = np.sqrt(np.diag(pcov))
        
        ln_D_fit = log_model((r_data, T_data), *popt)
        ss_res = np.sum((ln_D - ln_D_fit)**2)
        ss_tot = np.sum((ln_D - np.mean(ln_D))**2)
        R2 = 1 - ss_res/ss_tot
        
        return A, k, mu_air, mu_pb, c1, c2, eps, lambda_r, R2, list(perr)
    except Exception as e:
        st.error(f"Model fitting error: {e}")
        return None, None, None, None, None, None, None, None, 0, [0]*8

def predict_hybrid_corrected(params, r, T):
    """
    Calculate the prediction of the Hybrid Corrected model
    
    Args:
        params: Tuple of 8 parameters (A, k, mu_air, mu_pb, c1, c2, epsilon, lambda_r)
        r: Distance(s) in meters
        T: Shield thickness(es) in cm
    
    Returns:
        Predicted dose in Gy
    """
    A, k, mu_air, mu_pb, c1, c2, epsilon, lambda_r = params
    buildup = np.maximum(1 + c1*T + c2*T**2, 0.1)
    correction = 1 + epsilon * np.exp(-r/lambda_r)
    return A * r**(-k) * np.exp(-mu_air*r) * buildup * np.exp(-mu_pb*T) * correction

# ______________________________________________________________________________________________________________________
# User interface

# Sidebar: Model selection
st.sidebar.header("⚙️ Model Selection")

# Dictionary of available models
AVAILABLE_MODELS = {
    "Skyshine Enhanced": {
        "name": "Skyshine Enhanced (9 parameters)",
        "description": "Combines direct and skyshine components with polynomial build-up",
        "params_count": 9,
        "icon": "",
        "performance": "R² ≈ 0.9998 | Error ≈ 5%"
    },
    "Super Hybrid": {
        "name": "Super Hybrid (9 parameters)",
        "description": "Ultimate combination with variable exponent, build-up and correction",
        "params_count": 9,
        "icon": "",
        "performance": "R² ≈ 0.9999 | Error ≈ 3.6%"
    },
    "Hybrid Corrected": {
        "name": "Hybrid Corrected (8 parameters)",
        "description": "Hybrid model with polynomial build-up and exponential correction",
        "params_count": 8,
        "icon": "",
        "performance": "R² ≈ 0.9895 | Error ≈ variable"
    },
}

# Model selection radio button
selected_model = st.sidebar.radio(
    "Choose a fitting model:",
    options=list(AVAILABLE_MODELS.keys()),
    format_func=lambda x: f"{AVAILABLE_MODELS[x]['icon']} {AVAILABLE_MODELS[x]['name']}",
    index=0
)

# Load "wall" data
data = load_data('wall')

# Default values for filters
default_columns = ['Fissile', 'Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)']
default_values = {
    "Fissile": "__first__",  
    "Case": "__first__",
    "Code": "__first__",
    "Particle": "__first__",
    "Screen": ["None", "Concrete"],
    "Thickness (cm)": "__all__"
}

# Update data with uncertainty calculation
data["Absolute Uncertainty"] = data["Dose (Gy)"] * data["1s uncertainty"]

# Default color palette definition
default_colors = ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72']

# Initialize custom color palette in session_state
if 'custom_colors_skyshine' not in st.session_state:
    st.session_state.custom_colors_skyshine = default_colors.copy()

colors = st.session_state.custom_colors_skyshine

# ______________________________________________________________________________________________________________________
# Data selection for fitting

st.header("1️⃣ Data Selection for Fitting")

with st.expander("Choose data to fit (click to expand/collapse)", expanded=True):
    fit_data, fit_filters = df_multiselect_filters(
        data, 
        default_columns=default_columns, 
        default_values=default_values, 
        key="fit_data"
    )

# Check if there is data
if len(fit_data) == 0:
    st.warning("⚠️ No data selected. Please adjust the filters.")
    st.stop()

# Check if there is at least one non-zero thickness
if fit_data['Thickness (cm)'].max() == 0:
    st.warning("⚠️ No non-zero shield thickness in the selected data. The model requires shielded data.")
    st.stop()

# Extract arrays for fitting
r_fit = fit_data['Distance (m)'].values
T_fit = fit_data['Thickness (cm)'].values
D_fit = fit_data['Dose (Gy)'].values

st.success(f"✅ {len(fit_data)} points selected for fitting")
st.write(f"- Distance: {r_fit.min():.0f} - {r_fit.max():.0f} m")
st.write(f"- Thickness: {T_fit.min():.0f} - {T_fit.max():.0f} cm")
st.write(f"- Dose: {D_fit.min():.2e} - {D_fit.max():.2e} Gy")
st.write(f"- Unique thicknesses: {sorted(fit_data['Thickness (cm)'].unique())} cm")

# ______________________________________________________________________________________________________________________
# Model fitting

st.header(f"2️⃣ Model Fitting")

# Display which model is being used
st.info(f"📌 **Current model**: {AVAILABLE_MODELS[selected_model]['icon']} {AVAILABLE_MODELS[selected_model]['name']}")

if st.button(f"🔄 Run {selected_model} Fitting", type="primary"):
    with st.spinner("Fitting in progress..."):
        # Route to the appropriate fitting function based on selected model
        if selected_model == "Skyshine Enhanced":
            result = fit_skyshine_enhanced(r_fit, T_fit, D_fit)
        elif selected_model == "Super Hybrid":
            result = fit_super_hybrid(r_fit, T_fit, D_fit)
        elif selected_model == "Hybrid Corrected":
            result = fit_hybrid_corrected(r_fit, T_fit, D_fit)
        # Add more models here as they are implemented
        else:
            st.error(f"Model '{selected_model}' not yet implemented.")
            result = None
        
        if result[0] is not None:
            params = result[:-2]
            R2 = result[-2]
            errors = result[-1]
            
            # Store results in session_state with model-specific keys
            st.session_state[f'{selected_model}_params'] = params
            st.session_state[f'{selected_model}_R2'] = R2
            st.session_state[f'{selected_model}_errors'] = errors
            st.session_state[f'{selected_model}_fit_done'] = True
            st.session_state['current_fitted_model'] = selected_model
            
            st.success(f"✅ Fitting successful! R² = {R2:.6f}")
        else:
            st.error("❌ Fitting failed. Try with different data.")

# Model-specific documentation
with st.expander(f"ℹ️ About the {selected_model} Model", expanded=False):
    if selected_model == "Skyshine Enhanced":
        st.markdown(r"""
        ## 📐 Model Equation
        
        The **Skyshine Enhanced** model combines two components:
        
        $$D_{\text{total}}(r, T) = D_{\text{direct}}(r, T) + D_{\text{skyshine}}(r, T)$$
        
        **Direct component** (with build-up in the shield):
        $$D_{\text{direct}} = \frac{A}{r^2} \cdot e^{-\mu_{\text{air}} \cdot r} \cdot B_{\text{screen}}(T) \cdot e^{-\mu_{\text{screen}} \cdot T}$$
        
        where the **build-up factor** is:
        $$B_{\text{screen}}(T) = 1 + c_1 \cdot T + c_2 \cdot T^2$$
        
        **Skyshine component** (atmospheric scattering):
        $$D_{\text{skyshine}} = \frac{B}{r^{n_{\text{sky}}}} \cdot e^{-\mu_{\text{sky}} \cdot r} \cdot e^{-\mu_{\text{screen}} \cdot T / f_{\text{atten}}}$$
        
        ## 🔧 Model Parameters (9)
        
        | Parameter | Description | Unit |
        |-----------|-------------|------|
        | `A` | Direct component amplitude | - |
        | `B` | Skyshine component amplitude | - |
        | `μ_air` | Attenuation coefficient in air | m⁻¹ |
        | `μ_sky` | Skyshine attenuation coefficient | m⁻¹ |
        | `μ_screen` | Attenuation coefficient in shield | cm⁻¹ |
        | `n_sky` | Skyshine geometric exponent | - |
        | `c₁` | Linear build-up | cm⁻¹ |
        | `c₂` | Quadratic build-up | cm⁻² |
        | `f_atten` | Skyshine attenuation factor | - |
        """)
    elif selected_model == "Super Hybrid":
        st.markdown(r"""
        ## 📐 Model Equation
        
        The **Super Hybrid** model combines variable exponent, polynomial build-up and exponential correction:
        
        $$D(r, T) = A \cdot r^{-k} \cdot e^{-\mu_{\text{air}} \cdot r} \cdot B_{\text{screen}}(T) \cdot e^{-\mu_{\text{screen}} \cdot T} \cdot C(r)$$
        
        where:
        
        **Variable geometric exponent**:
        $$k = k_0 + k_1 \cdot T$$
        (constrained to [0.5, 3])
        
        **Polynomial build-up factor**:
        $$B_{\text{screen}}(T) = 1 + b_1 \cdot T + b_2 \cdot T^2$$
        
        **Exponential correction**:
        $$C(r) = 1 + \varepsilon \cdot e^{-r / \lambda_r}$$
        
        ## 🔧 Model Parameters (9)
        
        | Parameter | Description | Unit |
        |-----------|-------------|------|
        | `A` | Amplitude | - |
        | `k₀` | Base geometric exponent | - |
        | `k₁` | Thickness-dependent geometric term | cm⁻¹ |
        | `μ_air` | Attenuation in air | m⁻¹ |
        | `μ_screen` | Attenuation in shield | cm⁻¹ |
        | `b₁` | Linear build-up | cm⁻¹ |
        | `b₂` | Quadratic build-up | cm⁻² |
        | `ε` | Correction amplitude | - |
        | `λ_r` | Correction characteristic length | m |
        """)
    elif selected_model == "Hybrid Corrected":
        st.markdown(r"""
        ## 📐 Model Equation
        
        The **Hybrid Corrected** model combines fixed exponent, polynomial build-up and exponential correction:
        
        $$D(r, T) = A \cdot r^{-k} \cdot e^{-\mu_{\text{air}} \cdot r} \cdot B_{\text{screen}}(T) \cdot e^{-\mu_{\text{screen}} \cdot T} \cdot C(r)$$
        
        where:
        
        **Polynomial build-up factor**:
        $$B_{\text{screen}}(T) = \max(1 + c_1 \cdot T + c_2 \cdot T^2, 0.1)$$
        
        **Exponential correction**:
        $$C(r) = 1 + \varepsilon \cdot e^{-r / \lambda_r}$$
        
        ## 🔧 Model Parameters (8)
        
        | Parameter | Description | Unit |
        |-----------|-------------|------|
        | `A` | Amplitude | - |
        | `k` | Fixed geometric exponent | - |
        | `μ_air` | Attenuation in air | m⁻¹ |
        | `μ_screen` | Attenuation in shield | cm⁻¹ |
        | `c₁` | Linear build-up | cm⁻¹ |
        | `c₂` | Quadratic build-up | cm⁻² |
        | `ε` | Correction amplitude | - |
        | `λ_r` | Correction characteristic length | m |
        """)
    else:
        st.info(f"Documentation for '{selected_model}' model coming soon.")

# Display fitting results if they exist for the selected model
if f'{selected_model}_fit_done' in st.session_state and st.session_state[f'{selected_model}_fit_done']:
    params = st.session_state[f'{selected_model}_params']
    R2 = st.session_state[f'{selected_model}_R2']
    errors = st.session_state[f'{selected_model}_errors']
    
    # Model Parameters in expander
    with st.expander("📊 Fitted Model Parameters", expanded=True):
        # Define parameter information based on selected model
        if selected_model == "Skyshine Enhanced":
            param_names = ['A', 'B', 'μ_air', 'μ_sky', 'μ_screen', 'n_sky', 'c₁', 'c₂', 'f_atten']
            param_units = ['-', '-', 'm⁻¹', 'm⁻¹', 'cm⁻¹', '-', 'cm⁻¹', 'cm⁻²', '-']
            param_descriptions = [
                'Direct component amplitude',
                'Skyshine component amplitude',
                'Attenuation in air',
                'Skyshine attenuation',
                'Attenuation in shield',
                'Skyshine geometric exponent',
                'Linear build-up',
                'Quadratic build-up',
                'Skyshine attenuation factor'
            ]
        elif selected_model == "Super Hybrid":
            param_names = ['A', 'k₀', 'k₁', 'μ_air', 'μ_screen', 'b₁', 'b₂', 'ε', 'λ_r']
            param_units = ['-', '-', 'cm⁻¹', 'm⁻¹', 'cm⁻¹', 'cm⁻¹', 'cm⁻²', '-', 'm']
            param_descriptions = [
                'Amplitude',
                'Base geometric exponent',
                'Thickness-dependent geometric term',
                'Attenuation in air',
                'Attenuation in shield',
                'Linear build-up',
                'Quadratic build-up',
                'Correction amplitude',
                'Correction characteristic length'
            ]
        elif selected_model == "Hybrid Corrected":
            param_names = ['A', 'k', 'μ_air', 'μ_screen', 'c₁', 'c₂', 'ε', 'λ_r']
            param_units = ['-', '-', 'm⁻¹', 'cm⁻¹', 'cm⁻¹', 'cm⁻²', '-', 'm']
            param_descriptions = [
                'Amplitude',
                'Fixed geometric exponent',
                'Attenuation in air',
                'Attenuation in shield',
                'Linear build-up',
                'Quadratic build-up',
                'Correction amplitude',
                'Correction characteristic length'
            ]
        else:
            param_names = [f'p{i}' for i in range(len(params))]
            param_units = ['-'] * len(params)
            param_descriptions = ['Parameter'] * len(params)
        
        # Create a DataFrame to display parameters
        param_df = pd.DataFrame({
            'Parameter': param_names,
            'Value': [f'{p:.4e}' for p in params],
            'Uncertainty': [f'{e:.2e}' for e in errors],
            'Unit': param_units,
            'Description': param_descriptions
        })
        
        st.dataframe(param_df, hide_index=True, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Logarithmic R² (ln)", f"{R2:.6f}")
        with col2:
            # Calculate median relative error using the appropriate prediction function
            if selected_model == "Skyshine Enhanced":
                D_pred = predict_skyshine_enhanced(params, r_fit, T_fit)
            elif selected_model == "Super Hybrid":
                D_pred = predict_super_hybrid(params, r_fit, T_fit)
            elif selected_model == "Hybrid Corrected":
                D_pred = predict_hybrid_corrected(params, r_fit, T_fit)
            else:
                D_pred = np.zeros_like(D_fit)
            
            # Calculate relative error with robustness checks
            # Filter out invalid predictions (negative, zero, NaN, inf)
            valid_mask = (D_pred > 0) & (D_fit > 0) & np.isfinite(D_pred) & np.isfinite(D_fit)
            
            if np.sum(valid_mask) > 0:
                relative_errors = np.abs((D_fit[valid_mask] - D_pred[valid_mask]) / D_fit[valid_mask]) * 100
                median_error = np.median(relative_errors)
                
                # Warning if some predictions are invalid
                n_invalid = len(D_fit) - np.sum(valid_mask)
                if n_invalid > 0:
                    st.metric("Median relative error", f"{median_error:.1f}%", 
                             delta=f"⚠️ {n_invalid} invalid predictions", delta_color="off")
                else:
                    st.metric("Median relative error", f"{median_error:.1f}%")
            else:
                st.metric("Median relative error", "N/A", 
                         delta="⚠️ All predictions invalid", delta_color="off")
        
        st.info("""
        💡 **About the metrics:**
        
        **Logarithmic R² (ln)**: For data spanning multiple orders of magnitude (10⁻⁷ to 10⁰ Gy), the logarithmic R² is essential. It gives equal weight to each decade, providing a more accurate assessment of model quality across the entire dose range than a linear R². The natural logarithm (ln) is used as it is standard in physics and mathematical modeling.
        
        **Median Relative Error**: Represents the typical prediction error of the model, calculated as the median of |( D_measured - D_predicted ) / D_measured| × 100%. This metric is robust to outliers and gives a realistic view of model performance. Values < 10% indicate excellent predictions, < 20% good predictions, and > 50% suggest the model may not be suitable for this configuration.
        """)
    
    # ______________________________________________________________________________________________________________________
    # Fit visualization
    
    st.header("3️⃣ Fit Visualization & Prediction")
    
    # Visualization and prediction options
    col_viz1, col_viz2, col_viz3, col_viz4 = st.columns(4)
    with col_viz1:
        log_x = st.toggle("Logarithmic X-axis", value=True, key="log_x_fit")
    with col_viz2:
        log_y = st.toggle("Logarithmic Y-axis", value=True, key="log_y_fit")
    with col_viz3:
        show_custom = st.toggle("Show custom thickness", value=False, key="show_custom_thickness")
    
    # Initialize custom_thickness with a default value
    custom_thickness = 15.0
    with col_viz4:
        if show_custom:
            custom_thickness = st.number_input(
                "Thickness (cm)",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                key="custom_thickness_input"
            )
    
    # Create the plot
    fig_fit = go.Figure()
    
    # Get unique thicknesses
    unique_T = sorted(fit_data['Thickness (cm)'].unique())
    
    # Use colors from the custom palette directly
    fit_colors = [colors[i % len(colors)] for i in range(len(unique_T))]
    
    for idx, T_val in enumerate(unique_T):
        mask = (fit_data['Thickness (cm)'] == T_val)
        r_subset = fit_data.loc[mask, 'Distance (m)'].values
        D_subset = fit_data.loc[mask, 'Dose (Gy)'].values
        unc_subset = fit_data.loc[mask, 'Absolute Uncertainty'].values
        
        color = fit_colors[idx]
        
        # If custom thickness is shown, make existing data more discrete
        if show_custom:
            alpha_marker = 0.4
            marker_size = 8
            line_width = 2
            line_dash = 'dot'
            visible = True  # Visible by default
        else:
            alpha_marker = 1.0
            marker_size = 10
            line_width = 3
            line_dash = 'solid'
            visible = True
        
        # Experimental points
        fig_fit.add_trace(go.Scatter(
            x=r_subset, 
            y=D_subset,
            mode='markers',
            marker=dict(size=marker_size, color=color, symbol='circle', 
                       line=dict(width=2, color='black'), opacity=alpha_marker),
            error_y=dict(type='data', array=2*unc_subset, visible=True, thickness=1.5, color=color),
            name=f'T={T_val} cm (data)',
            legendgroup=f'T{T_val}',
            showlegend=True,
            visible=visible
        ))
        
        # Model curve
        r_model = np.logspace(np.log10(r_subset.min()), np.log10(r_subset.max()), 200)
        
        # Use appropriate prediction function based on selected model
        if selected_model == "Skyshine Enhanced":
            D_model = predict_skyshine_enhanced(params, r_model, np.full_like(r_model, T_val))
        elif selected_model == "Super Hybrid":
            D_model = predict_super_hybrid(params, r_model, np.full_like(r_model, T_val))
        elif selected_model == "Hybrid Corrected":
            D_model = predict_hybrid_corrected(params, r_model, np.full_like(r_model, T_val))
        else:
            D_model = np.zeros_like(r_model)
        
        fig_fit.add_trace(go.Scatter(
            x=r_model,
            y=D_model,
            mode='lines',
            line=dict(width=line_width, color=color, dash=line_dash),
            name=f'T={T_val} cm (model)',
            legendgroup=f'T{T_val}',
            showlegend=True,
            visible=visible
        ))
    
    # Add custom thickness curve if requested
    if show_custom:
        r_custom = np.logspace(np.log10(r_fit.min()), np.log10(r_fit.max()), 300)
        
        # Use appropriate prediction function based on selected model
        if selected_model == "Skyshine Enhanced":
            D_custom = predict_skyshine_enhanced(params, r_custom, np.full_like(r_custom, custom_thickness))
        elif selected_model == "Super Hybrid":
            D_custom = predict_super_hybrid(params, r_custom, np.full_like(r_custom, custom_thickness))
        elif selected_model == "Hybrid Corrected":
            D_custom = predict_hybrid_corrected(params, r_custom, np.full_like(r_custom, custom_thickness))
        else:
            D_custom = np.zeros_like(r_custom)
        
        # Use a color from the palette for custom thickness (next available color after data series)
        custom_color = colors[len(unique_T) % len(colors)]
        
        fig_fit.add_trace(go.Scatter(
            x=r_custom,
            y=D_custom,
            mode='lines',
            line=dict(width=5, color=custom_color),
            name=f'T={custom_thickness} cm ⭐ (PREDICTION)',
            legendgroup='custom',
            showlegend=True
        ))
    
    # Layout
    title_text = f"{selected_model} Model Fit (R² ln = {R2:.6f})"
    if show_custom:
        title_text += f" + Prediction for T={custom_thickness} cm"
    
    fig_fit.update_layout(
        title=title_text,
        xaxis_title="Distance (m)" + (" [Log10]" if log_x else ""),
        yaxis_title="Dose (Gy) ± 2σ" + (" [Log10]" if log_y else ""),
        height=700,
        hovermode='x',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    if log_x:
        fig_fit.update_xaxes(type='log')
    if log_y:
        fig_fit.update_yaxes(type='log', tickformat='.2e')
    
    fig_fit.update_xaxes(showgrid=True, minor=dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))
    fig_fit.update_yaxes(showgrid=True, minor=dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))
    
    st.plotly_chart(fig_fit, width='stretch')
    
    # Display prediction table if custom thickness is shown
    if show_custom:
        with st.expander("📋 Predicted Values", expanded=True):
            st.markdown(f"""
            **Prediction for custom shield thickness: {custom_thickness} cm**
            
            Use the fitted model to predict the dose at a shield thickness that was not calculated by a transport code.
            """)
            
            # Get unique distances from the fitting data
            unique_distances = sorted(fit_data['Distance (m)'].unique())
            r_custom = np.array(unique_distances)
            
            # Use appropriate prediction function based on selected model
            if selected_model == "Skyshine Enhanced":
                D_custom = predict_skyshine_enhanced(params, r_custom, np.full_like(r_custom, custom_thickness))
            elif selected_model == "Super Hybrid":
                D_custom = predict_super_hybrid(params, r_custom, np.full_like(r_custom, custom_thickness))
            elif selected_model == "Hybrid Corrected":
                D_custom = predict_hybrid_corrected(params, r_custom, np.full_like(r_custom, custom_thickness))
            else:
                D_custom = np.zeros_like(r_custom)
            
            pred_table = pd.DataFrame({
                'Distance (m)': r_custom,
                'Predicted Dose (Gy)': D_custom
            })
            
            pred_table_styled = pred_table.style.format({
                "Distance (m)": "{:.0f}",
                "Predicted Dose (Gy)": "{:.2e}"
            })
            
            st.dataframe(pred_table_styled, hide_index=True, width='stretch')

else:
    st.info("👆 Click the 'Run Model Fitting' button to start the analysis.")

# ______________________________________________________________________________________________________________________
# Expander to customize colors
with st.expander("🎨 Customize Color Palette", expanded=False):
    st.write("Customize the colors used in the plots.")
    
    num_colors_to_show = 12
    cols = st.columns(6)
    
    for i in range(num_colors_to_show):
        with cols[i % 6]:
            new_color = st.color_picker(
                f"Color {i+1}",
                value=st.session_state.custom_colors_skyshine[i],
                key=f"color_picker_skyshine_{i}"
            )
            st.session_state.custom_colors_skyshine[i] = new_color
    
    col_reset1, col_reset2, col_reset3 = st.columns([1, 1, 4])
    with col_reset1:
        if st.button("🔄 Reset", help="Reset all colors to default"):
            st.session_state.custom_colors_skyshine = default_colors.copy()
            st.rerun()
    with col_reset2:
        if st.button("🎲 Randomize", help="Generate random colors"):
            import random
            st.session_state.custom_colors_skyshine = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(default_colors))]
            st.rerun()
