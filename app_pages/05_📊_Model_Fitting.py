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
    df = pd.read_excel("./Database/All-at-once_DB.xlsx", sheet_name=sheet_name)
    if "Dose" in df.columns and "Dose (Gy)" not in df.columns:
        df = df.rename(columns={"Dose": "Dose (Gy)"})
    return df

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
# Simple Exponential model function

def fit_simple_exponential(r_data, T_data, D_data):
    """
    Simple exponential model with geometric and exponential attenuation
    
    Parameters (4):
    - A: Amplitude
    - k: Geometric exponent
    - b: Attenuation coefficient in air (m⁻¹)
    - c: Attenuation coefficient in shield (cm⁻¹)
    
    Model: D(r,T) = A × r^(-k) × exp(-b×r) × exp(-c×T)
    """
    r_data = np.array(r_data, dtype=float)
    T_data = np.array(T_data, dtype=float)
    D_data = np.array(D_data, dtype=float)
    
    valid = (~np.isnan(D_data)) & (D_data > 0) & (r_data > 0)
    r_data, T_data, D_data = r_data[valid], T_data[valid], D_data[valid]
    
    if len(D_data) == 0:
        return None, None, None, None, 0, [0]*4
    
    ln_D = np.log(D_data)
    ln_r = np.log(r_data)
    
    # Initial estimate using linear regression
    Y = ln_D + 2*ln_r
    X = np.column_stack([np.ones_like(r_data), r_data, T_data])
    Beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    
    A_init = np.exp(Beta[0])
    b_init = max(0.001, -Beta[1])
    c_init = max(0.01, -Beta[2])
    k_init = 2.0
    
    initial_guess = [A_init, k_init, b_init, c_init]
    
    def log_model(vars, A, k, b, c):
        r, T = vars
        return np.log(A) - k*np.log(r) - b*r - c*T
    
    try:
        popt, pcov = curve_fit(log_model, (r_data, T_data), ln_D,
                              p0=initial_guess, maxfev=50000,
                              bounds=([0, 0.5, 0, 0], 
                                     [np.inf, 3, 1, 1]))
        
        A, k, b, c = popt
        perr = np.sqrt(np.diag(pcov))
        
        ln_D_fit = log_model((r_data, T_data), *popt)
        ss_res = np.sum((ln_D - ln_D_fit)**2)
        ss_tot = np.sum((ln_D - np.mean(ln_D))**2)
        R2 = 1 - ss_res/ss_tot
        
        return A, k, b, c, R2, list(perr)
    except Exception as e:
        st.error(f"Model fitting error: {e}")
        return None, None, None, None, 0, [0]*4

def predict_simple_exponential(params, r, T):
    """
    Calculate the prediction of the Simple Exponential model

    Args:
        params: Tuple of 4 parameters (A, k, b, c)
        r: Distance(s) in meters
        T: Shield thickness(es) in cm

    Returns:
        Predicted dose in Gy
    """
    A, k, b, c = params
    return A * r**(-k) * np.exp(-b*r) * np.exp(-c*T)

# ______________________________________________________________________________________________________________________
# Coupled Kernel model function (C1)
#
# Physical rationale: separable models D = f(r) x g(T) miss the fact that a
# thicker shield hardens the spectrum, which shifts BOTH the geometric
# exponent and the air attenuation. This model couples r and T explicitly
# via T*ln(r) and T*r terms, and is fitted in one closed-form least-squares
# pass (no initial guess, no non-convergence risk).

def fit_coupled_kernel(r_data, T_data, D_data):
    """
    Coupled Point Kernel model (least-squares closed form)

    Parameters (7):
    - a0: Log-amplitude
    - a1: Base geometric exponent (coefficient of ln r)
    - a2: Spectral hardening of the exponent (coefficient of T.ln r)
    - a3: Air attenuation (coefficient of r, typically negative = -mu_air)
    - a4: Spectral hardening of air attenuation (coefficient of T.r)
    - a5: Shield attenuation (coefficient of T, typically negative = -mu_screen)
    - a6: Shield build-up (coefficient of T^2)

    Model: ln D(r,T) = a0 + (a1 + a2.T).ln r + (a3 + a4.T).r + a5.T + a6.T^2
    """
    r_data = np.array(r_data, dtype=float)
    T_data = np.array(T_data, dtype=float)
    D_data = np.array(D_data, dtype=float)

    valid = (~np.isnan(D_data)) & (D_data > 0) & (r_data > 0)
    r_data, T_data, D_data = r_data[valid], T_data[valid], D_data[valid]

    if len(D_data) == 0:
        return None, None, None, None, None, None, None, 0, [0]*7

    try:
        ln_D = np.log(D_data)
        u = np.log(r_data)
        X = np.column_stack([np.ones_like(r_data), u, T_data*u, r_data, T_data*r_data, T_data, T_data**2])
        beta, _, _, _ = np.linalg.lstsq(X, ln_D, rcond=None)

        ln_D_fit = X @ beta
        ss_res = np.sum((ln_D - ln_D_fit)**2)
        ss_tot = np.sum((ln_D - np.mean(ln_D))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

        return (*beta, R2, [0.0]*7)
    except Exception as e:
        st.error(f"Model fitting error: {e}")
        return None, None, None, None, None, None, None, 0, [0]*7

def predict_coupled_kernel(params, r, T):
    """
    Calculate the prediction of the Coupled Kernel model

    Args:
        params: Tuple of 7 parameters (a0, a1, a2, a3, a4, a5, a6)
        r: Distance(s) in meters
        T: Shield thickness(es) in cm

    Returns:
        Predicted dose in Gy
    """
    a0, a1, a2, a3, a4, a5, a6 = params
    u = np.log(r)
    return np.exp(a0 + a1*u + a2*T*u + a3*r + a4*T*r + a5*T + a6*T**2)

# ______________________________________________________________________________________________________________________
# Coupled Kernel + sqrt(r) Air Build-up model function (C4)

def fit_coupled_kernel_sqrt(r_data, T_data, D_data):
    """
    Coupled Point Kernel model with sqrt(r) air build-up term (least-squares closed form)

    Parameters (8):
    - a0: Log-amplitude
    - a1: Base geometric exponent (coefficient of ln r)
    - a2: Spectral hardening of the exponent (coefficient of T.ln r)
    - a3: Air attenuation (coefficient of r, typically negative = -mu_air)
    - a4: Spectral hardening of air attenuation (coefficient of T.r)
    - a5: Air build-up (coefficient of sqrt(r), typically positive)
    - a6: Shield attenuation (coefficient of T, typically negative = -mu_screen)
    - a7: Shield build-up (coefficient of T^2)

    Model: ln D(r,T) = a0 + (a1 + a2.T).ln r + (a3 + a4.T).r + a5.sqrt(r) + a6.T + a7.T^2
    """
    r_data = np.array(r_data, dtype=float)
    T_data = np.array(T_data, dtype=float)
    D_data = np.array(D_data, dtype=float)

    valid = (~np.isnan(D_data)) & (D_data > 0) & (r_data > 0)
    r_data, T_data, D_data = r_data[valid], T_data[valid], D_data[valid]

    if len(D_data) == 0:
        return None, None, None, None, None, None, None, None, 0, [0]*8

    try:
        ln_D = np.log(D_data)
        u = np.log(r_data)
        X = np.column_stack([np.ones_like(r_data), u, T_data*u, r_data, T_data*r_data,
                             np.sqrt(r_data), T_data, T_data**2])
        beta, _, _, _ = np.linalg.lstsq(X, ln_D, rcond=None)

        ln_D_fit = X @ beta
        ss_res = np.sum((ln_D - ln_D_fit)**2)
        ss_tot = np.sum((ln_D - np.mean(ln_D))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

        return (*beta, R2, [0.0]*8)
    except Exception as e:
        st.error(f"Model fitting error: {e}")
        return None, None, None, None, None, None, None, None, 0, [0]*8

def predict_coupled_kernel_sqrt(params, r, T):
    """
    Calculate the prediction of the Coupled Kernel + sqrt(r) model

    Args:
        params: Tuple of 8 parameters (a0, a1, a2, a3, a4, a5, a6, a7)
        r: Distance(s) in meters
        T: Shield thickness(es) in cm

    Returns:
        Predicted dose in Gy
    """
    a0, a1, a2, a3, a4, a5, a6, a7 = params
    u = np.log(r)
    return np.exp(a0 + a1*u + a2*T*u + a3*r + a4*T*r + a5*np.sqrt(r) + a6*T + a7*T**2)

# ______________________________________________________________________________________________________________________
# Kernel + Skyshine Floor model function (C3)

def fit_kernel_skyshine(r_data, T_data, D_data):
    """
    Hardened direct kernel plus an additive skyshine floor (robust far-field behavior)

    Parameters (10):
    - A: Direct component amplitude
    - p: Geometric exponent
    - mu_air: Air attenuation (m⁻¹)
    - mu_T: Spectral hardening of air attenuation (m⁻¹.cm⁻¹)
    - s1: Linear shield attenuation (cm⁻¹)
    - s2: Quadratic shield term (cm⁻²)
    - B: Skyshine component amplitude
    - n_sky: Skyshine geometric exponent
    - mu_sky: Skyshine attenuation in air (m⁻¹)
    - mu_s2: Skyshine attenuation in shield (cm⁻¹)

    Model: D(r,T) = A.r^(-p).exp(-(mu_air+mu_T.T).r).exp(-s1.T-s2.T^2)
                    + B.r^(-n_sky).exp(-mu_sky.r).exp(-mu_s2.T)
    """
    r_data = np.array(r_data, dtype=float)
    T_data = np.array(T_data, dtype=float)
    D_data = np.array(D_data, dtype=float)

    valid = (~np.isnan(D_data)) & (D_data > 0) & (r_data > 0)
    r_data, T_data, D_data = r_data[valid], T_data[valid], D_data[valid]

    if len(D_data) == 0:
        return None, None, None, None, None, None, None, None, None, None, 0, [0]*10

    ln_D = np.log(D_data)
    u = np.log(r_data)

    # Initial estimate for the direct component using linear regression (no floor)
    b, _, _, _ = np.linalg.lstsq(
        np.column_stack([np.ones_like(r_data), u, r_data, T_data*r_data, T_data, T_data**2]),
        ln_D, rcond=None
    )
    A0 = np.exp(np.clip(b[0], -50, 50))
    p0_init = max(-b[1], 0.5)
    mu_air0 = max(-b[2], 1e-3)
    mu_T0 = np.clip(b[3], -0.005, 0.005)
    s10 = max(-b[4], 0.0)
    s20 = b[5]

    lo = [0, 0.5, 0, -0.01, -0.5, -0.05, 0, 0.5, 0, 0]
    hi = [np.inf, 3, 0.2, 0.01, 1, 0.05, np.inf, 2.5, 0.05, 1]
    initial_guess = [A0, p0_init, mu_air0, mu_T0, s10, s20, A0*1e-3, 1.2, 0.005, 0.02]
    initial_guess = [min(max(initial_guess[k], lo[k] + 1e-9),
                         (hi[k] - 1e-9) if np.isfinite(hi[k]) else initial_guess[k])
                     for k in range(10)]

    def log_model(vars, A, p, mu_air, mu_T, s1, s2, B, n_sky, mu_sky, mu_s2):
        r, T = vars
        direct = A * r**(-p) * np.exp(-(mu_air + mu_T*T)*r) * np.exp(-s1*T - s2*T**2)
        sky = B * r**(-n_sky) * np.exp(-mu_sky*r) * np.exp(-mu_s2*T)
        return np.log(np.maximum(direct + sky, 1e-300))

    try:
        popt, pcov = curve_fit(log_model, (r_data, T_data), ln_D,
                              p0=initial_guess, maxfev=100000, bounds=(lo, hi))

        perr = np.sqrt(np.diag(pcov))
        ln_D_fit = log_model((r_data, T_data), *popt)
        ss_res = np.sum((ln_D - ln_D_fit)**2)
        ss_tot = np.sum((ln_D - np.mean(ln_D))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

        return (*popt, R2, list(perr))
    except Exception as e:
        st.error(f"Model fitting error: {e}")
        return None, None, None, None, None, None, None, None, None, None, 0, [0]*10

def predict_kernel_skyshine(params, r, T):
    """
    Calculate the prediction of the Kernel + Skyshine Floor model

    Args:
        params: Tuple of 10 parameters (A, p, mu_air, mu_T, s1, s2, B, n_sky, mu_sky, mu_s2)
        r: Distance(s) in meters
        T: Shield thickness(es) in cm

    Returns:
        Predicted dose in Gy
    """
    A, p, mu_air, mu_T, s1, s2, B, n_sky, mu_sky, mu_s2 = params
    direct = A * r**(-p) * np.exp(-(mu_air + mu_T*T)*r) * np.exp(-s1*T - s2*T**2)
    sky = B * r**(-n_sky) * np.exp(-mu_sky*r) * np.exp(-mu_s2*T)
    return direct + sky

# ______________________________________________________________________________________________________________________
# Coupled Kernel sqrt(r) + Skyshine Floor model function (C5) — best raw fit

def fit_coupled_sqrt_skyshine(r_data, T_data, D_data):
    """
    Coupled kernel with sqrt(r) air build-up plus an additive skyshine floor

    Parameters (12):
    - a0: Log-amplitude of the direct component
    - c_lnr: Base geometric exponent (coefficient of ln r)
    - c_Tlnr: Spectral hardening of the exponent (coefficient of T.ln r)
    - c_r: Air attenuation (coefficient of r)
    - c_Tr: Spectral hardening of air attenuation (coefficient of T.r)
    - c_sqrt: Air build-up (coefficient of sqrt(r))
    - c_T: Shield attenuation (coefficient of T)
    - c_T2: Shield build-up (coefficient of T^2)
    - lnB: Log-amplitude of the skyshine component
    - n_sky: Skyshine geometric exponent
    - mu_sky: Skyshine attenuation in air (m⁻¹)
    - mu_s2: Skyshine attenuation in shield (cm⁻¹)
    """
    r_data = np.array(r_data, dtype=float)
    T_data = np.array(T_data, dtype=float)
    D_data = np.array(D_data, dtype=float)

    valid = (~np.isnan(D_data)) & (D_data > 0) & (r_data > 0)
    r_data, T_data, D_data = r_data[valid], T_data[valid], D_data[valid]

    if len(D_data) == 0:
        return (None,)*12 + (0, [0]*12)

    ln_D = np.log(D_data)

    # Initialize the direct-component coefficients from the closed-form C4 fit
    c4 = fit_coupled_kernel_sqrt(r_data, T_data, D_data)
    a0, a1, a2, a3, a4, a5, a6, a7 = c4[:8]
    initial_guess = [a0, -a1, -a2, -a3, -a4, a5, -a6, -a7, a0 - 8, 1.2, 0.005, 0.02]

    lo = [-50, -3, -0.5, -0.5, -0.05, -5, -1, -0.5, -80, 0.5, 0, 0]
    hi = [ 50,  3,  0.5,  0.5,  0.05,  5,  1,  0.5,  50, 2.5, 0.05, 1]
    initial_guess = [min(max(initial_guess[k], lo[k] + 1e-6), hi[k] - 1e-6) for k in range(12)]

    def log_model(vars, a0, c_lnr, c_Tlnr, c_r, c_Tr, c_sqrt, c_T, c_T2, lnB, n_sky, mu_sky, mu_s2):
        r, T = vars
        u = np.log(r)
        direct = np.exp(np.clip(a0 - c_lnr*u - c_Tlnr*T*u - c_r*r - c_Tr*T*r
                                + c_sqrt*np.sqrt(r) - c_T*T - c_T2*T**2, -700, 700))
        sky = np.exp(np.clip(lnB, -700, 700)) * r**(-n_sky) * np.exp(-mu_sky*r) * np.exp(-mu_s2*T)
        return np.log(np.maximum(direct + sky, 1e-300))

    try:
        popt, pcov = curve_fit(log_model, (r_data, T_data), ln_D,
                              p0=initial_guess, maxfev=150000, bounds=(lo, hi))

        perr = np.sqrt(np.diag(pcov))
        ln_D_fit = log_model((r_data, T_data), *popt)
        ss_res = np.sum((ln_D - ln_D_fit)**2)
        ss_tot = np.sum((ln_D - np.mean(ln_D))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

        return (*popt, R2, list(perr))
    except Exception as e:
        st.error(f"Model fitting error: {e}")
        return (None,)*12 + (0, [0]*12)

def predict_coupled_sqrt_skyshine(params, r, T):
    """
    Calculate the prediction of the Coupled Kernel sqrt(r) + Skyshine Floor model

    Args:
        params: Tuple of 12 parameters
        r: Distance(s) in meters
        T: Shield thickness(es) in cm

    Returns:
        Predicted dose in Gy
    """
    a0, c_lnr, c_Tlnr, c_r, c_Tr, c_sqrt, c_T, c_T2, lnB, n_sky, mu_sky, mu_s2 = params
    u = np.log(r)
    direct = np.exp(np.clip(a0 - c_lnr*u - c_Tlnr*T*u - c_r*r - c_Tr*T*r
                            + c_sqrt*np.sqrt(r) - c_T*T - c_T2*T**2, -700, 700))
    sky = np.exp(np.clip(lnB, -700, 700)) * r**(-n_sky) * np.exp(-mu_sky*r) * np.exp(-mu_s2*T)
    return direct + sky

# ______________________________________________________________________________________________________________________
# User interface

# Sidebar: Model selection
st.sidebar.header("⚙️ Model Selection")

# Dictionary of available models
AVAILABLE_MODELS = {
    "Simple Exponential": {
        "name": "Simple Exponential (4 parameters)",
        "description": "Basic model with geometric and exponential attenuation (good for neutrons)",
        "params_count": 4,
        "icon": "",
        "performance": "R² ≈ variable | Simple & fast"
    },
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
    "Coupled Kernel": {
        "name": "Coupled Kernel (7 parameters)",
        "description": "Point kernel with r-T coupling for spectral hardening (closed-form fit)",
        "params_count": 7,
        "icon": "🔗",
        "performance": "R² ≈ 0.9994 | Error ≈ 10.2% (median, 68 configs)"
    },
    "Coupled Kernel + sqrt(r)": {
        "name": "Coupled Kernel + sqrt(r) (8 parameters)",
        "description": "Coupled kernel with an air build-up term in sqrt(r) (closed-form fit)",
        "params_count": 8,
        "icon": "🔗",
        "performance": "R² ≈ 0.9997 | Error ≈ 6.2% (median, 68 configs)"
    },
    "Kernel + Skyshine Floor": {
        "name": "Kernel + Skyshine Floor (10 parameters)",
        "description": "Hardened direct kernel with an additive skyshine floor (best generalization)",
        "params_count": 10,
        "icon": "🔗",
        "performance": "R² ≈ 0.9998 | Error ≈ 5.4% (median, 68 configs)"
    },
    "Coupled Kernel + Skyshine": {
        "name": "Coupled Kernel + Skyshine (12 parameters)",
        "description": "Coupled kernel with sqrt(r) build-up plus an additive skyshine floor (best raw fit)",
        "params_count": 12,
        "icon": "🔗",
        "performance": "R² ≈ 0.9998 | Error ≈ 4.8% (median, 68 configs)"
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
        if selected_model == "Simple Exponential":
            result = fit_simple_exponential(r_fit, T_fit, D_fit)
        elif selected_model == "Skyshine Enhanced":
            result = fit_skyshine_enhanced(r_fit, T_fit, D_fit)
        elif selected_model == "Super Hybrid":
            result = fit_super_hybrid(r_fit, T_fit, D_fit)
        elif selected_model == "Hybrid Corrected":
            result = fit_hybrid_corrected(r_fit, T_fit, D_fit)
        elif selected_model == "Coupled Kernel":
            result = fit_coupled_kernel(r_fit, T_fit, D_fit)
        elif selected_model == "Coupled Kernel + sqrt(r)":
            result = fit_coupled_kernel_sqrt(r_fit, T_fit, D_fit)
        elif selected_model == "Kernel + Skyshine Floor":
            result = fit_kernel_skyshine(r_fit, T_fit, D_fit)
        elif selected_model == "Coupled Kernel + Skyshine":
            result = fit_coupled_sqrt_skyshine(r_fit, T_fit, D_fit)
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
    if selected_model == "Simple Exponential":
        st.markdown(r"""
        ## 📐 Model Equation
        
        The **Simple Exponential** model is a compact 4-parameter model particularly suited for **neutron dose** calculations:
        
        $$D(r, T) = A \cdot r^{-k} \cdot e^{-b \cdot r} \cdot e^{-c \cdot T}$$
        
        This model captures:
        - **Geometric attenuation**: $r^{-k}$ with adjustable exponent $k$
        - **Exponential attenuation in air**: $e^{-b \cdot r}$
        - **Exponential attenuation in shield**: $e^{-c \cdot T}$
        
        ## 🔧 Model Parameters (4)
        
        | Parameter | Description | Unit | Typical Range |
        |-----------|-------------|------|---------------|
        | `A` | Amplitude (dose normalization) | - | > 0 |
        | `k` | Geometric exponent | - | 0.5 - 3.0 |
        | `b` | Attenuation coefficient in air | m⁻¹ | 0 - 1 |
        | `c` | Attenuation coefficient in shield | cm⁻¹ | 0 - 1 |
        
        ## 💡 Use Cases
        
        - **Primary**: Neutron dose modeling (simplified physics)
        - **Advantages**: Few parameters, fast fitting, good generalization
        - **Limitations**: No build-up or skyshine effects
        """)
    elif selected_model == "Skyshine Enhanced":
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
    elif selected_model == "Coupled Kernel":
        st.markdown(r"""
        ## 📐 Model Equation

        Separable models $D = f(r) \cdot g(T)$ miss an important effect: a thicker shield **hardens the spectrum**,
        which shifts both the geometric exponent and the air attenuation. The **Coupled Kernel** model couples
        $r$ and $T$ explicitly:

        $$\ln D(r, T) = a_0 + (a_1 + a_2 T)\ln r + (a_3 + a_4 T)\, r + a_5 T + a_6 T^2$$

        This model is fitted in a **single closed-form least-squares pass** (no initial guess, no risk of
        non-convergence), which makes it very robust across configurations.

        ## 🔧 Model Parameters (7)

        | Parameter | Description |
        |-----------|-------------|
        | `a₀` | Log-amplitude |
        | `a₁` | Base geometric exponent (≈ -2 for a point source) |
        | `a₂` | Spectral hardening of the exponent (coefficient of T·ln r) |
        | `a₃` | Air attenuation (coefficient of r, ≈ -μ_air) |
        | `a₄` | Spectral hardening of air attenuation (coefficient of T·r) |
        | `a₅` | Shield attenuation (coefficient of T, ≈ -μ_screen) |
        | `a₆` | Shield build-up (coefficient of T²) |

        ## 💡 Use Cases

        - **Advantages**: Closed-form fit (linear least squares), always converges, physically interpretable
          coefficients
        - **Benchmark**: Beats Skyshine Enhanced on 41/68 tested configurations (in-sample fit)
        """)
    elif selected_model == "Coupled Kernel + sqrt(r)":
        st.markdown(r"""
        ## 📐 Model Equation

        Extends the **Coupled Kernel** model with an air build-up term in $\sqrt{r}$:

        $$\ln D(r, T) = a_0 + (a_1 + a_2 T)\ln r + a_5\sqrt{r} + (a_3 + a_4 T)\, r + a_6 T + a_7 T^2$$

        Like the Coupled Kernel model, this is fitted in a **single closed-form least-squares pass**.
        It is the recommended default replacement for Skyshine Enhanced: best raw fit among the
        closed-form models, and no fitting risk.

        ## 🔧 Model Parameters (8)

        | Parameter | Description |
        |-----------|-------------|
        | `a₀` | Log-amplitude |
        | `a₁` | Base geometric exponent (≈ -2 for a point source) |
        | `a₂` | Spectral hardening of the exponent (coefficient of T·ln r) |
        | `a₃` | Air attenuation (coefficient of r, ≈ -μ_air) |
        | `a₄` | Spectral hardening of air attenuation (coefficient of T·r) |
        | `a₅` | Air build-up (coefficient of √r, typically > 0) |
        | `a₆` | Shield attenuation (coefficient of T, ≈ -μ_screen) |
        | `a₇` | Shield build-up (coefficient of T²) |

        ## 💡 Use Cases

        - **Advantages**: Closed-form fit, always converges, best in-sample fit among closed-form models
        - **Benchmark**: Beats Skyshine Enhanced on 59/68 tested configurations, median error 6.2% vs 9.8%
        """)
    elif selected_model == "Kernel + Skyshine Floor":
        st.markdown(r"""
        ## 📐 Model Equation

        Combines a **hardened direct kernel** with an **additive skyshine floor**, which keeps the far-field
        behavior physically sound and gives the best generalization when extrapolating to a shield thickness
        that was not used for the fit:

        $$D_{\text{total}}(r, T) = \underbrace{A\, r^{-p}\, e^{-(\mu_{\text{air}} + \mu_T T)\, r}\, e^{-s_1 T - s_2 T^2}}_{D_{\text{direct}} \text{ (hardened)}} \; + \; \underbrace{B\, r^{-n_{\text{sky}}}\, e^{-\mu_{\text{sky}}\, r}\, e^{-\mu_{s2}\, T}}_{D_{\text{skyshine}}}$$

        ## 🔧 Model Parameters (10)

        | Parameter | Description | Unit |
        |-----------|-------------|------|
        | `A` | Direct component amplitude | - |
        | `p` | Geometric exponent | - |
        | `μ_air` | Air attenuation | m⁻¹ |
        | `μ_T` | Spectral hardening of air attenuation | m⁻¹·cm⁻¹ |
        | `s₁` | Linear shield attenuation | cm⁻¹ |
        | `s₂` | Quadratic shield term | cm⁻² |
        | `B` | Skyshine component amplitude | - |
        | `n_sky` | Skyshine geometric exponent | - |
        | `μ_sky` | Skyshine attenuation in air | m⁻¹ |
        | `μ_s2` | Skyshine attenuation in shield | cm⁻¹ |

        ## 💡 Use Cases

        - **Advantages**: Best generalization across shield thicknesses (median error 9.6% under
          leave-one-thickness-out cross-validation, vs 17.5% for Skyshine Enhanced)
        - **Benchmark**: Beats Skyshine Enhanced on 66/68 tested configurations (in-sample fit)
        - **Recommended** when robustness to extrapolation matters more than the very best raw fit
        """)
    elif selected_model == "Coupled Kernel + Skyshine":
        st.markdown(r"""
        ## 📐 Model Equation

        Combines the coupled $\sqrt{r}$ direct kernel with an additive skyshine floor — the most flexible
        of the new models, and the one with the best raw fit:

        $$D_{\text{total}}(r, T) = \underbrace{\exp\!\big[a_0 - c_{\ln r}\ln r - c_{T\ln r}\, T\ln r - c_r\, r - c_{Tr}\, T r + c_{\sqrt{r}}\sqrt{r} - c_T T - c_{T^2} T^2\big]}_{D_{\text{direct}}} \; + \; \underbrace{e^{\ln B}\, r^{-n_{\text{sky}}}\, e^{-\mu_{\text{sky}}\, r}\, e^{-\mu_{s2}\, T}}_{D_{\text{skyshine}}}$$

        ## 🔧 Model Parameters (12)

        | Parameter | Description |
        |-----------|-------------|
        | `a₀` | Log-amplitude of the direct component |
        | `c_lnr` | Base geometric exponent |
        | `c_Tlnr` | Spectral hardening of the exponent (T·ln r) |
        | `c_r` | Air attenuation |
        | `c_Tr` | Spectral hardening of air attenuation (T·r) |
        | `c_sqrt` | Air build-up (√r) |
        | `c_T` | Shield attenuation |
        | `c_T2` | Shield build-up (T²) |
        | `lnB` | Log-amplitude of the skyshine component |
        | `n_sky` | Skyshine geometric exponent |
        | `μ_sky` | Skyshine attenuation in air |
        | `μ_s2` | Skyshine attenuation in shield |

        ## 💡 Use Cases

        - **Advantages**: Best raw fit of all tested models (median error 4.8%, R² ≈ 0.9998)
        - **Benchmark**: Beats Skyshine Enhanced on **68/68** tested configurations (in-sample fit)
        - **Trade-off**: 12 parameters, non-linear fit — use when fit quality matters more than
          fitting speed/robustness
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
        if selected_model == "Simple Exponential":
            param_names = ['A', 'k', 'b', 'c']
            param_units = ['-', '-', 'm⁻¹', 'cm⁻¹']
            param_descriptions = [
                'Amplitude (dose normalization)',
                'Geometric exponent',
                'Attenuation coefficient in air',
                'Attenuation coefficient in shield'
            ]
        elif selected_model == "Skyshine Enhanced":
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
        elif selected_model == "Coupled Kernel":
            param_names = ['a₀', 'a₁', 'a₂', 'a₃', 'a₄', 'a₅', 'a₆']
            param_units = ['-', '-', '-', 'm⁻¹', 'm⁻¹·cm⁻¹', 'cm⁻¹', 'cm⁻²']
            param_descriptions = [
                'Log-amplitude',
                'Base geometric exponent (coeff. of ln r)',
                'Spectral hardening of the exponent (coeff. of T.ln r)',
                'Air attenuation (coeff. of r)',
                'Spectral hardening of air attenuation (coeff. of T.r)',
                'Shield attenuation (coeff. of T)',
                'Shield build-up (coeff. of T²)'
            ]
        elif selected_model == "Coupled Kernel + sqrt(r)":
            param_names = ['a₀', 'a₁', 'a₂', 'a₃', 'a₄', 'a₅', 'a₆', 'a₇']
            param_units = ['-', '-', '-', 'm⁻¹', 'm⁻¹·cm⁻¹', '-', 'cm⁻¹', 'cm⁻²']
            param_descriptions = [
                'Log-amplitude',
                'Base geometric exponent (coeff. of ln r)',
                'Spectral hardening of the exponent (coeff. of T.ln r)',
                'Air attenuation (coeff. of r)',
                'Spectral hardening of air attenuation (coeff. of T.r)',
                'Air build-up (coeff. of sqrt(r))',
                'Shield attenuation (coeff. of T)',
                'Shield build-up (coeff. of T²)'
            ]
        elif selected_model == "Kernel + Skyshine Floor":
            param_names = ['A', 'p', 'μ_air', 'μ_T', 's₁', 's₂', 'B', 'n_sky', 'μ_sky', 'μ_s2']
            param_units = ['-', '-', 'm⁻¹', 'm⁻¹·cm⁻¹', 'cm⁻¹', 'cm⁻²', '-', '-', 'm⁻¹', 'cm⁻¹']
            param_descriptions = [
                'Direct component amplitude',
                'Geometric exponent',
                'Air attenuation',
                'Spectral hardening of air attenuation',
                'Linear shield attenuation',
                'Quadratic shield term',
                'Skyshine component amplitude',
                'Skyshine geometric exponent',
                'Skyshine attenuation in air',
                'Skyshine attenuation in shield'
            ]
        elif selected_model == "Coupled Kernel + Skyshine":
            param_names = ['a₀', 'c_lnr', 'c_Tlnr', 'c_r', 'c_Tr', 'c_sqrt', 'c_T', 'c_T2',
                           'lnB', 'n_sky', 'μ_sky', 'μ_s2']
            param_units = ['-', '-', '-', 'm⁻¹', 'm⁻¹·cm⁻¹', '-', 'cm⁻¹', 'cm⁻²',
                          '-', '-', 'm⁻¹', 'cm⁻¹']
            param_descriptions = [
                'Log-amplitude (direct component)',
                'Base geometric exponent',
                'Spectral hardening of the exponent (T.ln r)',
                'Air attenuation',
                'Spectral hardening of air attenuation (T.r)',
                'Air build-up (sqrt(r))',
                'Shield attenuation',
                'Shield build-up (T²)',
                'Log-amplitude (skyshine component)',
                'Skyshine geometric exponent',
                'Skyshine attenuation in air',
                'Skyshine attenuation in shield'
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
            if selected_model == "Simple Exponential":
                D_pred = predict_simple_exponential(params, r_fit, T_fit)
            elif selected_model == "Skyshine Enhanced":
                D_pred = predict_skyshine_enhanced(params, r_fit, T_fit)
            elif selected_model == "Super Hybrid":
                D_pred = predict_super_hybrid(params, r_fit, T_fit)
            elif selected_model == "Hybrid Corrected":
                D_pred = predict_hybrid_corrected(params, r_fit, T_fit)
            elif selected_model == "Coupled Kernel":
                D_pred = predict_coupled_kernel(params, r_fit, T_fit)
            elif selected_model == "Coupled Kernel + sqrt(r)":
                D_pred = predict_coupled_kernel_sqrt(params, r_fit, T_fit)
            elif selected_model == "Kernel + Skyshine Floor":
                D_pred = predict_kernel_skyshine(params, r_fit, T_fit)
            elif selected_model == "Coupled Kernel + Skyshine":
                D_pred = predict_coupled_sqrt_skyshine(params, r_fit, T_fit)
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
        if selected_model == "Simple Exponential":
            D_model = predict_simple_exponential(params, r_model, np.full_like(r_model, T_val))
        elif selected_model == "Skyshine Enhanced":
            D_model = predict_skyshine_enhanced(params, r_model, np.full_like(r_model, T_val))
        elif selected_model == "Super Hybrid":
            D_model = predict_super_hybrid(params, r_model, np.full_like(r_model, T_val))
        elif selected_model == "Hybrid Corrected":
            D_model = predict_hybrid_corrected(params, r_model, np.full_like(r_model, T_val))
        elif selected_model == "Coupled Kernel":
            D_model = predict_coupled_kernel(params, r_model, np.full_like(r_model, T_val))
        elif selected_model == "Coupled Kernel + sqrt(r)":
            D_model = predict_coupled_kernel_sqrt(params, r_model, np.full_like(r_model, T_val))
        elif selected_model == "Kernel + Skyshine Floor":
            D_model = predict_kernel_skyshine(params, r_model, np.full_like(r_model, T_val))
        elif selected_model == "Coupled Kernel + Skyshine":
            D_model = predict_coupled_sqrt_skyshine(params, r_model, np.full_like(r_model, T_val))
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
        if selected_model == "Simple Exponential":
            D_custom = predict_simple_exponential(params, r_custom, np.full_like(r_custom, custom_thickness))
        elif selected_model == "Skyshine Enhanced":
            D_custom = predict_skyshine_enhanced(params, r_custom, np.full_like(r_custom, custom_thickness))
        elif selected_model == "Super Hybrid":
            D_custom = predict_super_hybrid(params, r_custom, np.full_like(r_custom, custom_thickness))
        elif selected_model == "Hybrid Corrected":
            D_custom = predict_hybrid_corrected(params, r_custom, np.full_like(r_custom, custom_thickness))
        elif selected_model == "Coupled Kernel":
            D_custom = predict_coupled_kernel(params, r_custom, np.full_like(r_custom, custom_thickness))
        elif selected_model == "Coupled Kernel + sqrt(r)":
            D_custom = predict_coupled_kernel_sqrt(params, r_custom, np.full_like(r_custom, custom_thickness))
        elif selected_model == "Kernel + Skyshine Floor":
            D_custom = predict_kernel_skyshine(params, r_custom, np.full_like(r_custom, custom_thickness))
        elif selected_model == "Coupled Kernel + Skyshine":
            D_custom = predict_coupled_sqrt_skyshine(params, r_custom, np.full_like(r_custom, custom_thickness))
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
            elif selected_model == "Coupled Kernel":
                D_custom = predict_coupled_kernel(params, r_custom, np.full_like(r_custom, custom_thickness))
            elif selected_model == "Coupled Kernel + sqrt(r)":
                D_custom = predict_coupled_kernel_sqrt(params, r_custom, np.full_like(r_custom, custom_thickness))
            elif selected_model == "Kernel + Skyshine Floor":
                D_custom = predict_kernel_skyshine(params, r_custom, np.full_like(r_custom, custom_thickness))
            elif selected_model == "Coupled Kernel + Skyshine":
                D_custom = predict_coupled_sqrt_skyshine(params, r_custom, np.full_like(r_custom, custom_thickness))
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
