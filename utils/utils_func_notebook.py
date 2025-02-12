import numpy as np 
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt

def fit_data_to_distance(xdata, ydata, plot=False):
    """
    Fits the model y = A * x^(-k) * exp(-b * x) to the provided data.

    Parameters:
    - xdata: numpy array of x values in meters
    - ydata: numpy array of corresponding y values
    - plot: boolean indicating whether to display the fit plot

    Returns:
    - A_opt: optimal value of parameter A
    - k_opt: optimal value of parameter k
    - b_opt: optimal value of parameter b
    - R_squared: coefficient of determination R²
    - param_uncertainties: uncertainties on the parameters (A, k, b)
    """
    # Ensure xdata and ydata are numpy arrays
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # Handle missing or invalid data
    initial_count = len(ydata)
    valid_indices = ~np.isnan(ydata) & (ydata > 0)  # Exclude NaN and non-positive values
    xdata = xdata[valid_indices]
    ydata = ydata[valid_indices]
    excluded_count = initial_count - len(ydata)

    if excluded_count > 0:
        print(f"{excluded_count} data points were excluded (missing or invalid values).")

    # Check if there is enough data to fit
    if len(xdata) == 0 or len(ydata) == 0:
        raise ValueError("No valid data is available for fitting.")

    # Take the natural logarithm of y data
    ln_ydata = np.log(ydata)

    # Model function for the logarithms
    def log_model_func(x, lnA, k, b):
        return lnA - k * np.log(x) - b * x

    # Initial estimates from linear regression
    ln_x = np.log(xdata)
    X = np.column_stack((np.ones(len(ln_x)), ln_x, xdata))
    beta, residuals, rank, s = np.linalg.lstsq(X, ln_ydata, rcond=None)
    lnA_init = beta[0]
    k_init = -beta[1]
    b_init = -beta[2]

    initial_guess = [lnA_init, k_init, b_init]

    # Fit the model to the logarithmic data
    popt, pcov = curve_fit(log_model_func, xdata, ln_ydata, p0=initial_guess, maxfev=10000)

    # Extract optimal parameters
    lnA_opt, k_opt, b_opt = popt
    A_opt = np.exp(lnA_opt)

    # Calculate uncertainties on the parameters
    param_uncertainties = np.sqrt(np.diag(pcov))  # Standard deviations of lnA, k, and b
    A_uncertainty = A_opt * param_uncertainties[0]  # Convert lnA uncertainty to A uncertainty
    uncertainties = [A_uncertainty, param_uncertainties[1], param_uncertainties[2]]

    # Compute fitted values with optimal parameters
    ln_y_fit = log_model_func(xdata, lnA_opt, k_opt, b_opt)
    y_fit = np.exp(ln_y_fit)

    # Calculate the coefficient of determination R²
    ss_res = np.sum((ln_ydata - ln_y_fit) ** 2)
    ss_tot = np.sum((ln_ydata - np.mean(ln_ydata)) ** 2)
    R_squared = 1 - (ss_res / ss_tot)

    # Optionally, display the plot
    if plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(xdata, ydata, 'bo', label='Experimental data')
        plt.loglog(xdata, y_fit, 'r-', label='Fitted model')
        plt.xlabel('x (meters)')
        plt.ylabel('y')
        plt.title('Model fit to data')
        plt.legend()
        plt.grid(True, which="both", ls="--")

        # Add the formula with parameter values and R² to the plot
        formula_text = (f"y = {A_opt:.3e} * x^(-{k_opt:.3f}) * exp(-{b_opt:.3e} * x)\n"
                        f"$R^2$ = {R_squared:.6f}")
        plt.text(0.05, 0.1, formula_text, transform=plt.gca().transAxes, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5))

        plt.show()

    return A_opt, k_opt, b_opt, R_squared, uncertainties

def fit_data_to_distance_thickness(xdata, thickness_data, ydata, plot=True):
    """
    Ajuste le modèle :
        y = A * x^(-k) * exp(-b*x) * exp(-c*thickness)
    sur l'ensemble (xdata, thickness_data, ydata).

    Renvoie :
    - A_opt, k_opt, b_opt, c_opt : les paramètres optimaux
    - R_squared : le coefficient de détermination (calculé sur ln(y))
    - uncertainties : incertitudes (ecarts-types) sur A, k, b, c.
    """

    # Conversion en np.array, filtrage des valeurs invalides
    xdata = np.array(xdata)
    thickness_data = np.array(thickness_data)
    ydata = np.array(ydata, dtype=float)

    # On élimine les points y <= 0 ou NaN, car log(y) est impossible sinon.
    valid_mask = ~np.isnan(ydata) & (ydata > 0)
    xdata = xdata[valid_mask]
    thickness_data = thickness_data[valid_mask]
    ydata = ydata[valid_mask]

    if len(ydata) == 0:
        raise ValueError("Pas de données valides pour l'ajustement.")

    # On passe en log(y)
    ln_ydata = np.log(ydata)
    ln_x = np.log(xdata)  # pour gérer le terme x^(-k)

    # ----------------------------------------------------------------
    # 1) Estimation initiale par régression linéaire (np.linalg.lstsq)
    #    On écrit ln(y) = ln(A) + (-k)*ln(x) + (-b)*x + (-c)*thickness
    #    => ln(y) = Beta0 + Beta1*ln(x) + Beta2*x + Beta3*thickness
    # ----------------------------------------------------------------

    # Matrice de conception X (quatre colonnes : 1, ln(x), x, thickness)
    # Beta = [Beta0, Beta1, Beta2, Beta3]^T
    # Beta0 = ln(A)
    # Beta1 = -k
    # Beta2 = -b
    # Beta3 = -c
    ones = np.ones_like(xdata)
    X = np.column_stack((ones, ln_x, xdata, thickness_data))

    # Résolution du système
    Beta, residuals, rank, s = np.linalg.lstsq(X, ln_ydata, rcond=None)
    lnA_init, minus_k_init, minus_b_init, minus_c_init = Beta

    # On en déduit les valeurs initiales
    A_init = np.exp(lnA_init)
    k_init = -minus_k_init
    b_init = -minus_b_init
    c_init = -minus_c_init

    initial_guess = [lnA_init, k_init, b_init, c_init]

    # ----------------------------------------------------------------
    # 2) Ajustement final non-linéaire via curve_fit
    # ----------------------------------------------------------------

    # Définition de la fonction ln(model) pour curve_fit
    # On souhaite curve_fit(f, (x, thickness), ln_y),
    # donc f((x,th), lnA, k, b, c) = lnA - k ln(x) - b x - c thickness
    def log_model(vars, lnA, k, b, c):
        x, th = vars
        return lnA - k * np.log(x) - b * x - c * th

    # On appelle curve_fit, en lui passant (xdata, thickness_data) comme "X"
    popt, pcov = curve_fit(
        log_model,
        (xdata, thickness_data),
        ln_ydata,
        p0=initial_guess,
        maxfev=10_000
    )

    # Récupération des paramètres optimaux
    lnA_opt, k_opt, b_opt, c_opt = popt
    A_opt = np.exp(lnA_opt)

    # Incertitudes (erreurs types) sur [lnA, k, b, c]
    perr = np.sqrt(np.diag(pcov))

    # Pour convertir l’incertitude sur ln(A) en incertitude sur A :
    A_unc = A_opt * perr[0]
    k_unc = perr[1]
    b_unc = perr[2]
    c_unc = perr[3]

    uncertainties = [A_unc, k_unc, b_unc, c_unc]

    # ----------------------------------------------------------------
    # 3) Qualité de l'ajustement (R^2 basé sur ln(y))
    # ----------------------------------------------------------------
    ln_yfit = log_model((xdata, thickness_data), *popt)   # ln des valeurs prédites
    ss_res = np.sum((ln_ydata - ln_yfit)**2)
    ss_tot = np.sum((ln_ydata - np.mean(ln_ydata))**2)
    R_squared = 1 - ss_res / ss_tot

    # ----------------------------------------------------------------
    # 4) Optionnel : tracé
    #    Pour le tracé, on peut tracer TOUTES les données
    #    en différenciant chaque épaisseur par une couleur.
    # ----------------------------------------------------------------
    if plot:
        plt.figure(figsize=(8, 6))
        plt.title("Fit avec épaisseur comme variable")

        # Identifie les épaisseurs uniques
        unique_thicknesses = np.unique(thickness_data)

        # Palette de couleurs (par ex.)
        import matplotlib.cm as cm
        colors = cm.viridis(np.linspace(0, 1, len(unique_thicknesses)))

        for th_val, col in zip(unique_thicknesses, colors):
            mask = (thickness_data == th_val)
            x_sub = xdata[mask]
            y_sub = ydata[mask]

            # Données expérimentales
            plt.loglog(x_sub, y_sub, 'o', color=col, label=f"ép={th_val}mm (data)")

            # Courbe ajustée sur une gamme continue de x
            # (on peut reprendre le min et max de x_sub, ou sur l'ensemble)
            x_fit = np.logspace(np.log10(x_sub.min()), np.log10(x_sub.max()), 100)
            # dose prédite
            y_fit = A_opt * x_fit**(-k_opt) * np.exp(-b_opt*x_fit) * np.exp(-c_opt*th_val)
            plt.loglog(x_fit, y_fit, '-', color=col, label=f"ép={th_val}mm (fit)")

        plt.xlabel("Distance (m)")
        plt.ylabel("Dose (Gy)")
        plt.legend()
        plt.grid(True, which="both", ls="--")

        # Écrire la formule sur le graphe
        formula_text = (
            f"y = {A_opt:.3g} * x^(-{k_opt:.3f}) * exp(-{b_opt:.3g} x) * exp(-{c_opt:.3g} * ep)\n"
            f"R² (log-space) = {R_squared:.4f}"
        )
        plt.text(0.05, 0.05, formula_text, transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        plt.show()

    return A_opt, k_opt, b_opt, c_opt, R_squared, uncertainties
