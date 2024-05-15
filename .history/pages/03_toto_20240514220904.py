import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.st_filter_dataframe import df_multiselect_filters, df_selectbox_filters

# Configuration de la page Streamlit
def configure_page():
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

# Chargement des données
@st.cache_data
def load_data(sheet_name):
    data = pd.read_excel("./DB/All-at-once_DB.xlsx", sheet_name=sheet_name)
    return data

# Génération des valeurs de fission
def generate_fission_values():
    values = []
    for exponent in range(13, 21):
        base = 10 ** exponent
        values.extend([base * i for i in np.arange(1, 10, 0.1)])
    options = [f"{v:.1e}" for v in values]
    return options, f"{1e17:.1e}"

# Création du sélecteur de fission
def create_fission_selector(options, default_value):
    selected_value = st.sidebar.select_slider(
        'Select the number of fissions:',
        options=options,
        value=default_value
    )
    return float(selected_value)

# Affichage du graphique de la répartition de la dose
def plot_dose_distribution(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Distance (m)'], y=data['Dose (Gy)'], mode='lines', name='Dose'))
    fig.update_layout(title='Dose Distribution',
                      xaxis_title='Distance (m)',
                      yaxis_title='Dose (Gy)')
    st.plotly_chart(fig)

# Gestion des erreurs et anomalies
def handle_errors_and_anomalies(data):
    st.divider()
    st.header("Error Handling")
    tab4, tab5, tab6 = st.tabs(["❔ Missing values", "❗ High uncertainty (1σ>10%)", "🚨 Discrepancies"])

    with tab4:
        empty_values_df = data[data.isnull().any(axis=1)]
        st.dataframe(empty_values_df, hide_index=True)

    with tab5:
        high_uncertainty_df = data[data['1s uncertainty'] > 0.1]
        high_uncertainty_df = high_uncertainty_df.style.format({
            "1s uncertainty": "{:.2%}",
            "Dose (Gy)": "{:.2e}",
            "Absolute Uncertainty": "{:.2e}"
        })
        st.dataframe(high_uncertainty_df, hide_index=True)

    codes_uniques = data['Code'].unique()
    columns_to_group_by = ['Fissile', 'Case', 'Library', 'Flux-to-dose conversion factor', 'Screen', 'Thickness (cm)', 'Particle', 'Distance (m)']
    ecarts_significatifs = calculate_significant_discrepancies(data, columns_to_group_by)

    with tab6:
        if not ecarts_significatifs.empty():
            st.write("Relative difference exceeding 10%:")
            st.dataframe(ecarts_significatifs, hide_index=True)
        else:
            st.write("No significant discrepancies found between codes.")

# Calcul des écarts significatifs
def calculate_significant_discrepancies(data, grouping_columns):
    significant_differences = []

    grouped_data = data.groupby(grouping_columns)
    for name, group in grouped_data:
        unique_codes = group['Code'].unique()
        if len(unique_codes) > 1:
            for col in ['Dose (Gy)', '1s uncertainty']:
                max_val = group[col].max()
                min_val = group[col].min()
                if max_val != 0:
                    relative_difference = abs((max_val - min_val) / max_val)
                    if relative_difference > 0.1:
                        new_row_data = {col: relative_difference}
                        group_data = {col: val for col, val in zip(grouping_columns, name)}
                        full_row_data = {**group_data, **new_row_data}
                        significant_differences.append(full_row_data)

    return pd.DataFrame(significant_differences)

# Application principale
def main():
    configure_page()
    st.title("Slide Rule")
    st.write("My first Streamlit app")

    data = load_data('screen')
    options, default_value = generate_fission_values()
    selected_value_float = create_fission_selector(options, default_value)
    st.write(f"Number of fissions selected: {selected_value_float:.1e}")

    plot_dose_distribution(data)

    st.divider()
    st.header("Filtering options")
    filtered_data = df_multiselect_filters(data)
    filtered_data = df_selectbox_filters(filtered_data)
    st.dataframe(filtered_data)

    handle_errors_and_anomalies(data)

if __name__ == "__main__":
    main()
