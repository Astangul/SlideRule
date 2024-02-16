
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openpyxl import load_workbook

# Set page configuration
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

st.title("Slide Rule")
st.write("My first Streamlit app")

# Function to load data
@st.cache(allow_output_mutation=True)
def load_data(filepath):
    data = pd.read_excel(filepath)
    data['Filter Combo'] = data[['Particle', 'Screen', 'Code', 'Case', 'Thickness (cm)']].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    return data

# Load data
data = load_data("./DB/All-at-once_DB.xlsx")

# Slider for number of fissions
fissions = st.sidebar.select_slider(
    'Select the number of fissions:',
    options=np.logspace(16, 20, num=40, base=10).astype(int),
    format="%.1e"
)

st.write(f"Number of fissions selected: {fissions:.1e}")
dose_multiplier = fissions / 1e17

data["Absolute Uncertainty"] = data["Dose (Gy)"] * data["1s uncertainty"] * dose_multiplier
data["Dose (Gy)"] = data["Dose (Gy)"] * dose_multiplier

# Parameter selection section
st.header("Parameter Selection")
with st.expander("↳ Click to expand/collapse", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        # Multiple selection filters
        filter_options = {col: sorted(data[col].unique()) for col in ['Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)']}
        filters = {f'Select {col}': st.multiselect(f'{col}:', options, default=options[0]) for col, options in filter_options.items()}
    
    with col2:
        # Single selection filters for reference case
        ref_filters = {f'Reference {col}': st.selectbox(f'{col}:', options) for col, options in filter_options.items()}

# Data preparation for plots
def prepare_data(data, filters):
    query = ' & '.join([f'{col} == "{val}"' for col, val in filters.items() if val is not None])
    return data.query(query) if query else data

ref_data = prepare_data(data, ref_filters)
compare_data = prepare_data(data, filters)

# Plots
def plot_dose_graph(ref_data, compare_data):
    fig = go.Figure()
    # Add traces for reference and comparison data
    fig.add_trace(go.Scatter(x=ref_data["Distance (m)"], y=ref_data["Dose (Gy)"], name='Reference', mode='lines+markers'))
    for name, group in compare_data.groupby('Filter Combo'):
        fig.add_trace(go.Scatter(x=group["Distance (m)"], y=group["Dose (Gy)"], name=name, mode='lines+markers'))
    
    fig.update_layout(hovermode='x unified', height=700)
    st.plotly_chart(fig, use_container_width=True)
