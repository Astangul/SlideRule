import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.st_filter_dataframe import df_multiselect_filters, df_selectbox_filters

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

st.write("My first Streamlit app")

# Chargement des données avec mise en cache
@st.cache_data
def load_data(sheet_name):
    return pd.read_excel("./DB/All-at-once_DB.xlsx", sheet_name=sheet_name)

data = load_data('screen')

# Création des valeurs pour le select_slider
values = []
for exponent in range(13, 21):  # Pour 1.0E13 à 1.0E20
    base = 10 ** exponent
    values.extend([base * i for i in np.arange(1, 10, 0.1)])  # 1.0, 1.1, ..., 9.9

# Conversion des valeurs en chaînes de caractères pour l'affichage
options = [f"{v:.1e}" for v in values]
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

fissions_number_slider = float(selected_value)
fissions_number_input = st.sidebar.number_input(
    "OR enter number of fissions",
    min_value=1.0e+13,
    max_value=1.0e+20,
    value=fissions_number_slider,
    step=None,
    format="%.1e",
    key="fission_input",
    on_change=update_fission_slider
)

# Calcul du facteur de multiplication des doses
dose_multiplier = fissions_number_input / 1e17
st.write(f"Selected number of fissions: {fissions_number_input:.1e}")
st.write(f"Dose multiplier: {dose_multiplier}")

# Mise à jour des données en fonction du facteur de multiplication
data["Absolute Uncertainty"] = data["Dose (Gy)"] * data["1s uncertainty"] * dose_multiplier
data["Dose (Gy)"] = data["Dose (Gy)"] * dose_multiplier

# Définition des couleurs
colors = [
    '#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616',
    '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5',
    '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72'
]

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

def hex_to_complementary_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    complementary_rgb = [255 - x for x in rgb]
    return f'rgba({complementary_rgb[0]}, {complementary_rgb[1]}, {complementary_rgb[2]}, {alpha})'

st.divider()
st.header("Parameter selection")

with st.expander("Click to expand/collapse", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Series to compare")
        compare_data, compare_filters = df_multiselect_filters(data, default_columns=['Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)'], key="compare_series")
    with col2:
        st.subheader("Reference case")
        ref_data, ref_filters = df_selectbox_filters(data, default_columns=['Case', 'Code', 'Particle', 'Screen', 'Thickness (cm)'], key="reference_case")

ref_criteria = {col: ref_data.iloc[0][col] for col in ref_filters.keys()}
condition = np.logical_and.reduce([compare_data[col].eq(value) for col, value in ref_criteria.items()])
compare_data = compare_data[~condition]
combined_data = pd.concat([ref_data, compare_data]).drop_duplicates()

ref_label_parts = [str(value[0]) for key, value in ref_filters.items() if value]
ref_label = "_".join(ref_label_parts) + " (reference)"
compare_data['unique_key'] = compare_data.apply(lambda row: '_'.join([str(row[col]) for col in compare_filters.keys()]), axis=1)

formatted_data = combined_data.style.format({
    "1s uncertainty": "{:.2%}",
    "Dose (Gy)": "{:.2e}",
    "Absolute Uncertainty": "{:.2e}"
})

tab1, tab2, tab3 = st.tabs(["📈 Graph", "🆚 Comparison", "🔢 Dataframe"])

with tab3:
    st.dataframe(formatted_data, hide_index=True)

with tab1:
    log_x = st.toggle("X-axis log scale", value=True, key="log_x_fig1")
    log_y = st.toggle("Y-axis log scale", value=True, key="log_y_fig1")
    st.write(f"Estimated prompt dose based on total fissions: {fissions_number_input:.2e}")
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=ref_data["Distance (m)"],
                              y=ref_data["Dose (Gy)"],
                              mode='lines+markers', 
                              marker_symbol='diamond', marker_size=8,
                              name=ref_label,
                              error_y=dict(type='data', array=2*ref_data["Absolute Uncertainty"], visible=True)
                              ))

    for index, (key, group) in enumerate(compare_data.groupby('unique_key')):
        fig1.add_trace(go.Scatter(x=group["Distance (m)"], 
                                  y=group["Dose (Gy)"],
                                  mode='lines+markers', 
                                  marker_symbol='circle-dot', marker_size=8,
                                  line=dict(dash='dash', color=colors[index % len(colors)]), 
                                  name=f'{key}',
                                  error_y=dict(type='data', array=2*group["Absolute Uncertainty"], visible=True)
                                  ))

    fig1.update_layout(
        hovermode='x',
        showlegend=True,
        xaxis={'showgrid':True},
        yaxis={'showgrid':True},
        height=700,
        legend_title="Click on legends below to hide/show:",
    )
    fig1.update_xaxes(minor=dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))
    fig1.update_yaxes(minor=dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))

    if log_x:
        fig1.update_xaxes(type='log', title="Distance (m) [Log10]")
    else:
        fig1.update_xaxes(type='linear', title="Distance (m)")
    
    if log_y:
        fig1.update_yaxes(type='log', title="Dose (Gy) ± 2σ [Log10]", tickformat='.2e')
    else:
        fig1.update_yaxes(type='linear', title="Dose (Gy) ± 2σ", tickformat='.2e')

    st.plotly_chart(fig1, use_container_width=True)

def dose_ratio_scatter_plot(compare_data, ref_data, colors):
    fig2 = go.Figure()

    compare_data = compare_data.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
    compare_data['Dose Ratio'] = compare_data['Dose (Gy)'] / compare_data['Dose (Gy)_ref']
    compare_data['Combined Uncertainty'] = np.sqrt(np.square(compare_data['1s uncertainty']) + np.square(compare_data['1s uncertainty_ref']))
    compare_data['Absolute Combined Uncertainty'] = compare_data['Combined Uncertainty'] * compare_data['Dose Ratio']
    compare_data['Distance (m)'] = compare_data['Distance (m)'].astype(str)

    grouped_data = list(compare_data.groupby('unique_key'))
    grouped_data = sort_groups_by_first_numeric_column(grouped_data)

    for index, (key, group) in enumerate(grouped_data):
        color = colors[index % len(colors)]
        rgba_color = hex_to_rgba(color, alpha=0.3)

        upper_bound = group['Dose Ratio'] + group['Absolute Combined Uncertainty']
        lower_bound = group['Dose Ratio'] - group['Absolute Combined Uncertainty']

        legend_group = f"group_{index}"
        hovertemplate = '%{y:.3f} ± %{customdata:.2e}'

        fig2.add_trace(go.Scatter(
            x=group["Distance (m)"],
            y=group["Dose Ratio"],
            customdata=group['Absolute Combined Uncertainty'],
            error_y=dict(type='data', array=group['Absolute Combined Uncertainty'], visible=True),  
            mode='lines+markers',
            marker_symbol='circle-open-dot', marker_size=8,
            line=dict(dash='dash', color=color),
            name=key,
            legendgroup=legend_group,
            hovertemplate=hovertemplate
        ))

        fig2.add_trace(go.Scatter(
            x=group["Distance (m)"], y=upper_bound, mode='lines', 
            line=dict(width=0), hoverinfo='none', showlegend=False, 
            legendgroup=legend_group, fillcolor=rgba_color))
        fig2.add_trace(go.Scatter(
            x=group["Distance (m)"], y=lower_bound, mode='lines', 
            line=dict(width=0), fill='tonexty', fillcolor=rgba_color, 
            hoverinfo='none', showlegend=False, legendgroup=legend_group))

    fig2.update_layout(
        hovermode='x', 
        height=700, 
        showlegend=True, 
        legend_title="Click on legends below to hide/show:", 
        xaxis={'showgrid': True}, 
        yaxis={'showgrid': True, 
               'title': "Dose Ratio",
               'tickmode': 'auto',
               'minor': {
                   'ticks': "inside",
                   'ticklen': 6,
                   'griddash': 'dot',
                   'showgrid': True
               }
               })

    log_x_fig2 = st.toggle("X-axis log scale", value=True, key="log_x_fig2")
    if log_x_fig2:
        fig2.update_xaxes(type='log', title="Distance (m) [Log10]")
    else:
        fig2.update_xaxes(type='linear', title="Distance (m)")

    st.plotly_chart(fig2, use_container_width=True)

def dose_ratio_bar_chart(compare_data, ref_data, colors):
    fig2 = go.Figure()

    compare_data = compare_data.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
    compare_data['Dose Ratio'] = compare_data['Dose (Gy)'] / compare_data['Dose (Gy)_ref']
    compare_data['Combined Uncertainty'] = np.sqrt(np.square(compare_data['1s uncertainty']) + np.square(compare_data['1s uncertainty_ref']))
    compare_data['Absolute Combined Uncertainty'] = compare_data['Combined Uncertainty'] * compare_data['Dose Ratio']
    compare_data['Distance (m)'] = compare_data['Distance (m)'].astype(str)

    grouped_data = list(compare_data.groupby('unique_key'))
    grouped_data = sort_groups_by_first_numeric_column(grouped_data)

    for index, (key, group) in enumerate(grouped_data):
        color = colors[index % len(colors)]
        rgba_color = hex_to_rgba(color, alpha=0.6)  # Transparence ajoutée aux couleurs des barres
        complementary_color = hex_to_complementary_rgba(color)

        hovertemplate = '%{y:.3f} ± %{customdata:.2e}'

        fig2.add_trace(go.Bar(
            x=group["Distance (m)"],
            y=group["Dose Ratio"],
            customdata=group['Absolute Combined Uncertainty'],
            error_y=dict(
                type='data', 
                array=group['Absolute Combined Uncertainty'], 
                visible=True,
                thickness=2,
                color=complementary_color
            ),
            name=key,
            marker=dict(
                color=rgba_color,
                line=dict(color=color, width=2)
            ),
            hovertemplate=hovertemplate
        ))

    fig2.update_layout(
        hovermode='x', 
        height=700, 
        showlegend=True, 
        legend_title="Click on legends below to hide/show:", 
        xaxis=dict(
            title="Distance (m)",
            type='category'
        ),
        yaxis=dict(
            title="Dose Ratio",
            tickmode='auto',
            minor=dict(
                ticks="inside",
                ticklen=6,
                griddash='dot',
                showgrid=True
            )
        ),
        barmode='group'
    )

    st.plotly_chart(fig2, use_container_width=True)

def sort_groups_by_first_numeric_column(grouped_data):
    first_group = grouped_data[0][1]
    numeric_columns = first_group.select_dtypes(include=[np.number]).columns

    if not numeric_columns.empty:
        first_numeric_column = numeric_columns[0]
        grouped_data.sort(key=lambda x: x[1][first_numeric_column].iloc[0])
    return grouped_data

with tab2:
    plot_type = st.radio("Select plot type:", ["Scatter Plot", "Bar Chart"])
    
    if plot_type == "Scatter Plot":
        dose_ratio_scatter_plot(compare_data, ref_data, colors)
    elif plot_type == "Bar Chart":
        dose_ratio_bar_chart(compare_data, ref_data, colors)

@st.cache_data
def calculate_significant_discrepancies(input_data, grouping_columns):
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
                    new_row_data = {
                        'Code1': code1,
                        'Code2': code2,
                        'Dose_Code1': f"{dose_code1:.2e}",
                        'Dose_Code2': f"{dose_code2:.2e}",
                        'Relative_Difference': f"{ecart_relatif:.2%}"
                    }
                    
                    group_data = {}
                    if isinstance(name, tuple):
                        for col, val in zip(grouping_columns, name):
                            group_data[col] = val
                    else:
                        group_data[grouping_columns[0]] = name
                    
                    full_row_data = {**group_data, **new_row_data}
                    significant_differences.append(full_row_data)

    return pd.DataFrame(significant_differences)

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
    if not ecarts_significatifs.empty:
        st.write("Relative difference exceeding 10%:")
        st.dataframe(ecarts_significatifs, hide_index=True)
    else:
        st.write("No significant discrepancies found between codes.")
