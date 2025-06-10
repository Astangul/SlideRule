import colorsys
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.utils_func_st import generate_filter_combinations, generate_series_name
# ___________________________________________________________


def sort_groups_by_first_numeric_column(grouped_data):
    if not grouped_data:
        return grouped_data  # Retourner la liste vide si aucun groupe n'est présent

    # Check for the first group to determine the numeric columns
    first_group = grouped_data[0][1]
    numeric_columns = first_group.select_dtypes(include=[np.number]).columns

    # If there are numeric columns, sort by the first numeric column
    if not numeric_columns.empty:
        first_numeric_column = numeric_columns[0]
        grouped_data.sort(key=lambda x: x[1][first_numeric_column].iloc[0])
    return grouped_data


#  ______________________________________ 
def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

def hex_to_complementary_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    complementary_rgb = [255 - x for x in rgb]
    return f'rgba({complementary_rgb[0]}, {complementary_rgb[1]}, {complementary_rgb[2]}, {alpha})'

def generate_analogous_colors(base_color, num_colors=3, spread=0.1):
    """
    Generate a list of analogous colors based on the base color.
    :param base_color: Hex color string (e.g., "#ff5733")
    :param num_colors: Number of analogous colors to generate.
    :param spread: How much to spread the analogous colors around the base color's hue.
    :return: List of hex color strings.
    """
    if num_colors == 1:
        return [base_color]

    r = int(base_color[1:3], 16)
    g = int(base_color[3:5], 16)
    b = int(base_color[5:7], 16)

    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    
    # Generate colors by adjusting the hue within the spread range
    analogous_colors = []
    for i in range(num_colors):
        hue_adjustment = spread * (i - (num_colors // 2)) / (num_colors - 1)
        new_h = (h + hue_adjustment) % 1.0
        r, g, b = colorsys.hls_to_rgb(new_h, l, s)
        analogous_colors.append('#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)))

    return analogous_colors

# ________________________________________________________________________________
def dose_scatter_plot(ref_data, ref_label, compare_data, colors):
    log_x = st.toggle("X-axis log scale", value=True, key="log_x_fig1")
    log_y = st.toggle("Y-axis log scale", value=True, key="log_y_fig1")
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=ref_data["Distance (m)"],
                              y=ref_data["Dose (Gy)"],
                              mode='lines+markers', 
                              marker_symbol='diamond', marker_size=8,
                              name=ref_label,
                              error_y=dict(type='data', array=2*ref_data["Absolute Uncertainty"], visible=True)
                              ))
    # Group the data by 'unique_key' and sort the groups by the first numeric column
    grouped_data = list(compare_data.groupby('unique_key'))
    grouped_data = sort_groups_by_first_numeric_column(grouped_data)

    # Loop through each sorted group
    for index, (key, group) in enumerate(grouped_data):
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

    # Merge and calculate dose ratio and combined uncertainty for all data
    compare_data = compare_data.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
    compare_data['Dose Ratio'] = compare_data['Dose (Gy)'] / compare_data['Dose (Gy)_ref']
    compare_data['Combined Uncertainty'] = np.sqrt(np.square(compare_data['1s uncertainty']) + np.square(compare_data['1s uncertainty_ref']))
    compare_data['Absolute Combined Uncertainty'] = compare_data['Combined Uncertainty'] * compare_data['Dose Ratio']

    # Convert distances to strings for categorical x-axis
    compare_data['Distance (m)'] = compare_data['Distance (m)'].astype(str)

    # Group the data by 'unique_key' and sort the groups by the first numeric column
    grouped_data = list(compare_data.groupby('unique_key'))
    grouped_data = sort_groups_by_first_numeric_column(grouped_data)

    # Loop through each sorted group
    for index, (key, group) in enumerate(grouped_data):
        color = colors[index % len(colors)]
        rgba_color = hex_to_rgba(color, alpha=0.3)

        upper_bound = group['Dose Ratio'] + group['Absolute Combined Uncertainty']
        lower_bound = group['Dose Ratio'] - group['Absolute Combined Uncertainty']

        legend_group = f"group_{index}"
        hovertemplate = '%{y:.3f} ± %{customdata:.2e}'

        # Plotting the main line
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

        # Adding error bands
        fig2.add_trace(go.Scatter(
            x=group["Distance (m)"], y=upper_bound, mode='lines', 
            line=dict(width=0), hoverinfo='none', showlegend=False, 
            legendgroup=legend_group, fillcolor=rgba_color))
        fig2.add_trace(go.Scatter(
            x=group["Distance (m)"], y=lower_bound, mode='lines', 
            line=dict(width=0), fill='tonexty', fillcolor=rgba_color, 
            hoverinfo='none', showlegend=False, legendgroup=legend_group))

    # Update layout
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

    # Toggle for logarithmic scale on X-axis
    log_x_fig2 = st.toggle("X-axis log scale", value=True, key="log_x_fig2")
    # Set X-axis scale
    if log_x_fig2:
        fig2.update_xaxes(type='log', title="Distance (m) [Log10]")
    else:
        fig2.update_xaxes(type='linear', title="Distance (m)")

    st.plotly_chart(fig2, use_container_width=True)

def dose_ratio_bar_chart(compare_data, ref_data, colors):
    title = "Dose Ratio"
    
    fig2 = go.Figure()

    # Merge and calculate dose ratio and combined uncertainty for all data
    compare_data = compare_data.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
    compare_data['Dose Ratio'] = compare_data['Dose (Gy)'] / compare_data['Dose (Gy)_ref']
    compare_data['Combined Uncertainty'] = np.sqrt(np.square(compare_data['1s uncertainty']) + np.square(compare_data['1s uncertainty_ref']))
    compare_data['Absolute Combined Uncertainty'] = compare_data['Combined Uncertainty'] * compare_data['Dose Ratio']

    # Convert distances to strings for categorical x-axis
    compare_data['Distance (m)'] = compare_data['Distance (m)'].astype(str)

    # Group the data by 'unique_key' and sort the groups by the first numeric column
    grouped_data = list(compare_data.groupby('unique_key'))
    grouped_data = sort_groups_by_first_numeric_column(grouped_data)

    # Loop through each sorted group
    for index, (key, group) in enumerate(grouped_data):
        color = colors[index % len(colors)]
        rgba_color = hex_to_rgba(color, alpha=1.0)  # Adding transparency to the bar colors
        complementary_color = hex_to_complementary_rgba(color)

        hovertemplate = '%{y:.3f} ± %{customdata:.2e}'

        # Plotting the bar chart
        fig2.add_trace(go.Bar(
            x=group["Distance (m)"],
            y=group["Dose Ratio"],
            customdata=group['Absolute Combined Uncertainty'],
            error_y=dict(
                type='data', 
                array=group['Absolute Combined Uncertainty'], 
                visible=True,
                thickness=2,  # Thicker error bars for visibility
                color=complementary_color
            ),
            name=key,
            marker=dict(
                color=rgba_color,  # Transparent color for bar fill
                line=dict(color=color, width=2)  # Solid color for bar border
            ),
            hovertemplate=hovertemplate
        ))

    # Update layout
    fig2.update_layout(
        hovermode='x', 
        height=700, 
        showlegend=True, 
        legend_title="Click on legends below to hide/show:", 
        xaxis=dict(
            title="Distance (m)",
            type='category'  # Use categorical x-axis
        ),
        yaxis=dict(
            title=title,
            tickmode='auto',
            minor=dict(
                ticks="inside",
                ticklen=6,
                griddash='dot',
                showgrid=True
            )
        ),
        barmode='group'  # Group the bars by unique_key
    )

    st.plotly_chart(fig2, use_container_width=True)

def dose_scatter_plot_2(data, filters, colors):
    log_x = st.toggle("X-axis log scale", value=True, key="log_x_fig1")
    log_y = st.toggle("Y-axis log scale", value=True, key="log_y_fig1")
    fig1 = go.Figure()
    
    data['unique_key'] = data.apply(lambda row: '_'.join([str(row[col]) for col in filters.keys()]), axis=1)

    # Group the data by 'unique_key' and sort the groups by the first numeric column
    grouped_data = list(data.groupby('unique_key'))
    grouped_data = sort_groups_by_first_numeric_column(grouped_data)

    # Loop through each sorted group
    for index, (key, group) in enumerate(grouped_data):
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
        # legend_title="Click on legends below to hide/show:",
        legend=dict(
        title="Click on legends below to hide/show:",
        orientation="h",  # horizontal layout
        yanchor="bottom",  # anchor the legend to the bottom of its box
        y=1.02,  # position it slightly above the top of the plot area
        xanchor="center",  # center the legend horizontally
        x=0.5)  # center the legend
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

def dose_scatter_plot_3(data, filters, colors):
    """
    Génère un graphique de dispersion des doses avec Plotly, et renvoie l'objet figure.
    
    Args:
        data (pd.DataFrame): Données à tracer.
        filters (dict): Filtres appliqués aux données.
        colors (list): Liste de couleurs pour les courbes.
    
    Returns:
        go.Figure: L'objet figure Plotly.
    """
    # log_x = st.toggle("X-axis log scale", value=True, key="log_x_fig1")
    # log_y = st.toggle("Y-axis log scale", value=True, key="log_y_fig1")
    
    log_x = st.session_state.get("log_x_fig1", True)
    log_y = st.session_state.get("log_y_fig1", True)
    
    fig = go.Figure()

    # Déterminer dynamiquement les colonnes catégoriques à utiliser (exclut les colonnes numériques)
    ignored_columns = ["Distance (m)", "Dose (Gy)", "1s uncertainty", "Absolute Uncertainty"]
    all_categorical_columns = [col for col in data.columns if col not in ignored_columns]

    # Détecter les colonnes constantes (même valeur pour toutes les lignes)
    constant_columns = [col for col in all_categorical_columns if data[col].nunique() == 1]

    # Garder uniquement les colonnes catégoriques qui varient
    categorical_columns = [col for col in all_categorical_columns if col not in constant_columns]

    # Construire dynamiquement la clé unique en excluant les colonnes constantes
    data = data.copy()
    data["unique_key"] = data.apply(lambda row: '_'.join([f"{row[col]}" for col in categorical_columns]), axis=1)

    # Regrouper les données par 'unique_key' et trier
    grouped_data = list(data.groupby("unique_key"))
    grouped_data = sort_groups_by_first_numeric_column(grouped_data)

    # Boucle pour tracer les courbes
    for index, (key, group) in enumerate(grouped_data):
        fig.add_trace(go.Scatter(
            x=group["Distance (m)"],
            y=group["Dose (Gy)"],
            mode='lines+markers',
            marker_symbol='circle-dot',
            marker_size=8,
            line=dict(dash='dash', color=colors[index % len(colors)]),
            name=f'{key}',  # Clé unique sans colonnes constantes
            error_y=dict(type='data', array=2 * group["Absolute Uncertainty"], visible=True),
            visible='legendonly'  # Séries cachées par défaut
        ))

    # Mise à jour de la mise en page
    fig.update_layout(
        hovermode='x',
        showlegend=True,
        height=700,
        xaxis={'showgrid': True},
        yaxis={'showgrid': True},
        legend=dict(
            title="Click on legends below to hide/show:",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Mise à jour des axes (logarithmique ou linéaire selon l'option de l'utilisateur)
    if log_x:
        fig.update_xaxes(type='log', title="Distance (m) [Log10]", minor=dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))
    else:
        fig.update_xaxes(type='linear', title="Distance (m)", minor=dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))

    if log_y:
        fig.update_yaxes(type='log', title="Dose (Gy) ± 2σ [Log10]", tickformat='.2e', minor=dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))
    else:
        fig.update_yaxes(type='linear', title="Dose (Gy) ± 2σ", tickformat='.2e', minor=dict(ticks="inside", ticklen=6, griddash='dot', showgrid=True))

    return fig

def dose_ratio_scatter_plot_2(compare_data, compare_filters, ref_data, ref_filters, color, fig, series_number):
    compare_filter_combinations = generate_filter_combinations(compare_filters)
    
    # Génération de couleurs analogues
    num_cases = len(compare_filter_combinations)
    analogous_colors = generate_analogous_colors(color, num_colors=num_cases, spread=0.3)
    
    # Création de clés uniques pour grouper les données
    compare_data = compare_data.copy()
    compare_data.loc[:, 'unique_key'] = compare_data.apply(lambda row: '_'.join([str(row[col]) for col in compare_filters.keys()]), axis=1)
    ref_data['unique_key'] = ref_data.apply(lambda row: '_'.join([str(row[col]) for col in ref_filters.keys()]), axis=1)

    # Fusion des données et calcul du ratio de dose et de l'incertitude combinée
    merged_data = compare_data.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
    merged_data['Dose Ratio'] = merged_data['Dose (Gy)'] / merged_data['Dose (Gy)_ref']
    merged_data['Combined Uncertainty'] = np.sqrt(np.square(merged_data['1s uncertainty']) + np.square(merged_data['1s uncertainty_ref']))
    merged_data['Absolute Combined Uncertainty'] = merged_data['Combined Uncertainty'] * merged_data['Dose Ratio']

    # Conversion des distances en chaînes pour un axe x catégoriel
    merged_data['Distance (m)'] = merged_data['Distance (m)'].astype(str)

    # Grouper les données par 'unique_key' et trier
    grouped_data = list(merged_data.groupby('unique_key'))
    grouped_data = sort_groups_by_first_numeric_column(grouped_data)

    # Parcourir chaque groupe trié
    for index, (key, group) in enumerate(grouped_data):
        series_name = generate_series_name(series_number, compare_filter_combinations[index], ref_filters)
        # Utiliser une couleur analogue pour chaque cas
        adjusted_color = analogous_colors[index]  
        rgba_color = hex_to_rgba(adjusted_color, alpha=0.3)
        legend_group = f"series_{series_number}"  # Groupe de légende unique basé sur le numéro de série
        series_name = f"{series_name}"

        upper_bound = group['Dose Ratio'] + group['Absolute Combined Uncertainty']
        lower_bound = group['Dose Ratio'] - group['Absolute Combined Uncertainty']
        
        # Tracé de la ligne principale
        fig.add_trace(go.Scatter(
            x=group["Distance (m)"],
            y=group["Dose Ratio"],
            legendgroup=legend_group,
            name=series_name,
            customdata=group['Absolute Combined Uncertainty'],
            error_y=dict(type='data', array=group['Absolute Combined Uncertainty'], visible=True),  
            mode='lines+markers',
            marker_symbol='circle-open-dot',
            marker_size=8,
            line=dict(dash='dash', color=adjusted_color),
            hovertemplate='%{y:.3f} ± %{customdata:.2e}'
        ))

        # Ajout des bandes d'erreur
        fig.add_trace(go.Scatter(
            x=group["Distance (m)"], y=upper_bound, mode='lines', 
            line=dict(width=0), hoverinfo='none', showlegend=False, 
            legendgroup=legend_group, fillcolor=rgba_color))
        fig.add_trace(go.Scatter(
            x=group["Distance (m)"], y=lower_bound, mode='lines', 
            line=dict(width=0), fill='tonexty', fillcolor=rgba_color, 
            hoverinfo='none', showlegend=False, legendgroup=legend_group))

    # Mise à jour de la configuration du graphique
    fig.update_layout(
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

def dose_ratio_bar_chart_2(compare_data, compare_filters, ref_data, ref_filters, color, fig2, series_number):
    compare_filter_combinations = generate_filter_combinations(compare_filters)

    title = "Dose Ratio"

    # Génération de couleurs analogues
    num_cases = len(compare_filter_combinations)
    analogous_colors = generate_analogous_colors(color, num_colors=num_cases, spread=0.2)

    # Création de clés uniques pour identifier chaque ligne de manière unique
    compare_data = compare_data.copy()
    compare_data.loc[:, 'unique_key'] = compare_data.apply(lambda row: '_'.join([str(row[col]) for col in compare_filters.keys()]), axis=1)
    ref_data['unique_key'] = ref_data.apply(lambda row: '_'.join([str(row[col]) for col in ref_filters.keys()]), axis=1)

    # Fusion des données et calcul du ratio de dose et de l'incertitude combinée
    merged_data = compare_data.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
    merged_data['Dose Ratio'] = merged_data['Dose (Gy)'] / merged_data['Dose (Gy)_ref']
    merged_data['Combined Uncertainty'] = np.sqrt(np.square(merged_data['1s uncertainty']) + np.square(merged_data['1s uncertainty_ref']))
    merged_data['Absolute Combined Uncertainty'] = merged_data['Combined Uncertainty'] * merged_data['Dose Ratio']

    # Conversion des distances pour l'axe des x catégoriel
    merged_data['Distance (m)'] = merged_data['Distance (m)'].astype(str)

    # Regroupement des données par 'unique_key'
    grouped_data = merged_data.groupby('unique_key')

    # Boucle à travers chaque groupe trié
    for index, (key, group) in enumerate(grouped_data):
        series_name = generate_series_name(series_number, compare_filter_combinations[index], ref_filters)
        # Utilisation d'une couleur analogue pour chaque cas
        adjusted_color = analogous_colors[index]  
        rgba_color = hex_to_rgba(adjusted_color, alpha=0.75)  # Utilisation d'une certaine transparence pour les couleurs des barres
        complementary_color = hex_to_complementary_rgba(adjusted_color)
        legend_group = f"series_{series_number}"  # Groupe de légende unique pour chaque série
        series_name = f"{series_name}"  

        hovertemplate = '%{y:.3f} ± %{customdata:.2e}'

        # Tracé du graphique à barres
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
            legendgroup=legend_group,
            name=series_name,
            marker=dict(color=rgba_color, line=dict(color=adjusted_color, width=2)),
            hovertemplate=hovertemplate
        ))

    # Mise à jour de la mise en page
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
            title=title,
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

# def dose_ratio_bar_chart_2(compare_data, compare_filters, ref_data, ref_filters, color, fig2, series_number):
#     compare_filter_combinations = generate_filter_combinations(compare_filters)

#     title = "Dose Ratio (centered on 0)"

#     # Génération de couleurs analogues
#     num_cases = len(compare_filter_combinations)
#     analogous_colors = generate_analogous_colors(color, num_colors=num_cases, spread=0.2)

#     # Création de clés uniques pour identifier chaque ligne de manière unique
#     compare_data = compare_data.copy()
#     compare_data.loc[:, 'unique_key'] = compare_data.apply(lambda row: '_'.join([str(row[col]) for col in compare_filters.keys()]), axis=1)
#     ref_data['unique_key'] = ref_data.apply(lambda row: '_'.join([str(row[col]) for col in ref_filters.keys()]), axis=1)

#     # Fusion des données et calcul du ratio de dose centré et de l'incertitude combinée
#     merged_data = compare_data.merge(ref_data, on="Distance (m)", suffixes=('', '_ref'))
#     merged_data['Dose Ratio'] = (merged_data['Dose (Gy)'] / merged_data['Dose (Gy)_ref']) - 1
#     merged_data['Combined Uncertainty'] = np.sqrt(np.square(merged_data['1s uncertainty']) + np.square(merged_data['1s uncertainty_ref']))
#     merged_data['Absolute Combined Uncertainty'] = merged_data['Combined Uncertainty'] * (merged_data['Dose Ratio'] + 1)  # Ajustement pour refléter le ratio centré

#     # Conversion des distances pour l'axe des x catégoriel
#     merged_data['Distance (m)'] = merged_data['Distance (m)'].astype(str)

#     # Regroupement des données par 'unique_key'
#     grouped_data = merged_data.groupby('unique_key')

#     # Boucle à travers chaque groupe trié
#     for index, (key, group) in enumerate(grouped_data):
#         series_name = generate_series_name(series_number, compare_filter_combinations[index], ref_filters)
#         # Utilisation d'une couleur analogue pour chaque cas
#         adjusted_color = analogous_colors[index]  
#         rgba_color = hex_to_rgba(adjusted_color, alpha=0.75)  # Utilisation d'une certaine transparence pour les couleurs des barres
#         complementary_color = hex_to_complementary_rgba(adjusted_color)
#         legend_group = f"series_{series_number}"  # Groupe de légende unique pour chaque série
#         series_name = f"{series_name}"  

#         hovertemplate = '%{y:.3f} ± %{customdata:.2e}'

#         # Tracé du graphique à barres
#         fig2.add_trace(go.Bar(
#             x=group["Distance (m)"],
#             y=group["Dose Ratio"],
#             customdata=group['Absolute Combined Uncertainty'],
#             error_y=dict(
#                 type='data', 
#                 array=group['Absolute Combined Uncertainty'], 
#                 visible=True,
#                 thickness=2,
#                 color=complementary_color
#             ),
#             legendgroup=legend_group,
#             name=series_name,
#             marker=dict(color=rgba_color, line=dict(color=adjusted_color, width=2)),
#             hovertemplate=hovertemplate
#         ))

#     # Mise à jour de la mise en page
#     fig2.update_layout(
#         hovermode='x', 
#         height=700, 
#         showlegend=True, 
#         legend_title="Click on legends below to hide/show:", 
#         xaxis=dict(
#             title="Distance (m)",
#             type='category'
#         ),
#         yaxis=dict(
#             title=title,
#             tickmode='auto',
#             minor=dict(
#                 ticks="inside",
#                 ticklen=6,
#                 griddash='dot',
#                 showgrid=True
#             )
#         ),
#         barmode='group'
#     )
