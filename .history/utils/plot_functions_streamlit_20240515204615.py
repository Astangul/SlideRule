import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

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


def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

def hex_to_complementary_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    complementary_rgb = [255 - x for x in rgb]
    return f'rgba({complementary_rgb[0]}, {complementary_rgb[1]}, {complementary_rgb[2]}, {alpha})'