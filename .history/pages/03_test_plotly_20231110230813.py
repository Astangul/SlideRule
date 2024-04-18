import streamlit as st
from openpyxl import load_workbook
import pandas as pd
import plotly.graph_objs as go

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv')

# fig = go.Figure([
#     go.Scatter(
#         name='Measurement',
#         x=df['Time'],
#         y=df['10 Min Sampled Avg'],
#         mode='lines',
#         line=dict(color='rgb(31, 119, 180)'),
#     ),
#     go.Scatter(
#         name='Upper Bound',
#         x=df['Time'],
#         y=df['10 Min Sampled Avg']+df['10 Min Std Dev'],
#         mode='lines',
#         marker=dict(color="#444"),
#         line=dict(width=0),
#         showlegend=False
#     ),
#     go.Scatter(
#         name='Lower Bound',
#         x=df['Time'],
#         y=df['10 Min Sampled Avg']-df['10 Min Std Dev'],
#         marker=dict(color="#444"),
#         line=dict(width=0),
#         mode='lines',
#         fillcolor='rgba(68, 68, 68, 0.3)',
#         fill='tonexty',
#         showlegend=False
#     )
# ])
# fig.update_layout(
#     yaxis_title='Wind speed (m/s)',
#     title='Continuous, variable value error bars',
#     hovermode="x"
# )
# fig.show()

# @st.cache_data
# def get_chart_90910098():
#     import plotly.graph_objects as go

#     x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=x,
#         y=[10, 20, None, 15, 10, 5, 15, None, 20, 10, 10, 15, 25, 20, 10],
#         name = '<b>No</b> Gaps', # Style name/legend entry with html tags
#         connectgaps=True # override default to connect the gaps
#     ))
#     fig.add_trace(go.Scatter(
#         x=x,
#         y=[5, 15, None, 10, 5, 0, 10, None, 15, 5, 5, 10, 20, 15, 5],
#         name='Gaps',
#     ))


#     tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
#     with tab1:
#         st.plotly_chart(fig, theme="streamlit")
#     with tab2:
#         st.plotly_chart(fig, theme=None)

@st.cache_data
def get_chart_90910098():
    import plotly.graph_objects as go

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=[10, 20, None, 15, 10, 5, 15, None, 20, 10, 10, 15, 25, 20, 10],
        name = '<b>No</b> Gaps', # Style name/legend entry with html tags
        connectgaps=True # override default to connect the gaps
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=[5, 15, None, 10, 5, 0, 10, None, 15, 5, 5, 10, 20, 15, 5],
        name='Gaps',
    ))


    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)


get_chart_90910098()


@st.cache_data
def get_chart_46283721():
    import plotly.graph_objects as go
    import numpy as np



    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_rev = x[::-1]

    # Line 1
    y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1_upper = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y1_lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y1_lower = y1_lower[::-1]

    # Line 2
    y2 = [5, 2.5, 5, 7.5, 5, 2.5, 7.5, 4.5, 5.5, 5]
    y2_upper = [5.5, 3, 5.5, 8, 6, 3, 8, 5, 6, 5.5]
    y2_lower = [4.5, 2, 4.4, 7, 4, 2, 7, 4, 5, 4.75]
    y2_lower = y2_lower[::-1]

    # Line 3
    y3 = [10, 8, 6, 4, 2, 0, 2, 4, 2, 0]
    y3_upper = [11, 9, 7, 5, 3, 1, 3, 5, 3, 1]
    y3_lower = [9, 7, 5, 3, 1, -.5, 1, 3, 1, -1]
    y3_lower = y3_lower[::-1]


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y1_upper+y1_lower,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Fair',
    ))
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y2_upper+y2_lower,
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line_color='rgba(255,255,255,0)',
        name='Premium',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y3_upper+y3_lower,
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Ideal',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y1,
        line_color='rgb(0,100,80)',
        name='Fair',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y2,
        line_color='rgb(0,176,246)',
        name='Premium',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y3,
        line_color='rgb(231,107,243)',
        name='Ideal',
    ))

    fig.update_traces(mode='lines')

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

get_chart_46283721()


def my_chart():
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv')
    fig = go.Figure([
     go.Scatter(
         name='Measurement',
         x=df['Time'],
         y=df['10 Min Sampled Avg'],
         mode='lines',
         line=dict(color='rgb(31, 119, 180)'),
     ),
     go.Scatter(
         name='Upper Bound',
         x=df['Time'],
         y=df['10 Min Sampled Avg']+df['10 Min Std Dev'],
         mode='lines',
         marker=dict(color="#444"),
         line=dict(width=0),
         showlegend=False
     ),
     go.Scatter(
         name='Lower Bound',
         x=df['Time'],
         y=df['10 Min Sampled Avg']-df['10 Min Std Dev'],
         marker=dict(color="#444"),
         line=dict(width=0),
         mode='lines',
         fillcolor='rgba(68, 68, 68, 0.3)',
         fill='tonexty',
         showlegend=False
     )
])
    

my_chart()







# myfile = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

# if myfile:
#     st.info(f"File uploaded: {myfile.name}")
#     #st.info(f"Sheet names: {myfile.sheetnames}")
#     #my_df = pd.read_excel(myfile, header = 0)
#     #st.dataframe(filter_dataframe(my_df))
