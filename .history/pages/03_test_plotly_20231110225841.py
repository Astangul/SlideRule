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
def get_chart_42947925():
    import plotly.graph_objects as go
    import numpy as np

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 3, 2, 3, 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name="linear",
                        line_shape='linear'))
    fig.add_trace(go.Scatter(x=x, y=y + 5, name="spline",
                        text=["tweak line smoothness<br>with 'smoothing' in line object"],
                        hoverinfo='text+name',
                        line_shape='spline'))
    fig.add_trace(go.Scatter(x=x, y=y + 10, name="vhv",
                        line_shape='vhv'))
    fig.add_trace(go.Scatter(x=x, y=y + 15, name="hvh",
                        line_shape='hvh'))
    fig.add_trace(go.Scatter(x=x, y=y + 20, name="vh",
                        line_shape='vh'))
    fig.add_trace(go.Scatter(x=x, y=y + 25, name="hv",
                        line_shape='hv'))

    fig.update_traces(hoverinfo='text+name', mode='lines+markers')
    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))


    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

get_chart_42947925()

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
# myfile = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

# if myfile:
#     st.info(f"File uploaded: {myfile.name}")
#     #st.info(f"Sheet names: {myfile.sheetnames}")
#     #my_df = pd.read_excel(myfile, header = 0)
#     #st.dataframe(filter_dataframe(my_df))
