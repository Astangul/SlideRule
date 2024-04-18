import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openpyxl import load_workbook
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

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

def filter_categorical(column, data):
    user_cat_input = st.multiselect(
        f"Values for {column}",
        data.unique(),
        default=list(data.unique()),
        key=f"cat_input_{column}"
    )
    return user_cat_input

def filter_numeric(column, data):
    _min = float(data.min())
    _max = float(data.max())
    step = (_max - _min) / 100
    user_num_input = st.slider(
        f"Values for {column}",
        _min,
        _max,
        (_min, _max),
        step=step,
        key=f"num_input_{column}"
    )
    return user_num_input

def filter_date(column, data):
    user_date_input = st.date_input(
        f"Values for {column}",
        value=(data.min(), data.max()),
        key=f"date_input_{column}"
    )
    return user_date_input

def filter_text(column):
    user_text_input = st.text_input(
        f"Substring or regex in {column}",
        key=f"text_input_{column}"
    )
    return user_text_input

def filter_dataframe(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
        dict: Filter criteria
    """
    df = df.copy()

    filter_criteria = {}  # Store filter criteria

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, key='filter_columns')

        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")

            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = filter_categorical(column, df[column])
                filter_criteria[column] = user_cat_input
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                user_num_input = filter_numeric(column, df[column])
                filter_criteria[column] = user_num_input
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = filter_date(column, df[column])
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    filter_criteria[column] = (start_date, end_date)
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = filter_text(column)
                if user_text_input:
                    filter_criteria[column] = user_text_input
                    df = df[df[column].str.contains(user_text_input)]

    return df, filter_criteria

# Streamlit UI
st.title('Filter DataFrame App')

# Example usage:
data = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['x', 'y', 'x', 'z'],
    'C': pd.date_range('2023-01-01', periods=4),
})

tabs = st.tabs(['Filtered Data', 'Filter Data'])

with tabs[0]:  # First tab for filtered data
    filtered_data, _ = filter_dataframe(data)
    st.write(filtered_data)  # Display filtered dataframe in the first tab

with tabs[1]:  # Second tab for filtering options
    filtered_data, filter_criteria = filter_dataframe(data)  # Retrieve filter criteria
    st.write("Filter Criteria:", filter_criteria)  # Display filter criteria in the second tab