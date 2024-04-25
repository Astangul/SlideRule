import pandas as pd
import streamlit as st
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype

def df_multiselect_filters(df: pd.DataFrame, default_columns: list = None) -> (pd.DataFrame, dict):
    """
    Adds a UI on top of a dataframe to let viewers filter columns with an option to specify default columns to display.

    Args:
        df (pd.DataFrame): Original dataframe
        default_columns (list, optional): List of column names to display by default. Defaults to None, which means all columns.

    Returns:
        pd.DataFrame: Filtered dataframe
        dict: Dictionary of applied filters
    """
    df = df.copy()
    filters = {}  # Variable pour stocker les filtres sélectionnés
    modification_container = st.container()

    with modification_container:
        # Determine the default columns to display
        if default_columns is None:
            default_columns = list(df.columns)
        
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, default=default_columns)
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                filters[column] = user_cat_input  # Stocke la valeur du filtre pour cette colonne
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                filters[column] = user_num_input  # Stocke la valeur du filtre pour cette colonne
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    filters[column] = user_date_input  # Stocke la valeur du filtre pour cette colonne
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    filters[column] = user_text_input  # Stocke la valeur du filtre pour cette colonne
                    df = df[df[column].str.contains(user_text_input)]

    return df, filters  # Retourne à la fois le DataFrame filtré et le dictionnaire de filtres
