import pandas as pd
import streamlit as st
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype


def sort_numeric_columns(df):
    """
    Trie les colonnes numériques d'un dataframe par ordre croissant.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].apply(lambda x: x.sort_values().values)
    return df

def df_multiselect_filters(df: pd.DataFrame, default_columns: list = None, key: str = "default") -> (pd.DataFrame, dict):
    """
    Adds a UI on top of a dataframe to let viewers filter columns with an option to specify default columns to display
    and a unique key for widget differentiation. By default, selects only the first unique value for each column for categorical data.

    Args:
        df (pd.DataFrame): Original dataframe
        default_columns (list, optional): List of column names to display by default.
        key (str, optional): Base key for generating unique widget keys.

    Returns:
        pd.DataFrame: Filtered dataframe
        dict: Dictionary of applied filters
    """
    df = df.copy()
    filters = {}
    modification_container = st.container()

    with modification_container:
        if default_columns is None:
            default_columns = list(df.columns)

        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, default=default_columns, key=f"filter_columns_{key}")
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            unique_key = f"{column}_{key}"  # Generate a unique key based on the column and provided key

            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                # Select the first unique value by default for categorical columns
                default_value = [df[column].dropna().unique()[0]] if df[column].dropna().unique().size > 0 else []
                user_cat_input = right.multiselect(
                    label=f"Values for {column}",
                    options=df[column].unique().tolist(),
                    default=default_value,
                    key=f"cat_{unique_key}"  # Unique key for each multiselect
                )
                filters[column] = user_cat_input
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min, _max = float(df[column].min()), float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min, _max, (_min, _max),
                    step=step,
                    key=f"num_{unique_key}"  # Unique key for each slider numeric
                )
                filters[column] = user_num_input  # Store the filter value for this column
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(df[column].min(), df[column].max()),
                    key=f"date_{unique_key}"  # Unique key for each date input
                )
                if len(user_date_input) == 2:
                    start_date, end_date = map(pd.to_datetime, user_date_input)
                    filters[column] = (start_date, end_date)
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    key=f"text_{unique_key}"  # Unique key for each text input
                )
                if user_text_input:
                    filters[column] = user_text_input  # Store the filter value for this column
                    df = df[df[column].str.contains(user_text_input)]

    return sort_numeric_columns(df), filters


def df_selectbox_filters(df: pd.DataFrame, default_columns: list = None, key: str = "default") -> (pd.DataFrame, dict):
    """
    Adds a UI on top of a dataframe to let viewers filter columns with an option to specify default columns to display
    and a unique key for widget differentiation.

    Args:
        df (pd.DataFrame): Original dataframe
        default_columns (list, optional): List of column names to display by default.
        key (str, optional): Base key for generating unique widget keys.

    Returns:
        pd.DataFrame: Filtered dataframe
        dict: Dictionary of applied filters
    """
    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    
    filters = {}  # Variable pour stocker les filtres sélectionnés

    modification_container = st.container()

    with modification_container:
        if default_columns is None:
            default_columns = list(df.columns)

        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, default=default_columns, key=f"filter_columns_{key}")
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            unique_key = f"{column}_{key}"  # Générez une clé unique basée sur la colonne et la clé fournie

            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.selectbox(
                    label=f"Values for {column}",
                    options=df[column].unique().tolist(),
                    index=0,
                    key=f"cat_{unique_key}"  # Clé unique pour selectbox
                )
                filters[column] = [user_cat_input]  # Stocke la valeur du filtre pour cette colonne dans une liste
                df = df[df[column].isin([user_cat_input])]
            elif is_numeric_dtype(df[column]):
                _min, _max = float(df[column].min()), float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min, _max, (_min, _max),
                    step=step,
                    key=f"num_{unique_key}"  # Clé unique pour slider
                )
                filters[column] = user_num_input  # Stocke la valeur du filtre pour cette colonne
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(df[column].min(), df[column].max()),
                    key=f"date_{unique_key}"  # Clé unique pour date_input
                )
                if len(user_date_input) == 2:
                    start_date, end_date = map(pd.to_datetime, user_date_input)
                    filters[column] = (start_date, end_date)
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    key=f"text_{unique_key}"  # Clé unique pour text_input
                )
                if user_text_input:
                    filters[column] = user_text_input  # Stocke la valeur du filtre pour cette colonne
                    df = df[df[column].str.contains(user_text_input)]

    return sort_numeric_columns(df), filters  # Retourne à la fois le DataFrame filtré et le dictionnaire de filtres