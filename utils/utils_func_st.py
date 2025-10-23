# import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype
import itertools

# __________________________________
def normalize_filters(filters):
    """
    Normalize filters by converting all single-element lists to their single value.
    """
    normalized = {}
    for key, value in filters.items():
        if isinstance(value, list) and len(value) == 1:
            normalized[key] = value[0]  # Convert single-element lists to the single value
        else:
            normalized[key] = value
    return normalized

def generate_filter_combinations(filters):
    """
    Generate all possible filter combinations.
    Args:
        filters (dict): The filters to decompose.
    Returns:
        List[dict]: A list of dictionaries, each representing a unique filter combination.
    """
    keys, values = zip(*filters.items())
    value_combinations = itertools.product(*[v if isinstance(v, list) else [v] for v in values])
    return [dict(zip(keys, combination)) for combination in value_combinations]


def load_data_with_filters(data, filters):
    """
    Load data based on the given filters.
    """
    query = " and ".join([f"`{key}` == {repr(value)}" for key, value in filters.items()])
    return data.query(query)

def generate_series_name(series_number, compare_filter_single, ref_filters):
    differences = []
    for key in compare_filter_single.keys():
        compare_value = compare_filter_single[key]
        ref_value = ref_filters[key][0]
        if compare_value != ref_value:
            differences.append(f"'{key}': [{compare_value} vs. {ref_value}]")
    differences_str = ', '.join(differences)
    return f"Serie {series_number} - {differences_str}" if differences else f"Serie {series_number}"
# __________________________________
# def df_multiselect_filters(
#     df: pd.DataFrame,
#     default_columns: list = None,
#     key: str = "default",
#     default_values: dict = None  # Dictionnaire pour les valeurs par défaut
# ) -> (pd.DataFrame, dict):
#     """
#     Adds a UI on top of a dataframe to let viewers filter columns with an option to specify default columns to display,
#     default values for specific columns, and a unique key for widget differentiation.
#     By default, selects only the first unique value for each column for categorical data.

#     Args:
#         df (pd.DataFrame): Original dataframe
#         default_columns (list, optional): List of column names to display by default.
#         key (str, optional): Base key for generating unique widget keys.
#         default_values (dict, optional): Dictionary of default filter values for specific columns.

#     Returns:
#         pd.DataFrame: Filtered dataframe
#         dict: Dictionary of applied filters
#     """
#     df = df.copy()
#     filters = {}
#     default_values = default_values or {}  # Initialise un dictionnaire vide si non fourni

#     with st.container():
#         if default_columns is None:
#             default_columns = list(df.columns)

#         to_filter_columns = st.multiselect(
#             "Filter dataframe on",
#             df.columns,
#             default=default_columns,
#             key=f"filter_columns_{key}"
#         )
        
#         for column in to_filter_columns:
#             left, right = st.columns((1, 20))
#             left.write("↳")
#             unique_key = f"{column}_{key}"  # Clé unique pour chaque widget
#             column_defaults = default_values.get(column, None)  # Récupère les valeurs par défaut pour cette colonne

#             if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
#                 options = df[column].unique().tolist()
#                 # Filtrer les valeurs par défaut pour ne conserver que celles valides
#                 valid_defaults = (
#                     [val for val in column_defaults if val in options]
#                     if column_defaults else [options[0]] if options else []
#                 )
#                 user_cat_input = right.multiselect(
#                     label=f"Values for {column}",
#                     options=options,
#                     default=valid_defaults,
#                     key=f"cat_{unique_key}"
#                 )
#                 filters[column] = user_cat_input
#                 df = df[df[column].isin(user_cat_input)]
#             elif is_numeric_dtype(df[column]):
#                 _min, _max = float(df[column].min()), float(df[column].max())
#                 step = (_max - _min) / 100
#                 # Vérifier si la plage par défaut est valide
#                 default_range = (
#                     column_defaults if column_defaults and
#                     column_defaults[0] >= _min and column_defaults[1] <= _max
#                     else (_min, _max)
#                 )
#                 user_num_input = right.slider(
#                     f"Values for {column}",
#                     _min, _max,
#                     value=default_range,
#                     step=step,
#                     key=f"num_{unique_key}"
#                 )
#                 filters[column] = user_num_input
#                 df = df[df[column].between(*user_num_input)]
#             elif is_datetime64_any_dtype(df[column]):
#                 default_dates = (
#                     column_defaults if column_defaults and
#                     df[column].min() <= column_defaults[0] <= column_defaults[1] <= df[column].max()
#                     else (df[column].min(), df[column].max())
#                 )
#                 user_date_input = right.date_input(
#                     f"Values for {column}",
#                     value=default_dates,
#                     key=f"date_{unique_key}"
#                 )
#                 if len(user_date_input) == 2:
#                     start_date, end_date = map(pd.to_datetime, user_date_input)
#                     filters[column] = (start_date, end_date)
#                     df = df.loc[df[column].between(start_date, end_date)]
#             else:
#                 user_text_input = right.text_input(
#                     f"Substring or regex in {column}",
#                     value=column_defaults if column_defaults else "",
#                     key=f"text_{unique_key}"
#                 )
#                 if user_text_input:
#                     filters[column] = user_text_input
#                     df = df[df[column].str.contains(user_text_input)]

#     return df, filters

def df_multiselect_filters(
    df: pd.DataFrame,
    default_columns: list = None,
    key: str = "default",
    default_values: dict = None  # Dictionnaire pour les valeurs par défaut, avec les mots-clés "__all__" et "__first__"
) -> (pd.DataFrame, dict):
    """
    Adds a UI on top of a dataframe to let viewers filter columns with an option to specify default columns to display,
    default values for specific columns, and a unique key for widget differentiation.
    Supports special keywords in default_values:
    - "__all__": select all unique values for a column
    - "__first__": select only the first unique value for a column

    Args:
        df (pd.DataFrame): Original dataframe
        default_columns (list, optional): List of column names to display by default.
        key (str, optional): Base key for generating unique widget keys.
        default_values (dict, optional): Dictionary of default filter values for specific columns, with "__all__" and "__first__" support.

    Returns:
        pd.DataFrame: Filtered dataframe
        dict: Dictionary of applied filters
    """
    df = df.copy()
    filters = {}
    default_values = default_values or {}  # Initialise un dictionnaire vide si non fourni

    with st.container():
        if default_columns is None:
            default_columns = list(df.columns)

        to_filter_columns = st.multiselect(
            "Filter dataframe on",
            df.columns,
            default=default_columns,
            key=f"filter_columns_{key}"
        )
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            unique_key = f"{column}_{key}"  # Clé unique pour chaque widget
            column_defaults = default_values.get(column, None)  # Récupère les valeurs par défaut pour cette colonne

            if is_categorical_dtype(df[column]) or df[column].nunique() < 50:
                options = df[column].unique().tolist()
                # Si "__all__" est spécifié, utiliser toutes les valeurs uniques
                # Si "__first__" est spécifié, utiliser seulement la première valeur
                valid_defaults = (
                    options if column_defaults == "__all__" else
                    [options[0]] if column_defaults == "__first__" and options else
                    [val for val in column_defaults if val in options] if column_defaults else
                    [options[0]] if options else []
                )
                user_cat_input = right.multiselect(
                    label=f"Values for {column}",
                    options=options,
                    default=valid_defaults,
                    key=f"cat_{unique_key}"
                )
                filters[column] = user_cat_input
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min, _max = float(df[column].min()), float(df[column].max())
                step = (_max - _min) / 100
                # Si "__all__" est spécifié, utiliser toute la plage comme valeur par défaut
                default_range = (
                    (_min, _max) if column_defaults == "__all__" else
                    column_defaults if column_defaults and
                    column_defaults[0] >= _min and column_defaults[1] <= _max else
                    (_min, _max)
                )
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min, _max,
                    value=default_range,
                    step=step,
                    key=f"num_{unique_key}"
                )
                filters[column] = user_num_input
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                # Si "__all__" est spécifié, utiliser toute la plage de dates comme valeur par défaut
                default_dates = (
                    (df[column].min(), df[column].max()) if column_defaults == "__all__" else
                    column_defaults if column_defaults and
                    df[column].min() <= column_defaults[0] <= column_defaults[1] <= df[column].max() else
                    (df[column].min(), df[column].max())
                )
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=default_dates,
                    key=f"date_{unique_key}"
                )
                if len(user_date_input) == 2:
                    start_date, end_date = map(pd.to_datetime, user_date_input)
                    filters[column] = (start_date, end_date)
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    value=column_defaults if column_defaults else "",
                    key=f"text_{unique_key}"
                )
                if user_text_input:
                    filters[column] = user_text_input
                    df = df[df[column].str.contains(user_text_input)]

    return df, filters


# def df_selectbox_filters(df: pd.DataFrame, default_columns: list = None, key: str = "default") -> (pd.DataFrame, dict):
#     """
#     Adds a UI on top of a dataframe to let viewers filter columns with an option to specify default columns to display
#     and a unique key for widget differentiation.

#     Args:
#         df (pd.DataFrame): Original dataframe
#         default_columns (list, optional): List of column names to display by default.
#         key (str, optional): Base key for generating unique widget keys.

#     Returns:
#         pd.DataFrame: Filtered dataframe
#         dict: Dictionary of applied filters
#     """
#     df = df.copy()

#     # Try to convert datetimes into a standard format (datetime, no timezone)
#     for col in df.columns:
#         if is_object_dtype(df[col]):
#             try:
#                 df[col] = pd.to_datetime(df[col])
#             except Exception:
#                 pass

#         if is_datetime64_any_dtype(df[col]):
#             df[col] = df[col].dt.tz_localize(None)
    
#     filters = {}  # Variable pour stocker les filtres sélectionnés

#     modification_container = st.container()

#     with modification_container:
#         if default_columns is None:
#             default_columns = list(df.columns)

#         to_filter_columns = st.multiselect("Filter dataframe on", df.columns, default=default_columns, key=f"filter_columns_{key}")
        
#         for column in to_filter_columns:
#             left, right = st.columns((1, 20))
#             left.write("↳")
#             unique_key = f"{column}_{key}"  # Générez une clé unique basée sur la colonne et la clé fournie

#             if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
#                 user_cat_input = right.selectbox(
#                     label=f"Values for {column}",
#                     options=df[column].unique().tolist(),
#                     index=0,
#                     key=f"cat_{unique_key}"  # Clé unique pour selectbox
#                 )
#                 filters[column] = [user_cat_input]  # Stocke la valeur du filtre pour cette colonne dans une liste
#                 df = df[df[column].isin([user_cat_input])]
#             elif is_numeric_dtype(df[column]):
#                 _min, _max = float(df[column].min()), float(df[column].max())
#                 step = (_max - _min) / 100
#                 user_num_input = right.slider(
#                     f"Values for {column}",
#                     _min, _max, (_min, _max),
#                     step=step,
#                     key=f"num_{unique_key}"  # Clé unique pour slider
#                 )
#                 filters[column] = user_num_input  # Stocke la valeur du filtre pour cette colonne
#                 df = df[df[column].between(*user_num_input)]
#             elif is_datetime64_any_dtype(df[column]):
#                 user_date_input = right.date_input(
#                     f"Values for {column}",
#                     value=(df[column].min(), df[column].max()),
#                     key=f"date_{unique_key}"  # Clé unique pour date_input
#                 )
#                 if len(user_date_input) == 2:
#                     start_date, end_date = map(pd.to_datetime, user_date_input)
#                     filters[column] = (start_date, end_date)
#                     df = df.loc[df[column].between(start_date, end_date)]
#             else:
#                 user_text_input = right.text_input(
#                     f"Substring or regex in {column}",
#                     key=f"text_{unique_key}"  # Clé unique pour text_input
#                 )
#                 if user_text_input:
#                     filters[column] = user_text_input  # Stocke la valeur du filtre pour cette colonne
#                     df = df[df[column].str.contains(user_text_input)]

#     return df, filters  # Retourne à la fois le DataFrame filtré et le dictionnaire de filtres

def df_selectbox_filters(
    df: pd.DataFrame,
    default_columns: list = None,
    key: str = "default",
    default_values: dict = None  # Dictionnaire pour les valeurs par défaut, avec support de "__first__"
) -> (pd.DataFrame, dict):
    """
    Adds a UI on top of a dataframe to let viewers filter columns with an option to specify default columns to display,
    default values for specific columns, and a unique key for widget differentiation.
    Supports special keyword "__first__" in default_values to select the first unique value for a column.

    Args:
        df (pd.DataFrame): Original dataframe
        default_columns (list, optional): List of column names to display by default.
        key (str, optional): Base key for generating unique widget keys.
        default_values (dict, optional): Dictionary of default filter values for specific columns, with "__first__" support.

    Returns:
        pd.DataFrame: Filtered dataframe
        dict: Dictionary of applied filters
    """
    df = df.copy()

    # Convertir les colonnes datetime si nécessaire
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    filters = {}
    default_values = default_values or {}  # Initialise un dictionnaire vide si non fourni

    modification_container = st.container()

    with modification_container:
        if default_columns is None:
            default_columns = list(df.columns)

        to_filter_columns = st.multiselect(
            "Filter dataframe on",
            df.columns,
            default=default_columns,
            key=f"filter_columns_{key}"
        )
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            unique_key = f"{column}_{key}"  # Clé unique pour chaque widget
            column_defaults = default_values.get(column, None)  # Récupère les valeurs par défaut pour cette colonne

            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                options = df[column].unique().tolist()
                # Si "__first__" est spécifié, utiliser la première valeur
                default_value = (
                    options[0] if column_defaults == "__first__" and options else
                    column_defaults if column_defaults in options else 
                    options[0] if options else None
                )  # Vérifie si la valeur par défaut est valide
                if default_value is not None:
                    user_cat_input = right.selectbox(
                        label=f"Values for {column}",
                        options=options,
                        index=options.index(default_value),
                        key=f"cat_{unique_key}"
                    )
                    filters[column] = [user_cat_input]
                    df = df[df[column].isin([user_cat_input])]
            elif is_numeric_dtype(df[column]):
                _min, _max = float(df[column].min()), float(df[column].max())
                step = (_max - _min) / 100
                default_range = (
                    column_defaults if column_defaults and
                    column_defaults[0] >= _min and column_defaults[1] <= _max else (_min, _max)
                )  # Vérifie si la plage est valide
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min, _max,
                    value=default_range,
                    step=step,
                    key=f"num_{unique_key}"
                )
                filters[column] = user_num_input
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                default_dates = (
                    column_defaults if column_defaults and
                    df[column].min() <= column_defaults[0] <= column_defaults[1] <= df[column].max()
                    else (df[column].min(), df[column].max())
                )  # Vérifie si les dates par défaut sont valides
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=default_dates,
                    key=f"date_{unique_key}"
                )
                if len(user_date_input) == 2:
                    start_date, end_date = map(pd.to_datetime, user_date_input)
                    filters[column] = (start_date, end_date)
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    value=column_defaults if column_defaults else "",
                    key=f"text_{unique_key}"
                )
                if user_text_input:
                    filters[column] = user_text_input
                    df = df[df[column].str.contains(user_text_input)]

    return df, filters

