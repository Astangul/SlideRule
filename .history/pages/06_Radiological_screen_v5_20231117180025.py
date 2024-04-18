import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype, is_numeric_dtype, is_object_dtype

def filter_categorical(column, data):
    user_cat_input = st.multiselect(
        f"Values for {column}",
        data.unique(),
        default=list(data.unique()),
        key=f"cat_input_{column}"
    )
    return user_cat_input

# Autres fonctions de filtre (filter_numeric, filter_date, filter_text) restent inchangées...

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

    to_filter_columns = st.multiselect("Filter dataframe on", df.columns, key='filter_columns')

    for column in to_filter_columns:
        left, right = st.columns((1, 20))
        left.write("↳")

        if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
            user_cat_input = filter_categorical(column, df[column])
            filter_criteria[column] = user_cat_input
            df = df[df[column].isin(user_cat_input)]
        # Autres conditions de filtrage restent inchangées...

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
