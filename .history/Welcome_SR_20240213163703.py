import streamlit as st
import streamlit_authenticator as stauth
# from openpyxl import load_workbook
import yaml
from yaml.loader import SafeLoader
import pandas as pd

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
st.write("# Welcome to the Slide-Rule app! 👋")
st.title("Slide Rule")
st.write(
    """My first Streamlit app
    """
)

