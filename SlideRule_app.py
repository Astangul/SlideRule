import streamlit as st
import streamlit_authenticator as stauth
# from openpyxl import load_workbook
import yaml
from yaml.loader import SafeLoader

# import pandas as pd

# from pandas.api.types import (
#     is_categorical_dtype,
#     is_datetime64_any_dtype,
#     is_numeric_dtype,
#     is_object_dtype,
# )

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


st.Page(page="./pages/.py",  icon=":material/home:")

st.write("# Welcome to the Slide-Rule app! 👋")

with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

authenticator.login()

if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some content')
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


# ________________
ms = st.session_state
if "themes" not in ms: 
    ms.themes = {"current_theme": "light",
                    "refreshed": True,
                    
                    "light": {"theme.base": "dark", 
                              "theme.backgroundColor": "#0E1117", 
                              "theme.primaryColor": "#ff4b4b", #"#c98bdb"
                              "theme.secondaryBackgroundColor": "#262730", #"#5591f5",
                              "theme.textColor": "white",
                              "button_face": "🌜"},

                    "dark":  {"theme.base": "light",
                              "theme.backgroundColor": "white",
                              "theme.primaryColor": "#6c1d82", #"#5591f5",
                              "theme.secondaryBackgroundColor": "#F0F2F6", #"#82E1D7",
                              "theme.textColor": "#0a1464",
                              "button_face": "🌞"},
                    }
  

def ChangeTheme():
    previous_theme = ms.themes["current_theme"]
    tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
    for vkey, vval in tdict.items(): 
        if vkey.startswith("theme"): st._config.set_option(vkey, vval)
    ms.themes["refreshed"] = False
    if previous_theme == "dark": ms.themes["current_theme"] = "light"
    elif previous_theme == "light": ms.themes["current_theme"] = "dark"

btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]

if ms.themes["refreshed"] == False:
  ms.themes["refreshed"] = True
  st.rerun()

st.divider()
st.button(btn_face, on_click=ChangeTheme)
# _______________________
