import streamlit as st
import streamlit_authenticator as stauth
# from openpyxl import load_workbook
import yaml
from yaml.loader import SafeLoader

# ______________________________________________________________________________________________________________________
# Configuration de la page Streamlit
st.set_page_config(
    page_title="Slide-Rule",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': "https://gitlab.extra.irsn.fr/snc/SlideRule/-/issues",
        'About': "https://ncsp.llnl.gov/analytical-methods/criticality-slide-rule"
    }
)
sidebar_logo_path = "./icons/Slide-Rule_orange.png"
main_body_logo_path = "./icons/Slide-Rule_DallE-1.png"
st.logo(image = sidebar_logo_path, size="large", icon_image = sidebar_logo_path)

# Ajouter l'image dans la sidebar
# image_path = "./icons/Slide-Rule_orange.png"
# st.sidebar.image(image_path, use_column_width=True)
# ______________________________________________________________________________________________________________________

st.write("# Welcome to the Slide-Rule app!")

st.divider()
st.markdown("For more informations on the [NCSP Slide-Rule project](https://ncsp.llnl.gov/analytical-methods/criticality-slide-rule)")
st.markdown("If you have a feature request or found a bug, please [open an issue](https://gitlab.extra.irsn.fr/snc/SlideRule/-/issues), or [contact me](mailto:johann.herth@asnr.fr)")
#Contribute to the project

# ______________________________________________________________________________________________________________________
# Configuration de l'authentification
# with open('./config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )

# authenticator.login()

# if st.session_state["authentication_status"]:
#     authenticator.logout()
#     st.write(f'Welcome *{st.session_state["name"]}*')
#     st.title('Some content')
# elif st.session_state["authentication_status"] is False:
#     st.error('Username/password is incorrect')
# elif st.session_state["authentication_status"] is None:
#     st.warning('Please enter your username and password')


# ________________

def ChangeTheme():
    previous_theme = ms.themes["current_theme"]
    tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
    for vkey, vval in tdict.items(): 
        if vkey.startswith("theme"): st._config.set_option(vkey, vval)
    ms.themes["refreshed"] = False
    if previous_theme == "dark": ms.themes["current_theme"] = "light"
    elif previous_theme == "light": ms.themes["current_theme"] = "dark"

ms = st.session_state
if "themes" not in ms: 
    ms.themes = {
        "current_theme": "light",  
        "refreshed": True,
        "light": {  # Thème clair
            "theme.base": "light", 
            "theme.backgroundColor": "#FFFFFF", #F9F7F7
            "theme.primaryColor": "#6c1d82", 
            "theme.secondaryBackgroundColor": "#F0F2F6", 
            "theme.textColor": "#000000", # "#0a1464",
            "button_face": "🌞"
        },
        "dark": {  # Thème sombre
            "theme.base": "dark",
            "theme.backgroundColor": "#0E1117", 
            "theme.primaryColor": "#ff4b4b", 
            "theme.secondaryBackgroundColor": "#262730", 
            "theme.textColor": "#FFFFFF",
            "button_face": "🌜"
        },
    }
    # # Simuler le clic en appelant directement la fonction de callback pour appliquer le thème
    ChangeTheme()
    ms.theme_initialized = True  # pour éviter de répéter l'initialisation

btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]

if ms.themes["refreshed"] == False:
  ms.themes["refreshed"] = True
  st.rerun()

st.divider()
st.button(btn_face, on_click=ChangeTheme)
# _______________________
