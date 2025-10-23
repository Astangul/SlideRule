import streamlit as st
import streamlit_authenticator as stauth
# from openpyxl import load_workbook
import yaml
from yaml.loader import SafeLoader
from utils.analytics import inject_google_analytics

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

# ______________________________________________________________________________________________________________________
# Google Analytics - Injection du code de tracking
try:
    ga_id = st.secrets["analytics"]["google_analytics_id"]
    inject_google_analytics(ga_id)
except Exception:
    # Pas d'analytics configuré (normal en développement local)
    pass

# ______________________________________________________________________________________________________________________
# Initialisation du thème (global pour toute l'application)
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

if ms.themes["refreshed"] == False:
  ms.themes["refreshed"] = True
  st.rerun()

# ______________________________________________________________________________________________________________________
# Configuration de la navigation
pages = {
    "": [
        st.Page("app_pages/home.py", title="Welcome", icon="👋", default=True),
    ],
    "Slide-Rule Application": [
        st.Page("app_pages/02_1️⃣_Number_of_fissions.py", title="Number of fissions", icon="1️⃣"),
        st.Page("app_pages/03_2️⃣_Dose.py", title="Dose", icon="2️⃣"),
    ],
    "Resources": [
        st.Page("app_pages/04_📄_Documentation.py", title="Documentation", icon="📄"),
    ],
    "Advanced use": [
        st.Page("app_pages/01_0️⃣_Raw_results.py", title="Raw results", icon="0️⃣"),
    ],
}

pg = st.navigation(pages, expanded=True)
pg.run()
# ______________________________________________________________________________________________________________________
