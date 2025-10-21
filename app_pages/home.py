import streamlit as st

# ______________________________________________________________________________________________________________________
st.write("# Welcome to the Slide-Rule app!")

st.divider()
st.markdown("For more informations on the [NCSP Slide-Rule project](https://ncsp.llnl.gov/analytical-methods/criticality-slide-rule)")
st.markdown("If you have a feature request or found a bug, please [open an issue](https://gitlab.extra.irsn.fr/snc/SlideRule/-/issues), or [contact me](mailto:johann.herth@asnr.fr)")
#Contribute to the project

# ______________________________________________________________________________________________________________________
# Bouton de changement de thème (utilise la fonction et l'état définis dans le fichier principal)
def ChangeTheme():
    previous_theme = st.session_state.themes["current_theme"]
    tdict = st.session_state.themes["light"] if st.session_state.themes["current_theme"] == "light" else st.session_state.themes["dark"]
    for vkey, vval in tdict.items(): 
        if vkey.startswith("theme"): st._config.set_option(vkey, vval)
    st.session_state.themes["refreshed"] = False
    if previous_theme == "dark": st.session_state.themes["current_theme"] = "light"
    elif previous_theme == "light": st.session_state.themes["current_theme"] = "dark"

btn_face = st.session_state.themes["light"]["button_face"] if st.session_state.themes["current_theme"] == "light" else st.session_state.themes["dark"]["button_face"]

st.divider()
st.button(btn_face, on_click=ChangeTheme)
# _______________________

