import streamlit as st
import pandas as pd

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
accident_type = st.sidebar.selectbox(
    "Type of criticality accident",
    ("Solution", "Dry or low-moderated powder", "Rods/assemblies in water", "Dry solid metal")
)

boiling_status = st.sidebar.segmented_control(
    "Boiling ?", ["Yes", "No"], selection_mode="single", default = "Yes", key="boiling_status")

# ______________________________________________________________________________________________________________________
match accident_type:
    case "Solution":
        st.header("Fissile solution") 
        match boiling_status:
            case None:
                st.warning("Please select a boiling status.")
                st.stop()
            case "Yes":
                st.write("With boiling of the solution, the total number of fissions, whatever the duration of the accident, is estimated with the following formula:") 
                st.latex(r'''
                N_f = 1.3 \times 10^{16} \cdot V \cdot d_{sol} + 8 \times 10^{16} \cdot \left( V - \check{V}_{critical(geom)} \right)
                ''')
                col1, col2, col3 = st.columns(3)
                with col1:
                    V_sol = st.number_input("V (L)", value=1.0, help="Total volume of the solution (in liters).")
                with col2:
                    d_sol = st.number_input("d_*sol* (-)", value=1.0, help="Density of the solution (no unit) .")
                with col3:
                    V_crit_geo = st.number_input("V_crit_geom (L)", value=0.5, min_value=0.0, max_value=V_sol, help="Minimum critical volume of solution for the considered geometry (in liters). See https://licorne.irsn.fr/")
                
                if V_crit_geo >= V_sol:
                    # Message d'erreur si la condition n'est pas respectée
                    st.error("V_crit_geo must be less than V. Please adjust the values.")
                else:
                    # Calcul du nombre de fissions si la condition est respectée
                    NoF = 1.3e16 * V_sol * d_sol + 8e16 * (V_sol - V_crit_geo)
                    st.metric(label="Estimated number of fissions", value=f"{NoF:.1e}")
            case "No":
                st.write("Without boiling of the solution, the total number of fissions, as a function of the duration of the accident, taken into account heat loss, is estimated with the following formula:") 
                st.latex(r'''
                N_f = 1.3 \times 10^{16} \cdot V \cdot d_{sol} + 3.2 \times 10^{12} \cdot h \cdot S \cdot t
                ''')
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    V_sol = st.number_input("V (L)", value=1.0, key="V_input", help="Total volume of the solution (in liters).")
                with col2:
                    d_sol = st.number_input("d_*sol* (-)", value=1.0, help="Density of the solution (no unit).")
                with col3:
                    h = st.number_input("h (W/m²/°C)", value=10.0, help="Convection heat transfer coefficient (in W/m²/°C).")
                with col4:
                    S_placeholder = st.empty()  # Utilisé pour remplacer plus tard la valeur
                with col5:
                    t = st.number_input("t (s)", value=1.0, help="Duration of the criticality accident (in seconds).")

                with st.expander("Recommanded values for h and S (click to expand/collapse)"):
                    # Créer deux colonnes
                    col1, col2 = st.columns(2)
                    
                    # Première colonne : explications et tableau
                    with col1:
                        # Afficher le titre du tableau
                        st.write("### Convection heat transfer coefficient (h)")

                        # Ajouter une liste à puces pour les explications
                        st.markdown("""
                        - **10** : for equipment surrounded by air (the most likely case)
                        - **100** : for equipment surrounded by a cooling system 
                        """)

                        # Tableau HTML avec des lignes fusionnées
                        html_table = """
                        <table border="1" style="width:100%; border-collapse: collapse;">
                            <tr>
                                <th>Convection type</th>
                                <th>Case</th>
                                <th>h (W/m²/°C)</th>
                            </tr>
                            <tr>
                                <td rowspan="2">Free convection</td>
                                <td>Gases (air)</td>
                                <td>10</td>
                            </tr>
                            <tr>
                                <td>Liquids</td>
                                <td>50 - 1000</td>
                            </tr>
                            <tr>
                                <td rowspan="2">Forced convection</td>
                                <td>Gases</td>
                                <td>25 - 250</td>
                            </tr>
                            <tr>
                                <td>Liquids</td>
                                <td>100</td>
                            </tr>
                        </table>
                        """

                        # Afficher le tableau HTML dans Streamlit
                        st.markdown(html_table, unsafe_allow_html=True)

                    # Deuxième colonne : calculs et formules
                    with col2:
                        # Afficher la formule en LaTeX
                        st.markdown("### Heat transfer surface area (S):")
                        st.latex(r'S = k \cdot V^{2/3}')

                        # Créer trois colonnes pour les inputs k, V et la valeur calculée de S
                        col_input1, col_input2, col_input3 = st.columns(3)

                        # Entrées de k et V dans les colonnes
                        with col_input1:
                            k = st.number_input("k", value=6.00E-2, format="%.3e")
                        with col_input2:
                            V_value2 = st.number_input("V (L)", value=V_sol, key="V_input_copy", disabled=True, help="Total volume of the solution (in liters).")
                        
                        # Calcul de S
                        S_calculated = k * (V_sol ** (2/3))
                        S_placeholder.number_input("S (calculated)", value=S_calculated, disabled=True, key="S_input", help="Heat transfer surface area (in m²).")

                        # Afficher la valeur calculée de S dans la troisième colonne
                        with col_input3:
                            st.number_input("S (m²)", value=S_calculated, disabled=True, help="Heat transfer surface area (in m²).")
                        
                        # Explications supplémentaires en Markdown
                        st.markdown(""" 
                        *k* equals to:
                        - **6,000.10⁻²** for a cube
                        - **4,836.10⁻²** for a sphere
                        """)
                        # Texte final en gras pour la valeur recommandée
                        st.markdown("""
                        → The recommended default value for parameter *k* is **6.10⁻²**  
                        *(although it can be exceeded for geometries less compact than the cube or the orthocylinder)*
                        """)
                NoF = 1.3e16 * V_sol * d_sol + 3.2e12 * h * S_calculated * t
                st.metric(label="Estimated number of fissions", value=f"{NoF:.1e}")
# ______________________________________________________________________________________________________________________
match accident_type:
    case "Dry or low-moderated powder":
        st.header("Dry or low-moderated powder")
        st.info('''
                Applicable for UO₂
                ''')
        match boiling_status:
            case "Yes":
                st.write("With boiling of the water, the total number of fissions is estimated with the following formula:")
                st.latex(r''' N_f = 1.2 \times 10^{16} \cdot m_{water} + 8 \times 10^{16} \cdot \left( m_{water} - \check{m}_{water\_critical(geom)} \right) 
                        + 4 \times 10^{16} \cdot m_{powder}''')
                col1, col2, col3 = st.columns(3)
                with col2:
                    m_water = st.number_input("m_*water* (kg)", value=1.0, help="Total mass of water (in kg).")
                with col1:
                    m_powder = st.number_input("m_*powder* (kg)", value=1.0, help="Total mass of UO2 powder (in kg).")
                with col3:
                    m_water_crit_geo = st.number_input("m_*water_crit_geom* (kg)", value=0.5, min_value=0.0, max_value=m_water, help="Minimum critical mass of water for the considered geometry (in kg). See https://licorne.irsn.fr/")
                if m_water_crit_geo >= m_water:
                    # Message d'erreur si la condition n'est pas respectée
                    st.error("m_water_crit_geom must be less than m_water. Please adjust the values.")
                else:
                    # Calcul du nombre de fissions si la condition est respectée
                    NoF = 1.2e16 * m_water + 8e16 * (m_water - m_water_crit_geo) + 4e16 * m_powder
                    st.metric(label="Estimated number of fissions", value=f"{NoF:.1e}")
            case "No":
                st.write("Without boiling of water, the total number of fissions is estimated with the following formula:")
                st.latex(r'''
                N_f = 1.2 \times 10^{16} \cdot \left( m_{water} + 3.3 \times m_{powder} \right)
                ''')
                col1, col2 = st.columns(2)
                with col1:
                    m_water = st.number_input("m_*water* (kg)", value=1.0, key="m_water_no_boiling", help="Total mass of water (in kg).")
                with col2:
                    m_powder = st.number_input("m_*powder* (kg)", value=1.0, key="m_powder_no_boiling", help="Total mass of powder (in kg).")
                
                NoF = 1.2e16 * (m_water + 3.3 * m_powder)
                st.metric(label="Estimated number of fissions", value=f"{NoF:.1e}")
# ______________________________________________________________________________________________________________________

# # ______________________________________________________________________________________________________________________
match accident_type:
    case "Rods/assemblies in water":
        st.header("Rods/assemblies in water") 
        st.info('''
            The suggested formulae consider UO₂ rods (with Zircaloy cladding) in water. The following hypotheses are made, considering a UO₂ PWR “17x17” or “15x15” assembly:
            - The UO₂ mass of an assembly is about 600 kg.
            - The ratio between Zircaloy mass and UO₂ mass is about 0.17 (i.e. for each kg of UO₂, there is 0.17 kg of Zy).
            ''')
        match boiling_status:
            case "Yes":
                st.write("With boiling of the water, the total number of fissions is estimated with the following formula:")
                st.latex(r'''
                N_f = 1.2 \times 10^{16} \cdot m_{water} + 8 \times 10^{16} \cdot \left( m_{water} - \check{m}_{water\_critical(geom)} \right)
                + 4 \times 10^{16} \cdot m_{pellet} + 2.7 \times 10^{16} \cdot m_{cladding}
                ''')
                col1, col2, col3, col4 = st.columns(4)
                with col3:
                    m_water = st.number_input("m_*water* (kg)", value=1.0, help="Total mass of water (in kg).")
                with col1:
                    m_pellet = st.number_input("m_*pellet* (kg)", value=1.0, help="Total mass of UO2 pellet (in kg).")
                with col2:
                    m_cladding = st.number_input("m_*cladding* (kg)", value=1.0, help="Total mass of Zircaloy cladding (in kg).")
                with col4:
                    m_water_crit_geo = st.number_input("m_*water_crit_geom* (kg)", value=0.5, min_value=0.0, max_value=m_water, help="Minimum critical mass of water for the considered geometry (in kg). See https://licorne.irsn.fr/")
                if m_water_crit_geo >= m_water:
                    # Message d'erreur si la condition n'est pas respectée
                    st.error("m_water_crit_geom must be less than m_water. Please adjust the values.")
                else:
                    # Calcul du nombre de fissions si la condition est respectée
                    NoF = 1.2e16 * m_water + 8e16 * (m_water - m_water_crit_geo) + 4e16 * m_pellet + 2.7e16 * m_cladding
                    st.metric(label="Estimated number of fissions", value=f"{NoF:.1e}")
            case "No":
                st.write("Without boiling of water, the total number of fissions is estimated with the following formula:")
                st.latex(r'''
                N_f = 1.2 \times 10^{16} \cdot \left( m_{water} + 3.3 \times m_{pellet} + 2.2 \times m_{cladding} \right)
                ''')
                col1, col2, col3 = st.columns(3)
                with col3:
                    m_water = st.number_input("m_*water* (kg)", value=1.0, key="m_water_no_boiling", help="Total mass of water (in kg).")
                with col1:
                    m_pellet = st.number_input("m_*pellet* (kg)", value=1.0, key="m_pellet_no_boiling", help="Total mass of pellet (in kg).")
                with col2:
                    m_cladding = st.number_input("m_*cladding* (kg)", value=1.0, key="m_cladding_no_boiling", help="Total mass of cladding (in kg).")
                
                NoF = 1.2e16 * (m_water + 3.3 * m_pellet + 2.2 * m_cladding)
                st.metric(label="Estimated number of fissions", value=f"{NoF:.1e}")
# _______________________________________________________________________________________________________________________
# # ______________________________________________________________________________________________________________________
match accident_type:
    case "Dry solid metal":
        st.header("Dry solid metal")
        match boiling_status:
            case "Yes":
                st.error("No formula available for this case. Please select without boiling.")
            case "No":
                st.info('''
                        The suggested formula takes into account the kind of dry medium, considered as metal systems (plutonium, uranium, alloy of uranium and molybdenum). 
                        Melting of the system is not considered. It is considered that the entire fissile system reaches the melting temperature but without melting of the metal.
                        ''')
                st.write("The total number of fissions is estimated with the following formula:")
                st.latex(r'N_f = 6 \times 10^{15} \cdot k \cdot m_{\text{metal}}')
                col1, col2 = st.columns(2)
                with col1:
                    k_choices = {
                        "U-Mo systems (k=1)": 1.0,
                        "U systems (k=0.77)": 0.77,
                        "Pu systems (k=0.5)": 0.5
                    }
                    selected_k = st.selectbox(
                        "Parameter depending of the kind of metal:",
                        options=list(k_choices.keys())
                    )
                    k = k_choices[selected_k]
                    #st.write(f"Selected k value: {k}")
                with col2:
                    m_metal = st.number_input("m_*metal* (kg)", value=1.0, help="Total mass of metal (in kg)")
                
                NoF = 6e15 * k * m_metal
                st.metric(label="Estimated number of fissions", value=f"{NoF:.1e}")

# # ______________________________________________________________________________________________________________________
