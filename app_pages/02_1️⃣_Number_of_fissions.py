import streamlit as st
import pandas as pd

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
                with st.expander("📖 Derivation of the formula (click to expand/collapse)"):
                    st.markdown(r'''
                    **Principle** — bounding estimate based on the *heat energy equation* (1st law of thermodynamics), with the
                    conversion factor $\varepsilon = 3.51\times10^{10}$ fissions/J (1 fission $\approx$ 180 MeV of *thermal* energy;
                    the remaining $\approx$ 20 MeV — antineutrinos, escaping delayed radiation — does not heat the medium).
                    Assumptions: adiabatic conditions, constant pressure, no recondensation of the vapor, decay heat not deducted (conservative).

                    **Step 1 — Heating the solution to boiling.** With $m_{sol} = d_{sol}\cdot V$ ($d_{sol}$ in kg/L $\Leftrightarrow$ dimensionless
                    density, $V$ in liters), $T_0 = 20$ °C, $T_{boiling} = 110$ °C (nitrate solution) and $C_p = 4184$ J·kg⁻¹·°C⁻¹
                    (bounding value: $C_p$ of uranyl nitrate solutions *decreases* with the U and HNO₃ concentrations):
                    ''')
                    st.latex(r'''Nf_1 = \varepsilon\, m_{sol}\, C_p \left[T_{boiling}-T_0\right]
                             = 3.51\times10^{10}\times 4184\times 90 \cdot V d_{sol}
                             = 1.32\times10^{16}\, V d_{sol} \;\rightarrow\; 1.3\times10^{16}\, V d_{sol}''')
                    st.markdown(r'''
                    **Step 2 — Maintaining boiling.** Evaporating 1 kg of water requires
                    $\varepsilon \cdot \Delta H_{vap} = 3.51\times10^{10} \times 2.26\times10^{6} = 7.9\times10^{16}
                    \rightarrow 8\times10^{16}$ fissions/kg (rounded up, conservative).

                    **Step 3 — Evaporated mass.** CRAC/SILENE boiling experiments showed that the fissile mass stays in solution:
                    only **water** evaporates, so $\Delta m_{water} = (V - V_{final})\cdot d_{H_2O}$ with $d_{H_2O} \simeq 1$ kg/L.
                    In the worst case the accident goes on until the volume reaches the **minimum critical volume for the considered
                    geometry** $\check{V}_{critical(geom)}$ (below it, the system is subcritical whatever the concentration), hence:
                    ''')
                    st.latex(r'''Nf_2 = 8\times10^{16}\left(V - \check{V}_{critical(geom)}\right)''')
                    st.markdown(r'''
                    **Total** $N_f = Nf_1 + Nf_2$. Since the final state is an intrinsic physical limit, the formula is valid
                    **whatever the duration** of the accident.

                    *References: M. Duluc, NCSD 2009; M. Duluc et al., NCSD 2022 (Eq. 1).*
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
                with st.expander("📖 Derivation of the formula (click to expand/collapse)"):
                    st.markdown(r'''
                    **Principle** — heat energy equation with $\varepsilon = 3.51\times10^{10}$ fissions/J
                    (1 fission $\approx$ 180 MeV of thermal energy). Without boiling, the sensible-heat term alone is not bounding
                    for a long accident: the fission power can be evacuated by convection while the system stays critical, so a
                    **heat loss term** is added.

                    **Step 1 — Heating the solution to boiling** (same as the boiling case), with $C_p = 4184$ J·kg⁻¹·°C⁻¹
                    (bounding for nitrate solutions), $T_{boiling}-T_0 = 110-20 = 90$ °C:
                    ''')
                    st.latex(r'''Nf_1 = \varepsilon\, m_{sol}\, C_p \left[T_{boiling}-T_0\right]
                             = 1.32\times10^{16}\, V d_{sol} \;\rightarrow\; 1.3\times10^{16}\, V d_{sol}''')
                    st.markdown(r'''
                    **Step 2 — Convection heat loss.** The general balance adds
                    $\varepsilon \int_0^{t} h(t')\,S(t')\,(T(t')-T_{ext})\,dt'$. Each parameter is taken constant and penalizing:
                    $T(t') = T_{boiling} = 110$ °C during the whole accident and $T_{ext} = T_0 = 20$ °C (maximizes the losses,
                    hence the fissions compensating them):
                    ''')
                    st.latex(r'''N_{f,loss} = \varepsilon\, h S \left[T_{boiling}-T_{0}\right] t
                             = 3.51\times10^{10}\times 90 \cdot hSt = 3.16\times10^{12}\, hSt
                             \;\rightarrow\; 3.2\times10^{12}\, hSt''')
                    st.markdown(r'''
                    **Step 3 — Heat transfer surface area** $S = k\cdot V^{2/3}$. In SI units:
                    sphere $V=\tfrac{4}{3}\pi r^3,\; S = 4\pi r^2 \Rightarrow S = (36\pi)^{1/3}V^{2/3} = 4.836\,V^{2/3}$;
                    cube $V=a^3,\; S = 6a^2 = 6\,V^{2/3}$. Converting $V$ to liters
                    ($V_{m^3}^{2/3} = 10^{-2}\,V_{L}^{2/3}$) gives $k = 4.836\times10^{-2}$ (sphere) and
                    $k = 6\times10^{-2}$ (cube, recommended — bounds the sphere and the orthocylinder, but can be exceeded
                    by less compact geometries).

                    **Total** $N_f = Nf_1 + N_{f,loss}$, with $h$ in W·m⁻²·°C⁻¹, $S$ in m², $t$ in s.

                    *References: M. Duluc, NCSD 2009 (Eqs. 15–17); M. Duluc et al., NCSD 2022 (Eq. 2).*
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
                with st.expander("📖 Derivation of the formula (click to expand/collapse)"):
                    st.markdown(r'''
                    **Principle** — heat energy equation, $\varepsilon = 3.51\times10^{10}$ fissions/J. The system is
                    **heterogeneous**: fissions occur *in the powder*, heat is then transferred to the water. Three contributions:

                    **Step 1 — Heating the water to boiling** ($T_{boiling} = 100$ °C for water, $T_0 = 20$ °C):
                    ''')
                    st.latex(r'''\varepsilon\, C_{p(water)}\left[100-20\right]
                             = 3.51\times10^{10}\times4184\times80 = 1.17\times10^{16}
                             \;\rightarrow\; 1.2\times10^{16}\ \text{fissions/kg}''')
                    st.markdown(r'''
                    **Step 2 — Evaporation.** As for solutions: $\varepsilon\,\Delta H_{vap} = 7.9\times10^{16} \rightarrow 8\times10^{16}$
                    fissions per kg of evaporated water. In the worst case the accident goes on until the water mass reaches the
                    **minimum critical mass of water for the geometry** $\check{m}_{water\_critical(geom)}$ (computed with a penalizing
                    heterogeneous distribution of the moderation), hence the term
                    $8\times10^{16}(m_{water}-\check{m}_{water\_critical(geom)})$.

                    **Step 3 — Residual energy in the powder.** Thermal equilibrium between powder and water is generally *not*
                    achieved (finite particle-to-water heat transfer, possible vapor film): part of the fission energy stays in the
                    powder. Bounding assumption: **the whole powder mass reaches the UO₂ melting temperature** ($\approx$ 2850 °C)
                    while the water is at 100 °C. With $\bar{C}_p(UO_2) \approx 410$ J·kg⁻¹·°C⁻¹ (mean between 100 °C and melting):
                    ''')
                    st.latex(r'''Nf_3 = \varepsilon\, \bar{C}_{p}\left[2850-100\right]
                             = 3.51\times10^{10}\times410\times2750 = 3.96\times10^{16}
                             \;\rightarrow\; 4\times10^{16}\ \text{fissions/kg}''')
                    st.markdown(r'''
                    **Total** — sum of the three contributions (all roundings upward, conservative).

                    *References: M. Duluc & G. Caplin, ICNC 2011 (Eq. 10); M. Duluc et al., NCSD 2022 (Eq. 3).*
                    ''')
                col1, col2, col3 = st.columns(3)
                with col2:
                    m_water = st.number_input("m_*water* (kg)", value=1.0, help="Total mass of water (in kg).")
                with col1:
                    m_powder = st.number_input("m_*powder* (kg)", value=1.0, help="Total mass of UO2 powder (in kg).")
                with col3:
                    m_water_crit_geo = st.number_input("m_*water_crit_geom* (kg)", value=0.5, min_value=0.0, max_value=m_water, help="Minimum critical mass of water for the considered geometry (in kg). See https://licorne.asnr.fr/")
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
                with st.expander("📖 Derivation of the formula (click to expand/collapse)"):
                    st.markdown(r'''
                    **Principle** — heat energy equation, $\varepsilon = 3.51\times10^{10}$ fissions/J. Fissions occur *in the
                    powder*; even without boiling of the water, part of the fission energy remains stored in the powder
                    (thermal equilibrium is generally not achieved), so the powder can be far hotter than the water.

                    **Step 1 — Water term** ($T_0 = 20$ °C $\rightarrow$ $T_{boiling} = 100$ °C):
                    $\varepsilon\, C_{p(water)} \cdot 80 = 3.51\times10^{10}\times4184\times80 = 1.17\times10^{16}
                    \rightarrow 1.2\times10^{16}$ fissions/kg (rounded up).

                    **Step 2 — Powder term.** Bounding assumption for the energy retained in the powder: **the whole powder mass
                    reaches the UO₂ melting temperature** ($\approx$ 2850 °C). With
                    $\bar{C}_p(UO_2) \approx 410$ J·kg⁻¹·°C⁻¹ (mean between 100 °C and melting):
                    ''')
                    st.latex(r'''Nf_3 = \varepsilon\,\bar{C}_p\left[2850-100\right]
                             = 3.96\times10^{16}\ \text{fissions/kg}
                             = 3.3\times\left(1.2\times10^{16}\right)''')
                    st.markdown(r'''
                    which is the origin of the factor **3.3** (the small sensible-heat term of the powder below 100 °C,
                    $m_{powder}/17$ with $17 \approx C_{p(water)}/C_{p(UO_2, 20-100°C)} = 4184/250$, is absorbed in the rounding —
                    it would be the *thermal-equilibrium* variant $N_f = 1.2\times10^{16}(m_{water}+m_{powder}/17)$, not retained
                    because non-bounding during the transient).

                    *References: M. Duluc & G. Caplin, ICNC 2011 (Eqs. 8–9); M. Duluc et al., NCSD 2022 (Eq. 4).*
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
                with st.expander("📖 Derivation of the formula (click to expand/collapse)"):
                    st.markdown(r'''
                    **Principle** — same heterogeneous model as for powders ($\varepsilon = 3.51\times10^{10}$ fissions/J),
                    with a third medium: the Zircaloy cladding.

                    **Step 1 — Water**: $\varepsilon\,C_{p(water)}\cdot 80 = 1.17\times10^{16} \rightarrow 1.2\times10^{16}$ fissions/kg.

                    **Step 2 — Evaporation**: $\varepsilon\,\Delta H_{vap} = 7.9\times10^{16} \rightarrow 8\times10^{16}$ fissions/kg;
                    in the worst case the accident stops when the water mass reaches the **minimum critical mass of water for the
                    geometry** $\check{m}_{water\_critical(geom)}$.

                    **Step 3 — Residual energy in the rods.** Bounding assumptions: pellets reach the UO₂ melting temperature
                    ($\approx$ 2850 °C), cladding reaches the Zircaloy solidus ($\approx$ 2100 °C), while the water is at 100 °C:
                    ''')
                    st.latex(r'''\text{pellet: } \varepsilon\times410\times\left[2850-100\right] = 3.96\times10^{16}
                             \rightarrow 4\times10^{16}\ \text{fissions/kg}''')
                    st.latex(r'''\text{cladding: } \varepsilon\times370\times\left[2100-100\right] = 2.60\times10^{16}
                             \rightarrow 2.2\times1.2\times10^{16} = 2.64\times10^{16} \rightarrow 2.7\times10^{16}\ \text{fissions/kg}''')
                    st.markdown(r'''
                    with $\bar{C}_p(UO_2) \approx 410$ and $\bar{C}_p(Zy) \approx 370$ J·kg⁻¹·°C⁻¹ (mean values between 100 °C and
                    the melting/solidus temperature). All roundings upward (conservative).

                    *References: M. Duluc & G. Caplin, ICNC 2011 (Eq. 15); M. Duluc et al., NCSD 2022 (Eq. 5).*
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
                with st.expander("📖 Derivation of the formula (click to expand/collapse)"):
                    st.markdown(r'''
                    **Principle** — same heterogeneous model as for powders ($\varepsilon = 3.51\times10^{10}$ fissions/J):
                    fissions occur in the pellets; part of the energy remains stored in the rods (finite heat transfer through
                    pellet, gap and cladding), so pellets and cladding can be far hotter than the water.

                    **Step 1 — Water term**: $\varepsilon\,C_{p(water)}\cdot\left[100-20\right] = 1.17\times10^{16}
                    \rightarrow 1.2\times10^{16}$ fissions/kg (rounded up).

                    **Step 2 — Pellet term.** Bounding: pellets reach the UO₂ melting temperature ($\approx$ 2850 °C), with
                    $\bar{C}_p(UO_2) \approx 410$ J·kg⁻¹·°C⁻¹ (mean, 100 °C $\rightarrow$ melting):
                    ''')
                    st.latex(r'''\varepsilon\times410\times\left[2850-100\right] = 3.96\times10^{16}
                             = 3.3\times\left(1.2\times10^{16}\right)''')
                    st.markdown(r'''
                    **Step 3 — Cladding term.** Bounding: cladding reaches the Zircaloy solidus ($\approx$ 2100 °C), with
                    $\bar{C}_p(Zy) \approx 370$ J·kg⁻¹·°C⁻¹ (mean, 100 °C $\rightarrow$ solidus):
                    ''')
                    st.latex(r'''\varepsilon\times370\times\left[2100-100\right] = 2.60\times10^{16}
                             = 2.16\times\left(1.2\times10^{16}\right) \;\rightarrow\; 2.2 \ \text{(rounded up)}''')
                    st.markdown(r'''
                    (The sensible-heat terms of pellet and cladding below 100 °C — $m/17$ and $m/15$, thermal-equilibrium variant —
                    are absorbed in the roundings.)

                    *References: M. Duluc & G. Caplin, ICNC 2011 (Eqs. 13–14); M. Duluc et al., NCSD 2022 (Eq. 6).*
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
                with st.expander("📖 Derivation of the formula (click to expand/collapse)"):
                    st.markdown(r'''
                    **Principle** — "thermal formula": heat energy equation with $\varepsilon = 3.51\times10^{10}$ fissions/J and,
                    as bounding final state, **the entire metal mass at its melting temperature, without melting**
                    (no latent heat of fusion — the accident is assumed to stop before bulk melting, e.g. through thermal
                    expansion feedback):
                    ''')
                    st.latex(r'''N_f = \varepsilon\, m_{metal}\, \bar{C}_{p}\left[T_{melting}-T_0\right],
                             \qquad T_0 = 20\,^{\circ}\mathrm{C}''')
                    st.markdown(r'''
                    **Numerical evaluation** — input data from ICNC 2015 (Table II, established from LA-13638, IAEA 2008,
                    INL/EXT-10-19373 and the CRC Handbook):

                    | Metal | $T_{melting}$ (°C) | $C_p$ (J·kg⁻¹·°C⁻¹) | $C_p\,\Delta T$ (kJ/kg) | $N_f$ per kg |
                    |---|---|---|---|---|
                    | U-Mo (10 w% Mo) | 1150 | 150 | 169.5 | $5.95\times10^{15}$ |
                    | U (93.5 w% ²³⁵U) | 1135 | 116 | 129.3 | $4.54\times10^{15}$ |
                    | Pu (²³⁹Pu) | 640 | 130 | 80.6 | $2.83\times10^{15}$ |

                    The most penalizing case (U-Mo) fixes the leading coefficient, $5.95\times10^{15} \rightarrow 6\times10^{15}$
                    (rounded up), and the other metals are normalized to it:
                    $k_U = 4.54/5.95 = 0.76 \rightarrow 0.77$ and $k_{Pu} = 2.83/5.95 = 0.48 \rightarrow 0.5$
                    (both rounded up, conservative).

                    **Limits** — the thermal formula bounds all past metal criticality accidents except two (ICNC 2015, Fig. 5):
                    Sarov 1997 (6 days: heat loss neglected, adiabatic assumption no longer valid) and Livermore 1963
                    (10 kg of uranium actually melted, ~40% discrepancy). Very effective reflectors (WC, Be, U, LiD) are outside
                    the validation basis.

                    *References: M. Duluc & G. Caplin, ICNC 2015 (§3.2.2, Table II); M. Duluc et al., NCSD 2022 (Eq. 7).*
                    ''')
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
