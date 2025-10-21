import streamlit as st

# ______________________________________________________________________________________________________________________
st.write("# Documentation")

st.markdown("""
This section contains the technical documentation for the Slide-Rule project.
""")

st.divider()

# Création de deux colonnes pour afficher les PDFs côte à côte
col1, col2 = st.columns(2)

# ______________________________________________________________________________________________________________________
# Colonne 1 : Rapport principal
with col1:
    st.subheader("📊 Main Report")
    
    pdf_path_report = "./Ressources/Report/ASNR_2025-00084_EN_SlideRule_report_final.pdf"
    
    try:
        with open(pdf_path_report, "rb") as pdf_file:
            pdf_bytes_report = pdf_file.read()
        
        # Bouton de téléchargement
        st.download_button(
            label="📥 Download Report",
            data=pdf_bytes_report,
            file_name="ASNR_2025-00084_EN_SlideRule_report_final.pdf",
            mime="application/pdf",
            help="Click to download the main report",
            use_container_width=True
        )
        
        # Affichage du PDF
        st.pdf(pdf_bytes_report, height=800)
        
    except FileNotFoundError:
        st.error(f"File not found: {pdf_path_report}")
        st.info("Please check that the file exists in the Ressources/Report/ folder")
    except Exception as e:
        st.error(f"An error occurred while loading the PDF: {str(e)}")

# ______________________________________________________________________________________________________________________
# Colonne 2 : Spécifications des tâches
with col2:
    st.subheader("📋 Task Specifications")
    
    pdf_path_specs = "./Ressources/Specifications/Tasks_specifications.pdf"
    
    try:
        with open(pdf_path_specs, "rb") as pdf_file:
            pdf_bytes_specs = pdf_file.read()
        
        # Bouton de téléchargement
        st.download_button(
            label="📥 Download Specifications",
            data=pdf_bytes_specs,
            file_name="Tasks_specifications.pdf",
            mime="application/pdf",
            help="Click to download the task specifications",
            use_container_width=True
        )
        
        # Affichage du PDF
        st.pdf(pdf_bytes_specs, height=800)
        
    except FileNotFoundError:
        st.error(f"File not found: {pdf_path_specs}")
        st.info("Please check that the file exists in the Ressources/Specifications/ folder")
    except Exception as e:
        st.error(f"An error occurred while loading the PDF: {str(e)}")

