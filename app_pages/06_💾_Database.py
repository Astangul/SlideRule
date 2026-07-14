import streamlit as st
from pathlib import Path
import datetime

# ______________________________________________________________________________________________________________________
st.write("# Database")

st.markdown("""
Download the Excel database file used as the data source for this application.
""")

st.divider()

db_path = Path("./Database/All-at-once_DB.xlsx")

try:
    db_bytes = db_path.read_bytes()
    last_modified = datetime.datetime.fromtimestamp(db_path.stat().st_mtime).strftime("%Y-%m-%d")

    st.download_button(
        label="📥 Download database (All-at-once_DB.xlsx)",
        data=db_bytes,
        file_name="All-at-once_DB.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Click to download the Excel database used by the Slide-Rule application",
        width='stretch'
    )
    st.caption(f"Last updated: {last_modified}")

except FileNotFoundError:
    st.error(f"File not found: {db_path}")
    st.info("Please check that the file exists in the Database/ folder")
except Exception as e:
    st.error(f"An error occurred while loading the database file: {str(e)}")
