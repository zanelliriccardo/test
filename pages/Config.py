import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import io
from datetime import date, timedelta

def main():
    st.set_page_config(page_title="Material Management Config", layout="wide")
    st.title("Material Management Configuration Editor")

    config_data = st.session_state['config']
    
    # Sidebar or top selectbox to choose the sheet
    sheet_names = config_data.sheet_names
    selected_sheet = st.selectbox("Select a sheet in config file to display", sheet_names)

    # Load the selected sheet into a DataFrame
    df = config_data.parse(selected_sheet)

    # Display the sheet name and table
    st.subheader(f"Sheet: {selected_sheet}")
    st.data_editor(df)
        # Download button
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Save each table to a separate sheet
        for sheet_name in sheet_names:
            config_data.parse(sheet_name).to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()
    
    st.download_button(
        label="Download Configuration to Excel",
        data=buffer,
        file_name="config.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
if __name__ == "__main__":
    main()