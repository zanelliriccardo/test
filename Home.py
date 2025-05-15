import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import io
from datetime import datetime
import os
import shutil
import glob

# Set page config
st.set_page_config(page_title="Material Management Analysis", layout="wide")

# Function to get list of past runs
def get_past_runs():
    dirs = glob.glob("*_*")  # Matches vesselname_date patterns
    past_runs = []
    for dir_path in dirs:
        if os.path.isdir(dir_path):
            parts = dir_path.split("_")
            if len(parts) >= 2 and parts[1].isdigit():  # Simple validation
                date_str = datetime.strptime(parts[1], "%Y%m%d").strftime("%Y-%m-%d")
                past_runs.append({
                    "path": dir_path,
                    "vessel": parts[0],
                    "date": date_str,
                    "display": f"{parts[0]} ({date_str})"
                })
    return past_runs

# Function to load a past run
def load_past_run(run_dir):
    files = {
        "stocks": None,
        "consumption": None,
        "amos": None,
        "sap": None,
        "prices": None,
        "output": None
    }
    
    # Map file types to their expected filenames
    file_mapping = {
        "stocks": "stocks.xlsx",
        "consumption": "consumption.xlsx",
        "amos": "amos.xlsx",
        "sap": "sap.xlsx",
        "prices": "prices.xlsx",
        "output": "analysis_output.xlsx"
    }
    
    for file_type, filename in file_mapping.items():
        file_path = os.path.join(run_dir, filename)
        if os.path.exists(file_path):
            try:
                files[file_type] = pd.read_excel(file_path)
            except Exception as e:
                st.error(f"Error loading {file_type}: {str(e)}")
    
    return files

# Past runs selection section
past_runs = get_past_runs()
if past_runs:
    st.sidebar.header("Past Runs")
    selected_run = st.sidebar.selectbox(
        "Select a past run to load",
        options=[run["display"] for run in past_runs],
        index=0
    )
    
    if st.sidebar.button("Load Selected Run"):
        selected_run_dir = next(run["path"] for run in past_runs if run["display"] == selected_run)
        loaded_files = load_past_run(selected_run_dir)
        
        if loaded_files["output"] is not None:
            st.session_state.output_df = loaded_files["output"]
            st.success(f"Loaded analysis results from {selected_run}")
            
            # Display loaded files info
            with st.expander("Loaded Files Info"):
                st.write(f"**Vessel:** {next(run['vessel'] for run in past_runs if run['display'] == selected_run)}")
                st.write(f"**Date:** {next(run['date'] for run in past_runs if run['display'] == selected_run)}")
                
                cols = st.columns(5)
                file_status = {
                    "Stocks": "Loaded" if loaded_files["stocks"] is not None else "Missing",
                    "Consumption": "Loaded" if loaded_files["consumption"] is not None else "Missing",
                    "Amos": "Loaded" if loaded_files["amos"] is not None else "Missing",
                    "SAP": "Loaded" if loaded_files["sap"] is not None else "Missing",
                    "Prices":  "Loaded" if loaded_files["prices"] is not None else "Missing",
                }
                
                for i, (name, status) in enumerate(file_status.items()):
                    cols[i%4].metric(name, status)
        else:
            st.error("Could not load analysis results from selected run")

# Function to create vessel_date directory and save files
def save_to_vessel_directory(vessel_name, files_dict):
    # Create directory name with vessel and date
    current_date = datetime.now().strftime("%Y%m%d")
    dir_name = f"past_runs/{vessel_name}_{current_date}"
    
    # Create directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Save all files to the directory
    saved_files = []
    for file_type, file_data in files_dict.items():
        if file_data is not None:
            file_path = os.path.join(dir_name, f"{file_type}.xlsx")
            
            if isinstance(file_data, pd.DataFrame):
                with pd.ExcelWriter(file_path) as writer:
                    file_data.to_excel(writer, index=False)
            else:
                # For uploaded files
                with open(file_path, "wb") as f:
                    f.write(file_data.getbuffer())
            
            saved_files.append(file_path)
    
    return dir_name, saved_files

# Title and description
st.title("Material Management Analysis Tool")
st.markdown("""
This tool analyzes material management data based on uploaded Excel files.
Upload the required files and click "Run Analysis" to generate results.
""")


# Vessel name input and config version selection
st.header("Analysis Configuration")

vessel_name = st.selectbox(
    "Vessel Name*", 
    options=[
        "DVD", "Scarabeo 8", "Scarabeo 9", "Santorini",
        "Pioneer", "Perro Negro 4",
        "Perro Negro 7", "Perro Negro 8", "Perro Negro 10",
        "Perro Negro 11", "Perro Negro 12", "Perro Negro 13",
        "Saipem 10000", "Saipem 12000", 
    ], 
    help="Select the vessel name that will be used in the output filename"
)
# Check if config file exists
config_exists = os.path.exists(f"material_mgmt/{vessel_name}/config.xlsx")

# Set checkbox state - disabled if config doesn't exist
if config_exists:
    use_last_config = st.checkbox("Use last config version", value=True,
                                help="Use the existing config.xlsx file")
else:
    use_last_config = st.checkbox("Use last config version", value=False,
                                help="No config.xlsx file found - please upload a config file",
                                disabled=True)
    # Show config file uploader if needed
    if not use_last_config or not config_exists:
        config_file = st.file_uploader("Upload Config File", type=["xlsx"])

# Try to load config if enabled
config_data = None
if use_last_config and config_exists:
    try:
        config_data = pd.ExcelFile(f"material_mgmt/{vessel_name}/config.xlsx")
        st.session_state['config'] = config_data
        st.success("Config file loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load config file: {str(e)}")
        use_last_config = False  # Uncheck if loading fails
elif config_file is not None:
    try:
        config_data = pd.ExcelFile(config_file)
        st.session_state['config'] = config_data
        st.success("Uploaded config file loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load uploaded config file: {str(e)}")


# File upload section
st.header("Upload Required Files")

with st.expander("File Upload Instructions"):
    st.markdown("""
    Please upload the following Excel files:
    1. **Stocks Extraction**: Contains stock information (Sheet6)
    2. **Consumption Extraction**: Contains consumption data ('consumption' sheet)
    3. **Criticality Computation**: Contains criticality data (Sheet1)
    4. **Class Analysis**: Contains volume_value data
    """)

col1, col2 = st.columns(2)
with col1:
    stocks_file = st.file_uploader("Stocks Extraction", type="xlsx")
    consumption_file = st.file_uploader("Consumption Extraction", type="xlsx")
    amos_file = st.file_uploader("AMOS Extraction", type="xlsx")
with col2:
    sap_file = st.file_uploader("SAP Extraction", type="xlsx")
    prices_file = st.file_uploader("Prices", type="xlsx")

# Initialize session state for the output dataframe
if 'output_df' not in st.session_state:
    st.session_state.output_df = None


from utils import *
def run_analysis(stocks_file, consumption_file, prices_file, amos_file, sap_file):
    """Main analysis function that orchestrates all the steps."""
    # Load and clean data
    stocks, consumption, consumption_frequency, prices, sap, amos = load_data(
        stocks_file, consumption_file, prices_file, amos_file, sap_file
    )

    load_config(config_data=pd.ExcelFile(config_file))

    amos = clean_amos_data(amos)
    sap = clean_sap_data(sap)
    
    # Prepare stocks copy for merging
    stocks_copy = stocks.copy()
    stocks_copy['Number'] = stocks_copy['Number'].astype(str)
    
    # Merge datasets
    sap_amos = merge_datasets(sap, amos, consumption_frequency, stocks_copy)
    
    # Calculate criticality metrics
    sap_amos = calculate_criticality_weights(sap_amos, vessel_name)
    sap_amos = calculate_consumption_weights(sap_amos)
    sap_amos = calculate_supply_time_weights(sap_amos)
    sap_amos_criticality = calculate_criticality_score(sap_amos, vessel_name)
    
    # Merge all data
    merged = merge_all_data(stocks, consumption, sap_amos_criticality, prices)
    merged = prepare_merged_data(merged)
    
    # Calculate service levels and categories
    merged = calculate_service_level(merged)
    merged = calculate_pareto_class(merged)
    merged = categorize_items(merged)
    
    # Calculate inventory parameters
    merged = calculate_inventory_parameters(merged)
    merged = adjust_categories(merged)
    merged = adjust_inventory_parameters(merged)
    
    # Final calculations and output preparation
    merged = map_showed_categories(merged)
    merged = calculate_alignment_metrics(merged)
    output = prepare_final_output(merged)
    
    print('Merged Shape')
    print(merged.shape, consumption.shape, sap_amos_criticality.shape, prices.shape)
    
    return output

# Run analysis button
if st.button("Run Analysis"):
    if not vessel_name:
        st.error("Please enter a vessel name before running the analysis.")
    elif all([stocks_file, consumption_file, prices_file, amos_file, sap_file]):
        with st.spinner("Running analysis..."):
            try:
                # Run the analysis
                output_df = run_analysis(stocks_file, consumption_file, prices_file, amos_file, sap_file)
                #output_df.to_excel('output.xlsx')
                # Store in session state
                st.session_state.output_df = output_df
                
                st.success("Analysis completed successfully!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                raise e
    else:
        st.warning("Please upload all required files before running the analysis.")

# Display results if available
if st.session_state.output_df is not None:
    st.header("Analysis Results")
    
    # Create filters above the data editor
    st.subheader("Filters")
    
    # First row of filters
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        number_filter = st.multiselect(
            "Filter by Number",
            options=sorted(st.session_state.output_df['Number'].unique()),
            default=None
        )

    with row1_col2:
        category_filter = st.multiselect(
            "Filter by Category",
            options=sorted(st.session_state.output_df['Category'].unique()),
            default=None
        )

    with row1_col3:
        modify_category_filter = st.multiselect(
            "Filter by Modify Category",
            options=sorted(st.session_state.output_df['Modified Category'].unique()),
            default=None
        )

    # Second row of filters
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        class_filter = st.multiselect(
            "Filter by Class",
            options=sorted(st.session_state.output_df['Product Class'].unique()),
            default=None
        )

    with row2_col2:
        pareto_class_filter = st.multiselect(
            "Filter by Pareto Class",
            options=sorted(st.session_state.output_df['Pareto Class'].unique()),
            default=None
        )

    with row2_col3:
        showed_category_filter = st.multiselect(
            "Filter by Showed Category",
            options=sorted(st.session_state.output_df['Showed Category'].unique()),
            default=None
        )
    
    # Apply filters
    filtered_df = st.session_state.output_df.copy()
    
    if number_filter:
        filtered_df = filtered_df[filtered_df['Number'].isin(number_filter)]
    
    if category_filter:
        filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]
    
    if modify_category_filter:
        filtered_df = filtered_df[filtered_df['Modified Category'].isin(modify_category_filter)]
    
    if class_filter:
        filtered_df = filtered_df[filtered_df['Product Class'].isin(class_filter)]
    
    if pareto_class_filter:
        filtered_df = filtered_df[filtered_df['Pareto Class'].isin(pareto_class_filter)]

    if showed_category_filter:
        filtered_df = filtered_df[filtered_df['Showed Category'].isin(showed_category_filter)]
    
    # Show filtered dataframe in editor
    st.subheader("Filtered Results")
    edited_df = st.data_editor(
        filtered_df,
        num_rows="dynamic",
        use_container_width=True
    )

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Sum of under_reorder
        total_reorder_qty = filtered_df.loc[filtered_df['Under Reorder Quantity'] > 0,'Under Reorder Quantity'].sum()
        st.metric(
            label="Total Quantity Below Reorder Level",
            value=f"{total_reorder_qty:,.0f}",
            help="Sum of all items that need reordering"
        )

    with col2:
        # Sum of quanto_costa
        total_reorder_cost = filtered_df['Alignment Cost'].sum()
        st.metric(
            label="Total Reorder Cost",
            value=f"€ {total_reorder_cost:,.2f}",
            help="Total estimated cost for all reorders"
        )

    # Update session state with edited dataframe
    # Note: We only update the rows that match the filters to preserve other edits
    if any([number_filter, category_filter, modify_category_filter, class_filter, pareto_class_filter]):
        # Get the indices of the filtered rows
        filtered_indices = filtered_df.index
        # Update only those rows in the original dataframe
        st.session_state.output_df.loc[filtered_indices] = edited_df
    else:
        # If no filters are applied, update the entire dataframe
        st.session_state.output_df = edited_df

    # Modified Excel download with config sheet
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write analysis data to main sheet
        st.session_state.output_df.to_excel(writer, sheet_name='Analysis Results', index=False)
        
        # Write config data if available
        if config_data is not None:
            # Save each table to a separate sheet
            for sheet_name in config_data.sheet_names:
                config_data.parse(sheet_name).to_excel(writer, sheet_name=sheet_name, index=False)

    
    current_date = datetime.now().strftime("%Y%m%d %H:%M")
    filename = f"{vessel_name}_{current_date}_analysis_with_config.xlsx"
    
    st.download_button(
        label="Download Analysis with Config",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )