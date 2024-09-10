import os
import re
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from functions import get_input, handle_ticker, load_dataframe_from_excel, run_mappings, generate_13f_from_ws

from FactSet import FormulaDataProcessor

processor = FormulaDataProcessor()
is_healthy, health_info = processor.check_health()

st.header("SEC-13F Generation")

if is_healthy:
    st.info("""
            ✅ The FactSet API service is accessible.  
            This tool is set up to access the FactSet API using the :blue[WAYSTON-1792950] account.
            """)
else:
    st.info("""
            ❌ FactSet API upstream service is unavailable.  
            This tool is set up to access the FactSet API using the :blue[WAYSTON-1792950] account.
            """)

# Streamlit page setup
# st.set_page_config(page_title='13F Generation Tool', layout='wide')

# Creating tabs
tab1, tab2, tab3 = st.tabs(["SEC 13F Data", "Client Data", "SEC-13F-Report"])

def fill_in_ws(ws_df):
    # if 'working_sheet_df' in st.session_state:
    st.session_state['working_sheet_df'] = ws_df

# Map Clients CUSIPs data to generate 13F
def handle_cusips(client_df, cusip_column, quantity_column):
    print('Mapping client CUSIPs...')
    if client_df is not None:
        ws_df, runtime = run_mappings(client_df=client_df, sec13f_df=sec13f_df, eod_df=df_eod, cusip_col=cusip_column, quantity_col=quantity_column)
        
        ws_df = ws_df.rename({cusip_column: 'CUSIP (Client)'}, axis=1)

        ws_df['Match?'] = ws_df.apply(lambda x: True if x['CUSIP (SEC)'] is not None and x['CUSIP (Client)'].replace(' ', '') == x['CUSIP (SEC)'].replace(' ', '') else False, axis=1)
        ws_df['FIGI'] = ''
        ws_df['SH/PRN'] = 'SH'
        ws_df['PUT/CALL'] = ''
        ws_df['Market Value (Quantity*Price)'] = ''
        ws_df['De Minimis?'] = 'No'
        ws_df['Discretion Type'] = 'Sole'
        ws_df['Other Managers'] = ''
        ws_df['Sole'] = ws_df[quantity_column]
        ws_df['Shared'] = 0
        ws_df['None'] = 0
        ws_df['Complete?'] = False
        # ws_df['Market Value (Quantity*Price)'] = ws_df.apply(lambda x: x[quantity_column] * x['Price'], axis=1)
        ws_df['Market Value (Quantity*Price)'] = ws_df.apply(lambda x: 
                                                      np.nan if pd.isnull(x[quantity_column]) or pd.isnull(x['Price']) 
                                                      else x[quantity_column] * x['Price'], axis=1)
        ws_df['Market Value (Quantity*Price)'] = ws_df['Market Value (Quantity*Price)'].apply(np.ceil)

        ws_df = ws_df.rename({quantity_column: 'Quantity'}, axis=1)
        ws_df = ws_df[['Ticker (Client)','CUSIP (Client)', 'CUSIP (SEC)', 'Match?', 'Issuer Name (SEC)', 'Class (SEC)', 'SH/PRN', 'PUT/CALL', 'Quantity', 'Price', 'Market Value (Quantity*Price)', 'De Minimis?', 'Discretion Type', 'Other Managers', 'Sole', 'Shared', 'None', 'Complete?']]
        fill_in_ws(ws_df)
        st.info(f"Time Taken: {runtime}")
    else:
        st.error("Please upload client data first before trying to generate mappings")

def get_13f_file_content_as_bytes(path):
    with open(path, "rb") as file:  # Opening the file in binary mode
        return file.read()

with tab1:
    st.header(""":blue[SEC 13F Data]""")
    n_preview = 15
    sec_15_csv_path = '../resources/SEC-13F_FY2023_Q4.csv'
    sec_15_pdf_path = '../resources/SEC_13F_Securities_FY2024Q1.pdf'
    # Upload area for 13F Table
    # st.divider()

    st.caption(":white[Please ensure that the tool has been updated with the latest and most comprehensive \"SEC 13F\" data prior to processing \"Client Data\". Should a more recent or complete version be accessible, kindly upload it below:]")
    uploaded_sec13f_file = st.file_uploader(":blue[Please provide the most recent version of the SEC 13F Data in CSV format, adhering to the schema displayed below:]", type=['csv'], key="sec13f")
     # Only show the initial dataframe if no file has been uploaded yet.
    if uploaded_sec13f_file is None:
        sec13f_df = pd.read_csv(sec_15_csv_path)
        st.info(f"Tool is configured to use: :green[\"List of Section 13F Securities (SEC) FY2023-Q4\"] with :green[{len(sec13f_df)}] rows loaded")
        # st.caption("Default Preview:")
        st.dataframe(sec13f_df.head(n_preview))
        st.caption(f"Previewing :blue[{n_preview}] rows only")
    
        # Create a download button in the Streamlit app

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download SEC 13F File (CSV)",  # Text on the download button
                data=get_13f_file_content_as_bytes(sec_15_csv_path),  # The actual data for the file to be downloaded
                file_name=sec_15_csv_path.split('/')[-1],  # Name of the file that will be downloaded
                mime="text/csv"  # The MIME type of the file
            )

        with col2:
            st.download_button(
                label="Download SEC 13F File (PDF)",  # Text on the download button
                data=get_13f_file_content_as_bytes(sec_15_pdf_path),  # The actual data for the file to be downloaded
                file_name=sec_15_pdf_path.split('/')[-1],  # Name of the file that will be downloaded
                mime="application/pdf"  # The MIME type of the file
            )

    # Check if a file has been uploaded.
    if uploaded_sec13f_file is not None:
        # Define the expected column names
        expected_columns = ["CUSIP NO", "ASTRK", "ISSUER NAME", "ISSUER DESCRIPTION", "STATUS"]

        # Read the uploaded CSV file into a pandas DataFrame.
        sec13f_df = pd.read_csv(uploaded_sec13f_file)
        st.info(f"You uploaded SEC 13F csv with :green[{sec13f_df.shape[0]}] rows and :green[{sec13f_df.shape[1]}] columns")

        # Check if the uploaded file's schema matches the expected schema
        if list(sec13f_df.columns) == expected_columns:
            st.info(f"The newly uploaded file will be used for processing Client Data")        
        else:
            # The schema does not match, display an error message
            st.error("The uploaded file does not match the expected schema. Please ensure the CSV file has the following columns: " + ", ".join(expected_columns))

        # Perform your data processing here. For demonstration, we'll just display the DataFrame.
        st.write("Preview of uploaded SEC 13F Data:")
        st.dataframe(sec13f_df.head(n_preview))

    st.divider()
    
    # st.header(""":blue[EOD Data]""")
    # # Options for radio buttons
    # options = ["FactSet API", "EOD data (csv)", "FMP (Financial Modeling prep) API"]

    # # Create radio buttons and default to the first option
    # selection = st.radio("Select EOD data source:", options, index=0, horizontal=True)

    # # Logic to handle selections
    # if selection == options[1] or selection == options[2]:
    #     st.error("This option is currently disabled")
    # elif selection == options[0]:
    #     st.info("Tool is already configured to use FactSet API with account :blue[WAYSTON-1792950]")

    # Define the expected CSV schema
    # expected_schema = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
    
    # st.caption(":blue[https://www.eoddata.com/Download.aspx]")
    # Display the expected CSV schema to the user
    # st.caption(f"CSV Schema of upload: :blue[{expected_schema}]")
    

    # File uploader for the user to upload their EOD Data CSV
    # eod_file = st.file_uploader(":blue[Upload EOD Data (csv)]", type=['csv'], key="eod")

    # if eod_file is not None:
        # Attempt to read the uploaded file into a pandas DataFrame
    #     try:
    #         df_eod = pd.read_csv(eod_file)

    #         # Check if the uploaded file's schema matches the expected schema
    #         if list(df_eod.columns) == expected_schema:
    #             # Schema matches, proceed with displaying and processing the DataFrame
    #             n_preview = 15  # or whatever number of rows you want to preview
    #             st.caption(f"Previewing :blue[{n_preview}] rows only")
    #             st.dataframe(df_eod.head(n_preview))
    #             st.caption(f":green[{len(df_eod)}] data rows and :green[{len(df_eod.columns)}] columns are loaded")
    #         else:
    #             # Schema does not match, display an error message
    #             st.error("The uploaded file's schema does not match the expected schema. Please upload a file with the correct schema.")
    #     except Exception as e:
    #         # Handle any other error (e.g., file not readable as CSV)
    #         st.error(f"Error processing the uploaded file: {e}")
    # st.divider()

with tab2:
    client_df = None
    # Upload area for Client Data
    client_data_file = st.file_uploader(":blue[Upload Client Data]", type=['xlsx'], key="client_data")
    if client_data_file is not None:
        client_data_top_left, client_data_bottom_right, client_data_header = get_input("client_data")
        print(f"client_data_header = {client_data_header}")

        client_df = load_dataframe_from_excel(client_data_file, client_data_top_left, client_data_bottom_right, client_data_header)
        st.session_state['client_df'] = client_df
        st.write(f"Preview of Client Data ({n_preview} rows only):")
        st.dataframe(client_df.head(n_preview))
        st.caption(f":green[{len(client_df)}] data rows and :green[{len(client_df.columns)}] columns were loaded")
        
        # Dropdown for selecting the CUSIP column
        cusip_column = st.selectbox('Select the CUSIP column:', client_df.columns, key='cusip_col')
        # Dropdown for selecting the Quantity column
        quantity_column = st.selectbox('Select the Quantity column:', client_df.columns, key='quantity_col')
    st.divider()

    # Function that gets called when 'Client provided Ticker' is selected
    def handle_ticker():
        st.error("Support for \"Ticker Mapping\" has not been implemented yet.")

    # Radio buttons for user to select an option
    option = st.radio("Choose an option:", ("Client provided CUSIPs", "Client provided Ticker"))

    # Map button
    if st.button('Generate Mappings'):
        if cusip_column is not None:
            with st.spinner('Mapping CUSIPs to Ticker from QuantumOnline...'):
                if option == "Client provided CUSIPs":
                    handle_cusips(client_df, cusip_column, quantity_column)
                    st.success("Please switch to the \"SEC-13F\" tab to examine the mappings.")
                elif option == "Client provided Ticker":
                    handle_ticker()

with tab3: # '../resources/WorkingSheetTemplate.xlsx'

    def display_stdataframe(df):
        st.dataframe(df)

    def display_ag_grid_in_tab(df):
        # Customize your grid options as necessary
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_grid_options(enableRangeSelection=True)
        grid_options = gb.build()

        # Custom CSS for styling (if necessary)
        header_style = """
        <style>
        .ag-header-cell-label { justify-content: center; }
        /* Add your custom styles here */
        </style>
        """
        st.markdown(header_style, unsafe_allow_html=True)

        # Display the AGGrid with custom grid options
        AgGrid(df, gridOptions=grid_options, fit_columns_on_grid_load=True, update_mode='MODEL_CHANGED')

    header_details = [
        'Ticker (Client)', 'CUSIP (Client)', 'CUSIP (SEC)', 'FIGI', 'Issuer Name (N/A)',
        'Issuer Name (SEC)', 'Type', 'Class (SEC)', 'SH/PRN', 'PUT/CALL',
        'Quantity', 'Price', 'Market Value (Quantity*Price)', 'De Minimis?',
        'Discretion Type', 'Other Managers', 'Voting Authority Sole',
        'Voting Authority Shared', 'Voting Authority None', 'Complete?'
    ]

    st.caption(":blue[Generated Working Sheet]")
    # Check if the updated DataFrame is available in the session state
    if 'working_sheet_df' in st.session_state:
        # display_ag_grid_in_tab(st.session_state['working_sheet_df'])
        display_stdataframe(st.session_state['working_sheet_df'])
        quantum_join_summary = st.session_state['working_sheet_df']['Ticker (Client)'].count()
        sec_13f_join_summary = st.session_state['working_sheet_df']['CUSIP (SEC)'].count()
        eod_data_join_summary = st.session_state['working_sheet_df']['Price'].count()
        st.success(f"\"QuantumOnline\" found :blue[{quantum_join_summary}/{len(st.session_state['working_sheet_df'])}] Tickers. \"SEC 13F Data\" found :blue[{sec_13f_join_summary}/{len(st.session_state['working_sheet_df'])}] CUSIP matches. \"EOD Data\" found :blue[{eod_data_join_summary}/{len(st.session_state['working_sheet_df'])}] EOD Prices")
        # st.success(f"\"EOD Data\" found :blue[{eod_data_join_summary}/{len(st.session_state['working_sheet_df'])}] EOD Prices")

    else:
        # Display with an empty/default DataFrame if no update has been made
        working_sheet_df = pd.DataFrame(columns=header_details)
        st.session_state['working_sheet_df'] = working_sheet_df
        display_ag_grid_in_tab(working_sheet_df)
    

    st.divider()
    st.markdown("I intend to create the SEC 13F Excel Report using the aforementioned Working Sheet.")

    # st.json(st.session_state)
    if len(st.session_state['working_sheet_df']) > 0:
        sec_13f_excel = generate_13f_from_ws(st.session_state['working_sheet_df'])
        st.download_button('Download SEC-13F Report (Excel)',
                          data=sec_13f_excel,
                          file_name="sec_13f_report.xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    st.divider()
    st.markdown("I've updated the Working Sheet with the necessary information and wish to upload the revised version for generating the SEC 13F Excel Report.")

    uploaded_ws_file = st.file_uploader(":blue[Upload revised version of Working sheet]", type=['csv'], key="ws_revised")
    if uploaded_ws_file is not None:
        uploaded_ws_df = pd.read_csv(uploaded_ws_file)
        sec_13f_excel_revised = generate_13f_from_ws(uploaded_ws_df)

        st.download_button('Download SEC-13F Report (Excel) with Revised Working Sheet',
                          data=sec_13f_excel_revised,
                          file_name="sec_13f_report.xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

