import os
import time
from io import BytesIO
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode


from src.functions import (
    get_input,
    handle_ticker,
    load_dataframe_from_excel,
    run_mappings,
    generate_13f_from_ws,
    check_client_df_sanity,
    parallel_fetch_cusips,
    last_date_of_previous_quarter,
    Cleaning_Top_And_Bottom_rows,
)

from src.FactSet import FormulaDataProcessor


processor = FormulaDataProcessor()
is_healthy, health_info = processor.check_health()


st.header("SEC-13F Generation")


# Strreamlit setup Login
import hmac


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()


if is_healthy:
    st.info(
        """
           ✅ The FactSet API service is accessible. 
           This tool is set up to access the FactSet API using the :blue[WAYSTON-1844786] account.
           """
    )
else:
    st.info(
        """
           ❌ FactSet API upstream service is unavailable. 
           This tool is set up to access the FactSet API using the :blue[WAYSTON-1844786] account.
           """
    )


# Streamlit page setup
# st.set_page_config(page_title='13F Generation Tool', layout='wide')


# Creating tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Cleaning", "SEC-13F-Data", "Client Data", "Working Sheet", "SEC-13F-Report"]
)


def classify_sh_prn(class_value):
    equity_keywords = ["COM", "SHS", "CL A", "CL B", "CL C", "CLA", "ETF"]
    debt_keywords = ["EXP", "NOTE", "SDCV", "FRNT", "MTNF", "DEBT", "DBCV"]

    for keyword in equity_keywords:
        if keyword in class_value:
            return "SH"
    for keyword in debt_keywords:
        if keyword in class_value:
            return "PRN"
    return None  # or some default value


def classify_put_call(class_value):
    option_keywords = ["PUT", "CALL"]
    if "PUT" in class_value:
        return "PUT"
    elif "CALL" in class_value:
        return "CALL"
    return None


def fill_in_ws(ws_df):
    # if 'working_sheet_df' in st.session_state:
    st.session_state["working_sheet_df"] = ws_df


# Map Clients CUSIPs data to generate 13F
def handle_cusips(client_df, cusip_column, quantity_column, price_as_on_date):
    print("Mapping client CUSIPs...")
    if client_df is not None:
        # ws_df, runtime = run_mappings(client_df=client_df, sec13f_df=sec13f_df, cusip_col=cusip_column, quantity_col=quantity_column)
        ws_df = run_mappings(
            client_df=client_df,
            sec13f_df=sec13f_df,
            cusip_col=cusip_column,
            quantity_col=quantity_column,
            price_as_on_date=price_as_on_date,
        )

        ws_df = ws_df.rename(
            {cusip_column: "CUSIP (Client)", "EODPrice": "Price"}, axis=1
        )

        ws_df["SEC Match?"] = ws_df.apply(
            lambda x: True
            if pd.notnull(x["CUSIP (SEC)"])
            and str(x["CUSIP (Client)"]).replace(" ", "")
            == str(x["CUSIP (SEC)"]).replace(" ", "")
            else False,
            axis=1,
        )
        ws_df["FIGI"] = ""
        ws_df["SH/PRN"] = ws_df.apply(
            lambda x: classify_sh_prn(str(x["Class (SEC)"])), axis=1
        )
        ws_df["PUT/CALL"] = None
        ws_df["Market Value (Quantity*Price)"] = ""
        ws_df["De Minimis?"] = "No"
        ws_df["Discretion Type"] = "Sole"
        ws_df["Other Managers"] = ""
        ws_df["Sole"] = ws_df[quantity_column]
        ws_df["Shared"] = 0
        ws_df["None"] = 0
        ws_df["Complete?"] = ws_df.apply(
            lambda x: True if x["SEC Match?"] and x["Price"] and x["Ticker"] else False,
            axis=1,
        )
        ws_df["Market Value (Quantity*Price)"] = ws_df.apply(
            lambda x: np.nan
            if pd.isnull(x[quantity_column]) or pd.isnull(x["Price"])
            else x[quantity_column] * x["Price"],
            axis=1,
        )
        # changed ceil to round
        ws_df["Market Value (Quantity*Price)"] = ws_df[
            "Market Value (Quantity*Price)"
        ].apply(np.round)
        # PUT/CALL IMPLEMENTATION
        if desc_column is not None:
            ws_df["PUT/CALL"] = desc_col_values
            # print("ws----------------------descripti4oireog", ws_df)
            # ws_df['PUT/CALL'] = ws_df.apply(lambda x: "PUT" if ("put" in x["description"].lower()) or (
            #         re.findall("p[0-9]", x["description"].lower()) and
            #         re.findall("p[0-9]", x["description"].lower())[0] in x["description"].lower()) else
            # ("CALL" if ("call" in x["description"].lower()) or (re.findall("c[0-9]", x["description"].lower()) and
            #                                                     re.findall("c[0-9]", x["description"].lower())[0] in x[
            #                                                         "description"].lower()) else None), axis=1)
            # print("ws----------------------descripti4oireog", ws_df)
            # ws_df.drop('description', axis=1, inplace=True)

        # deminimus by client
        # ws_df['De Minimis?'] = ws_df.apply(
        #     lambda x: "No" if x[quantity_column] < 10000 and x['Market Value (Quantity*Price)'] < 200000 else "Yes",
        #     axis=1)
        if ws_df is not None:
            ws_df["De Minimis?"] = ws_df.apply(
                lambda x: "-"
                if pd.isna(x[quantity_column]) or pd.isna(x["Market Value (Quantity*Price)"])
                else "Yes"
                if x[quantity_column] < 10000
                   and x["Market Value (Quantity*Price)"] < 200000
                else "No",
                axis=1)
        else:
            print("Not Available")
        print("Data............", ws_df)
        ws_df = ws_df.rename({quantity_column: "Quantity"}, axis=1)
        # Excluding CUsip's WhicH are not in SEC 13f
        print("Beforeeeeeeeeeee\n:", ws_df)
        ws_df = ws_df.dropna(subset=["CUSIP (SEC)"]).reset_index(drop=True)
        print("Afterreeeeeeeeeee\n:", ws_df)

        fill_in_ws(ws_df)
    else:
        st.error("Please upload client data first before trying to generate mappings")


def get_13f_file_content_as_bytes(path):
    with open(path, "rb") as file:  # Opening the file in binary mode
        return file.read()


def convert_ws_to_excel(ws_df):
    from io import BytesIO

    st.write("Download Working sheet with colour coded rows for missing information.")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        ws_df.to_excel(writer, index=False, sheet_name="SEC-13F WorkingSheet")

        # Get the xlsxwriter workbook and worksheet objects.
        workbook = writer.book
        worksheet = writer.sheets["SEC-13F WorkingSheet"]

        # Define the formats.
        yellow_format = workbook.add_format({"bg_color": "#FFFF00"})  # Yellow
        orange_format = workbook.add_format({"bg_color": "#FFA500"})  # Orange

        # Get the default header list
        headers = list(ws_df.columns)

        # Conditional formatting for specific columns with None or blank values.
        # Adjusting column letter references might be needed based on actual data.
        for i, header in enumerate(headers, 1):  # Column indices in Excel start from 1
            col_letter = chr(65 + i - 1)  # Convert column index to letter
            if header in ["Issuer Name (SEC)", "Class (SEC)"]:
                worksheet.conditional_format(
                    f"{col_letter}2:{col_letter}{len(ws_df) + 1}",
                    {"type": "blanks", "format": yellow_format},
                )
                worksheet.conditional_format(
                    f"{col_letter}2:{col_letter}{len(ws_df) + 1}",
                    {
                        "type": "formula",
                        "criteria": f'={col_letter}2=""',
                        "format": yellow_format,
                    },
                )
            elif header in ["Ticker", "Price"]:
                worksheet.conditional_format(
                    f"{col_letter}2:{col_letter}{len(ws_df) + 1}",
                    {"type": "blanks", "format": orange_format},
                )
                worksheet.conditional_format(
                    f"{col_letter}2:{col_letter}{len(ws_df) + 1}",
                    {
                        "type": "formula",
                        "criteria": f'={col_letter}2=""',
                        "format": orange_format,
                    },
                )

    output.seek(0)  # Important: move back to the beginning of the BytesIO object!
    return output


def validate_ws_cols(df):
    # List of required columns
    required_columns = [
        "CUSIP (Client)",
        "Quantity",
        "CUSIP (SEC)",
        "Issuer Name (SEC)",
        "Class (SEC)",
        "Status (SEC)",
        "Price",
        "Ticker",
        "SEC Match?",
        "FIGI",
        "SH/PRN",
        "Market Value (Quantity*Price)",
        "De Minimis?",
        "Discretion Type",
        "Other Managers",
        "Sole",
        "Shared",
        "None",
        "Complete?",
    ]

    # Optional columns
    optional_columns = ["PUT/CALL"]

    # Check for missing required columns
    missing_columns = set(required_columns) - set(df.columns)

    # Check for extra columns excluding optional ones
    extra_columns = set(df.columns) - set(required_columns) - set(optional_columns)

    if missing_columns:
        return (
            False,
            f"DataFrame is missing the following required columns: {', '.join(missing_columns)}",
        )
    if extra_columns:
        return (
            False,
            f"DataFrame has the following extra columns that are not required: {', '.join(extra_columns)}",
        )

    # If all required columns are present and extra columns are valid, return success
    print("DataFrame has the required columns.")
    return True, ""


# Example usage:
# result, message = validate_ws_cols(your_dataframe)
# if not result:
#     print(message)


# Example usage:
# result, message = validate_ws_cols(your_dataframe)
# if not result:
#     print(message)


with tab0:
    n_preview = 15
    client_df = None
    st.subheader("Combine Datasets")
    client_data_file_1 = st.file_uploader(
        ":blue[Upload Client Data first file]", type=["xlsx"], key="client_data3"
    )
    client_data_file_2 = st.file_uploader(
        ":blue[Upload Client Data second file]", type=["xlsx"], key="client_data4"
    )

    if client_data_file_1 and client_data_file_2 is not None:
        client_df_1 = pd.read_excel(client_data_file_1)
        st.session_state["client_df_1"] = client_df_1
        st.write(f"Preview of Client Data ({n_preview} rows only):")
        st.dataframe(client_df_1.head(n_preview))
        st.caption(
            f":green[{len(client_df_1)}] data rows and :green[{len(client_df_1.columns)}] columns were loaded"
        )

        client_df_2 = pd.read_excel(client_data_file_2)
        st.session_state["client_df_2"] = client_df_2
        st.write(f"Preview of Client Data ({n_preview} rows only):")
        st.dataframe(client_df_2.head(n_preview))
        st.caption(
            f":green[{len(client_df_2)}] data rows and :green[{len(client_df_2.columns)}] columns were loaded"
        )

        list_of_path = [client_data_file_1, client_data_file_2]
        combined_df = pd.DataFrame()

        for i in list_of_path:
            df = pd.read_excel(i)
            unnamed_columns = df.columns[
                df.columns.str.startswith("Unnamed:") & ~df.columns.isna()
            ]
            percentage_unnamed = len(unnamed_columns) / len(df.columns)

            if percentage_unnamed > 0.7:
                df.columns = df.iloc[0]
                df = df.drop(df.index[0])

            nan_count = df.columns.isnull().sum()
            total_columns = len(df.columns)
            nan_proportion = nan_count / total_columns

            if nan_proportion > 0.7:
                while any(pd.isna(df.columns)):
                    df.columns = df.iloc[0]
                    df = df.drop(df.index[0])

            df.reset_index(drop=True, inplace=True)
            last_20_rows = df.tail(20)
            none_counts = last_20_rows.isna().sum(axis=1)
            none_threshold = 0.6 * len(df.columns)
            filtered_rows = last_20_rows[none_counts >= none_threshold]
            df.drop(filtered_rows.index, inplace=True)

            df = df.reset_index(drop=True)
            combined_df = pd.concat([combined_df, df], ignore_index=True, sort=False)

        file_name = st.text_input(
            "Enter a name for the merged file:", value="Merged_File"
        )

        if st.button("Merge Files"):

            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Sheet1")
                processed_data = output.getvalue()
                return processed_data

            def main():
                excel_data = to_excel(combined_df)
                st.caption(
                    "Your file is ready. Please click the below button to download the merged file."
                )
                st.download_button(
                    label="Download Excel file",
                    data=excel_data,
                    file_name=f"{file_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            if __name__ == "__main__":
                main()

    # Upload area for Client Data
    st.subheader("Combined multiple sheets into one sheet ")
    st.write("""Kindly provide the client data in an XLSX file. """)
    client_data_file = st.file_uploader(
        ":blue[Upload Client Data]", type=["xlsx"], key="client_data_combined"
    )

    # Path to the Excel file
    # excel_file_path = r"C:\Users\mm3816\Downloads\Client raw files (1)\Octsgon\Portfolio 3.31.24 (2).xlsx"
    combined_df = pd.DataFrame()
    if client_data_file is not None:
        xls = pd.ExcelFile(client_data_file)

        combined_df = pd.DataFrame()

        # Loop through each sheet name
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            combined_df = Cleaning_Top_And_Bottom_rows(df, combined_df)

        # # Save the combined DataFrame to a new Excel file
        # combined_df.to_excel(r'C:\Users\mm3816\Desktop\ravi_files\combined_sheets.xlsx', index=False)

        # print("All sheets have been combined into 'combined_sheets_aligned.xlsx'.")
        # print(combined_df.columns)
        # print(combined_df)
        st.dataframe(combined_df)
        if st.button("combine all sheets into one"):

            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Sheet1")
                processed_data = output.getvalue()
                return processed_data

            def main():
                excel_data = to_excel(combined_df)
                # Create a download button
                st.download_button(
                    label="Download Excel file",
                    data=excel_data,
                    file_name=f"cleaned.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            if __name__ == "__main__":
                main()
    # Upload area for Client Data
    st.subheader("Locate Missing CUSIPs")
    st.write("Please note: The ticker symbol is required.")
    st.write("""Kindly provide the client data in an XLSX file. """)
    client_data_file = st.file_uploader(
        ":blue[Upload Client Data]", type=["xlsx"], key="client_data1"
    )
    combined_df = pd.DataFrame()
    if client_data_file is not None:
        client_df = pd.read_excel(client_data_file)
        st.session_state["client_df"] = client_df
        n_preview = 5  # Set the number of rows to preview
        st.write(f"Preview of Client Data ({n_preview} rows only):")
        # client_df = Cleaning_Top_And_Bottom_rows(client_df, combined_df)
        st.dataframe(client_df.head(n_preview))
        st.caption(
            f":green[{len(client_df)}] data rows and :green[{len(client_df.columns)}] columns were loaded."
        )

        # Check if 'cusip' column exists (case insensitive)
        if not any(client_df.columns.str.lower() == "cusip"):
            client_df["cusip"] = None  # Add cusip column if it doesn't exist

        # Dropdown for selecting the cusip column
        cusip_column = st.selectbox(
            "Select the cusip column :", client_df.columns, key="cusip_col1"
        )
        # Dropdown for selecting the ticker column
        ticker_column = st.selectbox(
            "Select the ticker column :", client_df.columns, key="ticker_col1"
        )
        st.caption(
            f"Now we have :green[{len(client_df)}] data rows and :green[{client_df[cusip_column].isnull().sum()}] missing cusips"
        )

    if st.button("Find Cusips"):
        if client_data_file is not None:
            start_time = time.time()
            mapped_df = parallel_fetch_cusips(
                client_df, ticker_col=ticker_column, cusip_col=cusip_column
            )
            end_time = time.time()
            print(end_time - start_time)
            st.dataframe(client_df)
            st.caption(
                f"Now  we have :green[{client_df[cusip_column].isnull().sum()}] missing cusips"
            )

            # Function to convert dataframe to excel
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Sheet1")
                processed_data = output.getvalue()
                return processed_data

            # Main function to handle download button
            def main():
                excel_data = to_excel(client_df)
                # Create a download button
                st.download_button(
                    label="Download Excel file",
                    data=excel_data,
                    file_name=f'{client_data_file.name.split(".")[0]}_cleaned.xlsx',
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            if __name__ == "__main__":
                main()


with tab1:
    st.header(""":blue[SEC 13F Data]""")
    n_preview = 15
    sec_15_csv_path = "mm-waystone-sec13f-rpa-automation/resources/13flist2024q2.csv"

    sec_15_pdf_path = "mm-waystone-sec13f-rpa-automation/resources/13flist2024q2.pdf"
    # Upload area for 13F Table
    # st.divider()

    st.caption(
        ':white[Please ensure that the tool has been updated with the latest and most comprehensive "SEC 13F" data prior to processing "Client Data". Should a more recent or complete version be accessible, kindly upload it below:]'
    )
    uploaded_sec13f_file = st.file_uploader(
        ":blue[Please provide the most recent version of the SEC 13F Data in CSV format, adhering to the schema displayed below:]",
        type=["csv"],
        key="sec13f",
    )
    # Only show the initial dataframe if no file has been uploaded yet.
    if uploaded_sec13f_file is None:
        sec13f_df = pd.read_csv(sec_15_csv_path)
        file_name = os.path.splitext(os.path.basename(sec_15_csv_path))[0]
        info_string = f'The tool is presently set up to utilize the :green["List of Section 13F Securities ({file_name})"] dataset, which comprises :green[{len(sec13f_df)}] rows'

        # Display the information
        st.info(info_string)
        st.dataframe(sec13f_df.head(n_preview))
        st.caption(f"Previewing :blue[{n_preview}] rows only")

        # Create a download button in the Streamlit app

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download SEC 13F File (CSV)",  # Text on the download button
                data=get_13f_file_content_as_bytes(
                    sec_15_csv_path
                ),  # The actual data for the file to be downloaded
                file_name=sec_15_csv_path.split("/")[
                    -1
                ],  # Name of the file that will be downloaded
                mime="text/csv",  # The MIME type of the file
            )

        with col2:
            st.download_button(
                label="Download SEC 13F File (PDF)",  # Text on the download button
                data=get_13f_file_content_as_bytes(
                    sec_15_pdf_path
                ),  # The actual data for the file to be downloaded
                file_name=sec_15_pdf_path.split("/")[
                    -1
                ],  # Name of the file that will be downloaded
                mime="application/pdf",  # The MIME type of the file
            )

    # Check if a file has been uploaded.
    if uploaded_sec13f_file is not None:
        # Define the expected column names
        expected_columns = [
            "CUSIP NO",
            "ASTRK",
            "ISSUER NAME",
            "ISSUER DESCRIPTION",
            "STATUS",
        ]

        # Read the uploaded CSV file into a pandas DataFrame.
        sec13f_df = pd.read_csv(uploaded_sec13f_file)

        # Check if the uploaded file's schema matches the expected schema
        if list(sec13f_df.columns) == expected_columns:
            st.info(
                f"""
                   You uploaded :green[List of Section 13F Securities {uploaded_sec13f_file.name}] with :green[{sec13f_df.shape[0]}] rows and :green[{sec13f_df.shape[1]}] columns. 
                   The newly uploaded file will be used for processing client data.
                   """
            )
        else:
            # The schema does not match, display an error message
            st.error(
                "The uploaded file does not match the expected schema. Please ensure the CSV file has the following columns: "
                + ", ".join(expected_columns)
            )

        # Perform your data processing here. For demonstration, we'll just display the DataFrame.
        st.write("Preview of uploaded SEC 13F Data:")
        st.dataframe(sec13f_df.head(n_preview))

    st.divider()


with tab2:
    client_df = None
    # Upload area for Client Data
    client_data_file = st.file_uploader(
        ":blue[Upload Client Data]", type=["xlsx"], key="client_data"
    )
    if client_data_file is not None:
        client_data_file_name = client_data_file.name
        if "_cleaned" in client_data_file_name:
            client_data_file_name = client_data_file_name.split("_cleaned")[0]
        else:
            client_data_file_name = client_data_file_name.split(".xlsx")[0]
        client_df = pd.read_excel(client_data_file)
        st.session_state["client_df"] = client_df
        st.write(f"Preview of Client Data ({n_preview} rows only):")
        st.dataframe(client_df.head(n_preview))
        st.caption(
            f":green[{len(client_df)}] data rows and :green[{len(client_df.columns)}] columns were loaded"
        )

        client_df_columns = list(client_df.columns)
        client_df_columns.append(None)
        # Dropdown for selecting the CUSIP column
        cusip_column = st.selectbox(
            "Select the Identifier column:*",
            client_df_columns,
            index=client_df_columns.index(None),
            key="cusip_col",
        )
        # Dropdown for selecting the Quantity column
        quantity_column = st.selectbox(
            "Select the Quantity column:*",
            client_df_columns,
            index=client_df_columns.index(None),
            key="quantity_col",
        )

        desc_column = st.selectbox(
            "Select the Description(Options) column:*",
            client_df_columns,
            index=client_df_columns.index(None),
            key="description1",
        )
        # price_column = st.selectbox("Select the Price column:*", client_df_columns, index=client_df_columns.index(None), key='price1')
        market_value_column = st.selectbox(
            "Select the Market column:*",
            client_df_columns,
            index=client_df_columns.index(None),
            key="market_value1",
        )

        # DF with all positive values
        if quantity_column is not None:

            def drop_rows(row):
                if "-" in str(row[quantity_column]):
                    client_df.drop(index=row.name, inplace=True)

            # Apply the function using lambda and axis=1 to apply row-wise
            client_df.apply(lambda x: drop_rows(x), axis=1)
            client_df[quantity_column] = client_df[quantity_column].replace("$", "")
        if cusip_column is not None:
            # ids = list(str(client_df[cusip_column]))
            # ids=list(str(client_df[cusip_column]))

            pad_cusip = lambda x: x if len(x) == 9 else "0" * (9 - len(x)) + x
            # Apply the lambda function to the 'cusip' column
            client_df[cusip_column] = client_df[cusip_column].astype(str)
            client_df[cusip_column] = (
                client_df[cusip_column].apply(pad_cusip).astype(str)
            )

            # client_df[cusip_column] = client_df[cusip_column].str.replace('[^a-zA-Z0-9]', '', regex=True)

            # Filter out CUSIPs with more than 9 characters
            # client_df = client_df[client_df[cusip_column].str.len() <= 9]

            # Pad CUSIPs with fewer than 9 characters with leading zeros
            # client_df[cusip_column] = client_df[cusip_column].str.apply(lambda x: x.zfill(9))

            ids = list(client_df[cusip_column])
            price_as_on_date = datetime(2024, 6, 30)
            as_on_date = price_as_on_date.strftime("%m/%d/%Y")
            display_names = ["EODPrice"]
            timeSeries_formulas = ["P_PRICE(NOW)"]
            cross_formulas = [f"FG_PRICE({as_on_date})"]
            # cross_series_df = processor.fetch_data(ids, cross_formulas, display_names)
            # time_Series_df = processor.fetch_time_series_data(
            #     ids, timeSeries_formulas, display_names
            # )
            # merged_df = pd.merge(
            #     cross_series_df,
            #     time_Series_df,
            #     on="requestId",
            #     suffixes=("_df1", "_df2"),
            # )
            # print("merged_df", merged_df)
           
            def fetch_with_retry(fetch_function, *args, max_retries=5, delay=5, **kwargs):
                """
                Fetch data with retries if None is returned.
                
                Args:
                    fetch_function: The function to fetch data.
                    *args: Positional arguments to pass to the fetch function.
                    max_retries: Maximum number of retries before giving up.
                    delay: Time in seconds to wait between retries.
                    **kwargs: Keyword arguments to pass to the fetch function.
                    
                Returns:
                    DataFrame: Retrieved data as a DataFrame.
                """
                retries = 0
                while retries < max_retries:
                    result = fetch_function(*args, **kwargs)
                    if result is not None and not result.empty:
                        return result
                    retries += 1
                    print(f"Data fetch failed. Retrying {retries}/{max_retries}...")
                    time.sleep(delay)
                raise RuntimeError("Failed to fetch data after several retries.")
            
            # Example usage
            cross_series_df = fetch_with_retry(processor.fetch_data, ids, cross_formulas, display_names)
            print("cross:\n", cross_series_df)
            
            time_series_df = fetch_with_retry(processor.fetch_time_series_data, ids, timeSeries_formulas, display_names)
            print("time:\n", time_series_df)
            
            # Ensure data is valid before merging
            if not cross_series_df.empty and not time_series_df.empty:
                merged_df = pd.merge(
                    cross_series_df,
                    time_series_df,
                    on="requestId",
                    suffixes=("_df1", "_df2"),
                )
                print("merged_df:\n", merged_df)
            else:
                print("One or both dataframes are empty. Unable to merge.")

            cross_series_df["EODPrice"].fillna(merged_df["EODPrice_df2"], inplace=True)
            # cross_series_df['Ticker'].fillna(merged_df['Ticker_df2'], inplace=True)
            print("priceeeeeeeeeeeeeeeeeeeee:\n", cross_series_df)
            if cross_series_df is not None:
                # Do something with the results_df
                pass
            df2_aggregated = cross_series_df.groupby('requestId', as_index=False).agg({'EODPrice': 'first'})
            client_df = client_df.merge(
                df2_aggregated[["requestId", "EODPrice"]],
                left_on=cusip_column,
                right_on="requestId",
                how="left",
            )
            print("merge\n", client_df)

        # Calculate the market value column based on the given price and quantity
        if market_value_column is not None:
            client_df[quantity_column] = client_df.apply(
                lambda x: x[quantity_column]
                if round(x[quantity_column] * x["EODPrice"], 2)
                == round(x[market_value_column], 2)
                or round(x[quantity_column] * 100 * x["EODPrice"], 2)
                != round(x[market_value_column], 2)
                else x[quantity_column] * 100,
                axis=1,
            )
        if cusip_column is not None:
            client_df = client_df.rename({cusip_column: cusip_column.lower()}, axis=1)
            cusip_column = cusip_column.lower()
        if desc_column is not None:
            # desc_col_values = client_df[desc_column]

            # client_df['PUT/CALL'] = client_df.apply(lambda x: "PUT" if ("put" in x[desc_column].lower()) or (
            #         re.findall("p[0-9]", x[desc_column].lower()) and
            #         re.findall("p[0-9]", x[desc_column].lower())[0] in x[desc_column].lower()) else
            # ("CALL" if ("call" in x[desc_column].lower()) or (re.findall("c[0-9]", x[desc_column].lower()) and
            #                                                   re.findall("c[0-9]", x[desc_column].lower())[0] in x[
            #                                                       desc_column].lower()) else None), axis=1)
            #
            client_df["PUT/CALL"] = client_df.apply(
                lambda x: (
                    "PUT"
                    if x[desc_column] is not None
                    and x[desc_column] is not np.nan
                    and (
                        "put" in x[desc_column].lower()
                        or (
                            re.findall("p[0-9]", x[desc_column].lower())
                            and re.findall("p[0-9]", x[desc_column].lower())[0]
                            in x[desc_column].lower()
                        )
                    )
                    else (
                        "CALL"
                        if x[desc_column] is not None
                        and x[desc_column] is not np.nan
                        and (
                            "call" in x[desc_column].lower()
                            or (
                                re.findall("c[0-9]", x[desc_column].lower())
                                and re.findall("c[0-9]", x[desc_column].lower())[0]
                                in x[desc_column].lower()
                            )
                        )
                        else None
                    )
                )
                if x[desc_column] is not None and x[desc_column] is not np.nan
                else None,
                axis=1,
            )
            print("client-df\n", client_df)
            client_df["PUT/CALL"].fillna(" ", inplace=True)
            print("client-----------------df", client_df)
            # Group by 'cusip_column' and 'PUT/CALL', sum 'quantity_column', and reset index
            client_df = (
                client_df.groupby([cusip_column, "PUT/CALL"])[quantity_column]
                .sum()
                .reset_index()
            )
            # Optionally, you can replace 'NoneValue' back to None after the groupby operation if needed.
            client_df["PUT/CALL"].replace("No Value", np.nan, inplace=True)
            print("client-----------------df", client_df)
            desc_col_values = client_df["PUT/CALL"]
            client_df.drop(["PUT/CALL"], axis=1, inplace=True)
        elif cusip_column and quantity_column is not None:
            client_df = (
                client_df.groupby([cusip_column])[quantity_column].sum().reset_index()
            )
        st.write(
            f"Preview of UNIQUE CUSIPs and Quantity(shares) ({n_preview} rows only):"
        )
        st.dataframe(client_df.head(n_preview))
        st.caption(
            f":green[{len(client_df)}] data rows and :green[{len(client_df.columns)}] columns were loaded"
        )
    st.write()
    st.divider()

    # # AsOn Date Picker...
    # default_as_on_date = datetime(2024, 3, 31)
    # selected_date = st.date_input("Select a \"As Of\" date to retrieve Price data via the FactSet API (Usually this is End of Quarter date)", default_as_on_date)
    # as_on_date = selected_date.strftime("%m/%d/%Y")
    # print(f"User selected as_on date: {as_on_date}")
    # Call the function to get the last date of the previous quarter

    # Calculate the last day of the previous quarter
    last_date_str = last_date_of_previous_quarter()
    # Parse the string into year, month, and day
    year, month, day = map(int, last_date_str.split("-"))

    # Create a datetime object
    default_as_on_date = datetime(year, month, day)
    # Streamlit date input
    selected_date = st.date_input(
        "Select an 'As Of' date to retrieve Price data via the FactSet API (Usually this is End of Quarter date)",
        default_as_on_date,
    )
    as_on_date = selected_date.strftime("%m/%d/%Y")
    st.write(f"User selected as_on date: {as_on_date}")

    # Map button
    if st.button("Generate Mappings"):
        if client_data_file is not None:
            if cusip_column == quantity_column:
                st.error(
                    f'The "Identifier" and "Quantity" columns must be distinct. Please choose the correct mappings for each column.'
                )
            else:
                is_clean, msg = check_client_df_sanity(
                    client_df, cusip_col=cusip_column, quantity_col=quantity_column
                )
                if not is_clean:
                    st.error(msg)
                else:
                    with st.spinner("Mapping Client data with FactSet API..."):
                        handle_cusips(
                            client_df, cusip_column, quantity_column, as_on_date
                        )
                        st.success(
                            'Please switch to the "Working Sheet" tab to examine the mappings.'
                        )
        else:
            st.error(
                "Kindly upload client data file before pressing `Generate Mappings`"
            )


with tab3:  # '../resources/WorkingSheetTemplate.xlsx'

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
        AgGrid(
            df,
            gridOptions=grid_options,
            fit_columns_on_grid_load=True,
            update_mode="MODEL_CHANGED",
        )

    header_details = [
        "Ticker (Client)",
        "CUSIP (Client)",
        "CUSIP (SEC)",
        "FIGI",
        "Issuer Name (N/A)",
        "Issuer Name (SEC)",
        "Type",
        "Class (SEC)",
        "SH/PRN",
        "PUT/CALL",
        "Quantity",
        "Price",
        "Market Value (Quantity*Price)",
        "De Minimis?",
        "Discretion Type",
        "Other Managers",
        "Voting Authority Sole",
        "Voting Authority Shared",
        "Voting Authority None",
        "Complete?",
    ]

    st.caption(":blue[Generated Working Sheet]")
    # Check if the updated DataFrame is available in the session state
    if "working_sheet_df" in st.session_state:
        # display_ag_grid_in_tab(st.session_state['working_sheet_df'])
        display_stdataframe(st.session_state["working_sheet_df"])
        # quantum_join_summary = st.session_state['working_sheet_df']['Ticker (Client)'].count()
        # sec_13f_join_summary = st.session_state['working_sheet_df']['CUSIP (SEC)'].count()
        # eod_data_join_summary = st.session_state['working_sheet_df']['Price'].count()
        # st.success(f"\"QuantumOnline\" found :blue[{quantum_join_summary}/{len(st.session_state['working_sheet_df'])}] Tickers. \"SEC 13F Data\" found :blue[{sec_13f_join_summary}/{len(st.session_state['working_sheet_df'])}] CUSIP matches. \"EOD Data\" found :blue[{eod_data_join_summary}/{len(st.session_state['working_sheet_df'])}] EOD Prices")
        # st.success(f"\"EOD Data\" found :blue[{eod_data_join_summary}/{len(st.session_state['working_sheet_df'])}] EOD Prices")

    else:
        # Display with an empty/default DataFrame if no update has been made
        working_sheet_df = pd.DataFrame(columns=header_details)
        st.session_state["working_sheet_df"] = working_sheet_df
        display_ag_grid_in_tab(working_sheet_df)

    st.write(
        "The generated worksheet is available for download to review and address any incomplete data."
    )
    if st.button("Create Download Link"):
        ws_df_excel = convert_ws_to_excel(
            st.session_state["working_sheet_df"]
        )  # Convert DataFrame to Excel
        filename = client_data_file_name + "_WorkingSheet Sec13F.xlsx"
        st.download_button(
            label="📥 Download Working Sheet Excel",
            data=ws_df_excel,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


with tab4:
    uploaded_ws_file = st.file_uploader(
        ":blue[Upload Working sheet to generate final SEC 13F Report]",
        type=["xls", "xlsx"],
        key="ws_revised",
    )
    if uploaded_ws_file is not None:
        client_data_file_name = uploaded_ws_file.name
        if "_WorkingSheet Sec13F" in client_data_file_name:
            client_data_file_name = client_data_file_name.split("_WorkingSheet Sec13F")[0]
        else:
            client_data_file_name = client_data_file_name.split(".xlsx")[0]
        uploaded_ws_df = pd.read_excel(uploaded_ws_file)
        cusips_with_put_call_yes = uploaded_ws_df[uploaded_ws_df['PUT/CALL'].isin(['PUT', 'CALL']) & (uploaded_ws_df['De Minimis?'] == 'Yes')]
        cusips_with_put_call_No = uploaded_ws_df[uploaded_ws_df['PUT/CALL'].isin(['PUT', 'CALL']) & (uploaded_ws_df['De Minimis?'] == 'No')]
        cusips_all_no_not_put_call = uploaded_ws_df[~uploaded_ws_df['PUT/CALL'].isin(['PUT', 'CALL']) & (uploaded_ws_df['De Minimis?'] == 'No')]
        filtered_df1 = cusips_with_put_call_yes[cusips_with_put_call_yes['CUSIP (Client)'].isin(cusips_all_no_not_put_call['CUSIP (Client)'])]
        result_df_matched = pd.concat([cusips_all_no_not_put_call, filtered_df1], ignore_index=True)
        uploaded_ws_df = pd.concat([result_df_matched, cusips_with_put_call_No], ignore_index=True)
        uploaded_ws_df = uploaded_ws_df.sort_values(by='Issuer Name (SEC)')
        is_correct, msg = validate_ws_cols(uploaded_ws_df)
        if is_correct:
            sec_13f_excel_revised = generate_13f_from_ws(uploaded_ws_df,client_data_file_name)
            filename = client_data_file_name + "_Sec_13F_report.xlsx"
            st.download_button(
                "Download SEC-13F Report (Excel)",
                data=sec_13f_excel_revised,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error(msg)

    st.divider()

    # st.divider()
    # st.markdown("I intend to create the SEC 13F Excel Report using the aforementioned Working Sheet.")

    # # st.json(st.session_state)
    # if len(st.session_state['working_sheet_df']) > 0:
    #     sec_13f_excel = generate_13f_from_ws(st.session_state['working_sheet_df'])
    #     st.download_button('Download SEC-13F Report (Excel)',
    #                       data=sec_13f_excel,
    #                       file_name="sec_13f_report.xlsx",
    #                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # st.divider()
    # st.markdown("I've updated the Working Sheet with the necessary information and wish to upload the revised version for generating the SEC 13F Excel Report.")

    # uploaded_ws_file = st.file_uploader(":blue[Upload revised version of Working sheet]", type=['csv'], key="ws_revised")
    # if uploaded_ws_file is not None:
    #     uploaded_ws_df = pd.read_csv(uploaded_ws_file)
    #     sec_13f_excel_revised = generate_13f_from_ws(uploaded_ws_df)

    #     st.download_button('Download SEC-13F Report (Excel) with Revised Working Sheet',
    #                       data=sec_13f_excel_revised,
    #                       file_name="sec_13f_report.xlsx",
    #                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
