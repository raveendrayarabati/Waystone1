import os
import zipfile
import pandas as pd
import requests
import streamlit as st
import io


def extract_zip_files(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def read_excel_files(folder):
    data_frames = []
    for item in os.listdir(folder):
        if item.endswith('.xlsx'):
            file_path = os.path.abspath(os.path.join(folder, item))
            df = pd.read_excel(file_path)
            data_frames.append((file_path, df))
    return data_frames


def find_missing_cusips(data_frames):
    missing_cusip_dfs = []
    for file_path, df in data_frames:
        missing_cusip_df = df[df['CUSIP'].isnull()]
        if not missing_cusip_df.empty:
            missing_cusip_dfs.append((file_path, missing_cusip_df))
    return missing_cusip_dfs


def fetch_cusip(symbol):
    url =f"http://www.quantumonline.com/search.cfm?tickersymbol={cusip}&sopt=cusip"  # Replace with actual API endpoint
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('cusip')
    return None


def update_missing_cusips(data_frames):
    for file_path, df in data_frames:
        for index, row in df.iterrows():
            if pd.isnull(row['CUSIP']):
                cusip = fetch_cusip(row['Symbol'])
                if cusip:
                    df.at[index, 'CUSIP'] = cusip


def save_updated_files(data_frames):
    for file_path, df in data_frames:
        csv_file_path = file_path.replace('.xlsx', '.csv')
        df.to_csv(csv_file_path, index=False)
        st.download_button(
            label="Download updated file",
            data=open(csv_file_path, "rb").read(),
            file_name=os.path.basename(csv_file_path),
            mime="text/csv"
        )


def main():
    st.title("13F Filings CUSIP Updater")

    zip_file = st.file_uploader("Upload a ZIP file containing Excel files", type="zip")

    if zip_file:
        extract_to = "extracted_files"
        os.makedirs(extract_to, exist_ok=True)

        with open("uploaded_zip.zip", "wb") as f:
            f.write(zip_file.getbuffer())

        extract_zip_files("uploaded_zip.zip", extract_to)
        st.success("ZIP file extracted successfully")

        excel_files = read_excel_files(extract_to)

        missing_cusip_data = find_missing_cusips(excel_files)

        if missing_cusip_data:
            update_missing_cusips(missing_cusip_data)
            save_updated_files(excel_files)
            st.success("CUSIP values updated successfully")

            for file_path, _ in excel_files:
                st.write(f"Updated file: {file_path.replace('.xlsx', '.csv')}")
        else:
            st.info("No missing CUSIP values found")


if __name__ == "__main__":
    main()
