import os
import requests
import numpy as np
import pandas as pd
from lxml import html
import concurrent.futures

def fetch_ticker_from_cusip(cusip):
    """
    Fetches the ticker symbol for a given CUSIP number from QuantumOnline.

    Parameters:
    cusip (str): The CUSIP number for which the ticker symbol is required.

    Returns:
    str: The ticker symbol associated with the given CUSIP number, or None if not found or an error occurs.
    """
    # Construct the search URL
    url = f"http://www.quantumonline.com/search.cfm?tickersymbol={cusip}&sopt=cusip"
    
    try:
        # Attempt to fetch the page
        response = requests.get(url, timeout=10)  # Added timeout for the request
        # Check if the request was successful
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve page: {e}")
        return None

    try:
        # Parse the HTML content
        tree = html.fromstring(response.content)
        # Define the XPath to locate the ticker symbol text
        xpath = "//center[b[contains(text(), 'Ticker Symbol:')]]"
        # Attempt to extract the ticker symbol using the XPath
        ticker_element = tree.xpath(xpath)
        if ticker_element:
            ticker_symbol = ticker_element[0].text_content().split("CUSIP:")[0].split(":")[-1].strip()
            return ticker_symbol
        else:
            # Handle case where the ticker symbol is not found in the page
            print("Ticker symbol not found in the page.")
            return None
    except Exception as e:
        # Handle unexpected parsing errors
        print(f"Error extracting ticker symbol: {e}")
        return None

def fetch_ticker_from_cusip_parallel(cusip):
    """
    Fetches the ticker symbol for a given CUSIP number from QuantumOnline.
    """
    url = f"http://www.quantumonline.com/search.cfm?tickersymbol={cusip}&sopt=cusip"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve page: {e}")
        return None

    try:
        tree = html.fromstring(response.content)
        xpath = "//center[b[contains(text(), 'Ticker Symbol:')]]"
        ticker_element = tree.xpath(xpath)
        if ticker_element:
            ticker_symbol = ticker_element[0].text_content().split("CUSIP:")[0].split(":")[-1].strip()
            return ticker_symbol
        else:
            print("Ticker symbol not found in the page.")
            return None
    except Exception as e:
        print(f"Error extracting ticker symbol: {e}")
        return None

def parallel_fetch_tickers(client_df, cusip_col):
    """
    Fetches ticker symbols for a list of CUSIP numbers in parallel.
    
    Parameters:
    cusips (list): A list of CUSIP numbers to fetch ticker symbols for.
    
    Returns:
    A list of ticker symbols (or None for failures).
    """
    cusips = client_df[cusip_col].tolist()
    # Determine the number of workers to use; we use N-1 cores, ensuring at least 1 core is free.
    num_workers = max(1, os.cpu_count() - 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map fetch_ticker_from_cusip across the CUSIP numbers in parallel
        results = list(executor.map(fetch_ticker_from_cusip, cusips))
    
    client_ticker_mapped_df = client_df.copy()
    client_ticker_mapped_df['Ticker (Client)'] = results
    client_ticker_mapped_df = client_ticker_mapped_df.fillna(np.nan)
    return client_ticker_mapped_df

def map_sec13f(client_data_df, sec13f_df, cusip_col):
    print(client_data_df.head())
    print(sec13f_df.head())

    client_data_df['cusip_normalized'] = client_data_df[cusip_col].str.replace(' ', '')
    sec13f_df['CUSIP NO_normalized'] = sec13f_df['CUSIP NO'].str.replace(' ', '')

    # Perform the merge using the normalized CUSIP columns
    merged_df = pd.merge(client_data_df, sec13f_df, left_on='cusip_normalized', right_on='CUSIP NO_normalized', how='left')

    # Drop the normalized columns if they are no longer needed
    merged_df.drop(columns=['cusip_normalized', 'CUSIP NO_normalized'], inplace=True)
    merged_df = merged_df.rename({'CUSIP NO': 'CUSIP (SEC)', 'ISSUER NAME': 'Issuer Name (SEC)', 'ISSUER DESCRIPTION': 'Class (SEC)', 'STATUS': 'STATUS (SEC)'}, axis=1)
    merged_df = merged_df.drop(['ASTRK'], axis=1, errors='ignore')
    return merged_df

def map_eod(client_data_df, eod_df):
    client_data_df = client_data_df.replace({np.nan: None})
    eod_df = eod_df.replace({np.nan: None})
    result_df = client_data_df.merge(eod_df[['Symbol', 'Close']], left_on='Ticker (Client)', right_on='Symbol', how='left')
    result_df = result_df.rename({'Close': 'Price'}, axis=1)
    result_df['Price'] = result_df.apply(lambda x: None if x['Ticker (Client)'] == None else x['Price'], axis=1)
    return result_df

# Main block to run the function if the script is executed directly
if __name__ == '__main__':
    # # Example CUSIP to test
    # cusip = "032654105"
    # ticker = fetch_ticker_from_cusip(cusip)
    # if ticker:
    #     print(f"The ticker symbol for CUSIP {cusip} is: {ticker}")
    # else:
    #     print("Failed to find a ticker symbol for the given CUSIP.")

    # Parallel Example list of CUSIP numbers to test
    client_df = pd.read_csv('../resources/ClientCUSIPs.csv')
    # cusips = ['18453H106', '22788C105', '23804L103', 'G29018101', '26142V105', 'M6191J100', '58733R102', '60937P106', '65345M108', 'G6683N103', '707569109', '72352L106', '88160R101', '888787108', '98954M101']
    client_ticker_mapped_df = parallel_fetch_tickers(client_df, 'Cusip')
    print(client_ticker_mapped_df.shape)

    sec13f_df = pd.read_csv('src/resources/SEC-13F_FY2023_Q4.csv')
    print(sec13f_df.shape)
    R1 = map_sec13f(client_ticker_mapped_df, sec13f_df, 'Cusip')
    print(R1.head(100))
    print(R1.shape)

    eod_df = pd.read_csv('../resources/EODData_20231229.csv')
    R2 = map_eod(R1, eod_df)
    print(R2.head(100))
    print(R2.shape)
