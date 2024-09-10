#!/usr/bin/env python
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import certifi
import json
import csv

class CusipIterator:
    def __init__(self, csv_file_path, n_limit=None, offset=0):
        self.csv_file_path = csv_file_path
        self.n_limit = n_limit
        self.offset = offset
        self.current_count = 0
        self.file = open(csv_file_path, mode='r', encoding='utf-8')
        self.reader = csv.DictReader(self.file)
        # Skip rows according to offset
        for _ in range(offset):
            next(self.reader, None)  # Discard the rows until offset is reached

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_limit is not None and self.current_count >= self.n_limit:
            # If we've reached the n_limit, close the file and stop iteration
            self.file.close()
            raise StopIteration

        try:
            # Attempt to return the next "CUSIP NO" value
            cusip_no = next(self.reader)["CUSIP NO"]
            self.current_count += 1
            return cusip_no
        except StopIteration:
            # If no more data, close the file and raise StopIteration
            self.file.close()
            raise StopIteration

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def save_to_csv(data, file_path):
    if not data:
        print("No data to save.")
        return
    
    # Specify the order of columns
    headers = ['cusip', 'ticker', 'company']
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Data saved to {file_path}")

if __name__ == '__main__':
    fmp_api_key = 'piI4XSF2XhYHsyqRRYaq6MxnqBHkf6jW'

    offset = 100
    cusip_iter = CusipIterator("../../resources/SEC-13F_FY2023_Q4.csv", n_limit=25, offset=offset)

    all_data = []
    for cusip_no in cusip_iter:
        print(cusip_no)
        cusip = cusip_no.replace(' ', '')
        url = (f"https://financialmodelingprep.com/api/v3/cusip/{cusip}?apikey={fmp_api_key}")

        data = get_jsonparsed_data(url)
        # Check if data is empty and handle accordingly
        
        if data:
            all_data.extend(data)
        else:
            # Create a dictionary with an empty 'ticker' and 'company' but with 'cusip_no'
            empty_data = {'cusip': cusip_no, 'ticker': '', 'company': ''}
            all_data.append(empty_data)

    save_to_csv(all_data, f"../../resources/fmp/fmp_cusip_ticker_map_{offset}.csv")
