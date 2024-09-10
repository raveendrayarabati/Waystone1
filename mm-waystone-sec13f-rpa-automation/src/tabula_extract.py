import pandas as pd
from tabula import read_pdf

class SEC13FLoader:
    def __init__(self) -> None:
        self.pdf_path = '../resources/SEC_13F_Securities_FY2023Q4.pdf'
        self.out_path = '../resources/SEC-13F_FY2023_Q4.csv.csv'

    def extract_dataframe(self):
        # Extract tables without treating the first row as headers
        tables = read_pdf(self.pdf_path, pages='all', multiple_tables=True, pandas_options={'header': None})

        tables_list = []
        for each in tables:
            # Check if the table has 5 columns, assign headers directly
            if each.shape[1] == 5:
                each.columns = ["CUSIP NO", "ASTRK", "ISSUER NAME", "ISSUER DESCRIPTION", "STATUS"]
            elif each.shape[1] == 4:
                # For tables with 4 columns, add the missing 'STATUS' column
                each.columns = ["CUSIP NO", "ASTRK", "ISSUER NAME", "ISSUER DESCRIPTION"]
                each['STATUS'] = pd.NA  # Use pd.NA for missing values in 'STATUS'
            else:
                print(f"unmatched table shape = {each.shape}")
                print(each)
                # Skip tables that don't match the expected structure (optional)
                continue
            tables_list.append(each)

        # Concatenate all the tables
        consolidated_df = pd.concat(tables_list, ignore_index=True)
        
        # Save the consolidated dataframe to a CSV file
        consolidated_df.to_csv(self.out_path, index=False)
        return consolidated_df

if __name__ == '__main__':
    loader = SEC13FLoader()
    consolidated_dataframe = loader.extract_dataframe()
