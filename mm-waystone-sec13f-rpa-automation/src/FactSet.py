import time
import requests
import os
import pandas as pd
from pprint import pprint
from fds.sdk.utils.authentication import ConfidentialClient
from fds.sdk.Formula import Configuration, ApiClient, ApiException
from fds.sdk.Formula.apis import CrossSectionalApi, BatchProcessingApi
from fds.sdk.Formula.models import CrossSectionalRequest, CrossSectionalRequestData, BatchDataRequest, \
    BatchDataRequestData
from fds.sdk.Formula.models import TimeSeriesRequest, TimeSeriesRequestData
from fds.sdk.Formula.apis import TimeSeriesApi


class FormulaDataProcessor:
    def __init__(self, config_file='mm-waystone-sec13f-rpa-automation/src/config/appconfig.json'):
        # Check if the config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        self.config = Configuration(fds_oauth_client=ConfidentialClient(config_file))

    def check_health(self):
        # Obtain the token using ConfidentialClient from the SDK
        client_credentials = self.config.fds_oauth_client
        access_token = client_credentials.get_access_token()

        # Define the header with the access token
        headers = {
            'Authorization': f'Bearer {access_token}'
        }

        # Health check endpoint
        health_check_url = "https://api.factset.com/formula-api/health"

        # Make the GET request
        response = requests.get(health_check_url, headers=headers)

        if response.status_code == 200:
            return True, response.json()  # Health check successful
        else:
            return False, response.text  # Health check failed or other error

    def fetch_data(self, ids, formulas, display_names):
        with ApiClient(self.config) as api_client:
            api_instance = CrossSectionalApi(api_client)
            batch_request = CrossSectionalRequest(
                data=CrossSectionalRequestData(
                    ids=ids,
                    formulas=formulas,
                    displayName=display_names,
                    fsymId="Y",
                    batch="Y",
                    flatten='Y'
                ),
            )

            try:
                # Send Request and Get Batch ID
                api_response_wrapper = api_instance.get_cross_sectional_data_for_list(batch_request)
                api_response = api_response_wrapper.get_response_202()
                batch_id = api_response["data"]["id"]

                # Request Results Using Batch Id
                api_instance = BatchProcessingApi(api_client)
                return self._get_batch_results(api_instance, batch_id)
            except ApiException as e:
                print(f"An exception occurred: {str(e)}")
                return None

    def _get_batch_results(self, api_instance, batch_id):
        processing = True
        while processing:
            # Check Request Status
            api_response_wrapper = api_instance.get_batch_status(batch_id)
            if api_response_wrapper["data"]["status"] == "DONE":
                # Get Data from Successful Request
                batch_data_request = BatchDataRequest(data=BatchDataRequestData(id=batch_id))
                api_response_wrapper = api_instance.get_batch_data_with_post(batch_data_request,
                                                                             _check_return_type=False)
                api_response = api_response_wrapper.get_response_200()

                # Convert to Pandas DataFrame
                results = pd.DataFrame(api_response.to_dict()['data'])
                pprint(results)

                processing = False
                return results
            elif api_response_wrapper["data"]["status"] == "EXECUTING":
                print("Processing Batch...")
                time.sleep(10)
            else:
                print(
                    f"Error: {api_response_wrapper['data']['status']}: {api_response_wrapper['data'].get('error', '')}")
                processing = False
                return None

    def fetch_time_series_data(self, ids, formulas, display_names):
        with ApiClient(self.config) as api_client:
            api_instance = TimeSeriesApi(api_client)
            time_series_request = TimeSeriesRequest(
                data=TimeSeriesRequestData(
                    ids=ids,
                    formulas=formulas,
                    displayName=display_names,
                    fsymId="Y",
                    batch="Y",
                    flatten='Y'
                ),
            )

            try:
                # Send Request and Get Batch ID
                api_response_wrapper = api_instance.get_time_series_data_for_list(time_series_request)
                api_response = api_response_wrapper.get_response_202()
                batch_id = api_response["data"]["id"]

                # Request Results Using Batch Id
                api_instance = BatchProcessingApi(api_client)
                return self._get_batch_results(api_instance, batch_id)
            except ApiException as e:
                print(f"An exception occurred: {str(e)}")
                return None


# Example Usage:
if __name__ == "__main__":
    processor = FormulaDataProcessor()

    # Check Health
    is_healthy, health_info = processor.check_health()
    if is_healthy:
        print("Formula API Service is healthy:", health_info)
    else:
        print("There may be an issue with the Formula API Service:", health_info)

    # Make Batch requests
    ids = ["252131107", "38141G104", "457669AB5", "501889208", "61174X109", "759916AC3", "V7780T103", "AAPL"]
    formulas = ["FF_CUSIP(CURR)", "P_PRICE(NOW)", "FSYM_TICKER_EXCHANGE", "P_EXCOUNTRY"]
    display_names = ["CUSIP", "EODPrice", "Ticker", "Country"]

    results_df = processor.fetch_data(ids, formulas, display_names)
    if results_df is not None:
        # Do something with the results_df
        pass



