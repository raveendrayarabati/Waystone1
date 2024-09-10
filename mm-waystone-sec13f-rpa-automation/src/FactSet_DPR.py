
import pandas as pd
import time
from dateutil.parser import parse as dateutil_parser
from pprint import pprint

# SDK Packages
import fds.sdk.Formula # To install, please use "pip install fds.sdk.Formula==2.2.1"
from fds.sdk.Formula.apis import *
from fds.sdk.Formula.models import *
from fds.sdk.utils.authentication import ConfidentialClient # To install, please use "pip install fds.sdk.utils"



# Authentication using OAuth
configuration = fds.sdk.Formula.Configuration(
     fds_oauth_client=ConfidentialClient('./config/app-config.json') # Path to config file from OAuth registration
)


# Batch Processing Workflow
with fds.sdk.Formula.ApiClient(configuration) as api_client:
    # Create Instance
    api_instance = CrossSectionalApi(api_client)

    # Request Object to Define Parameters
    batch_request = CrossSectionalRequest(
        data=CrossSectionalRequestData(
            ids= ["252131107", "38141G104", "457669AB5", "501889208", "61174X109", "759916AC3", "V7780T103", "AAPL"],
            formulas = ["FF_CUSIP(CURR)", "P_PRICE(NOW)", "FSYM_TICKER_EXCHANGE", "P_EXCOUNTRY"],
            displayName = ["CUSIP", "EODPrice", "Ticker", "Country"],
            fsymId = "Y",
            batch = "Y",
            flatten = 'Y'
        ),
    )

    # Send Request            
    try:
        api_response_wrapper = api_instance.get_cross_sectional_data_for_list(batch_request)
        api_response = api_response_wrapper.get_response_202()
        id = api_response["data"]["id"] # Batch Request Id

        # Request Results Using Batch Id
        api_instance = BatchProcessingApi(api_client)
        processing = True

        while processing:
            # Check Request Status
            api_response_wrapper = api_instance.get_batch_status(id)
            #
            if api_response_wrapper["data"]["status"] == "DONE":
                # Get Data from Successful Request  
                batch_data_request = BatchDataRequest(
                    data=BatchDataRequestData(
                        id = id,
                    ),
                )
                api_response_wrapper = api_instance.get_batch_data_with_post(batch_data_request,_check_return_type=False)
                api_response = api_response_wrapper.get_response_200()

                # Convert to Pandas Dataframe
                results = pd.DataFrame(api_response.to_dict()['data'])
                pprint(results)

                processing = False
            elif api_response_wrapper["data"]["status"] == "EXECUTING":
                print("Processing Batch...")
                time.sleep(10) # Wait Before Checking Batch Status Again
            else:
                print("Something went wrong.")
                print(api_response_wrapper["data"]["status"] + ": "+ api_response_wrapper["data"]["error"])
                break

    except fds.sdk.Formula.ApiException as e:
        print(str(e).splitlines()[1])
