import boto3
import pandas as pd
import configparser
from datetime import datetime
from enum import Enum, auto

class DatasetType(Enum):
    DS_SEC13F = auto()
    DS_EOD = auto()
    DS_CUSIP_TICKER_MAP = auto()

class S3DatasetManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.config['s3']['aws_access_key_id'],
            aws_secret_access_key=self.config['s3']['aws_secret_access_key'],
            region_name=self.config['s3']['region_name']
        )
        self.bucket_name = self.config['s3']['bucket_name']
    
    def list_dataset_versions(self, dataset_type):
        prefix = f'{dataset_type.name}/'
        versions_list = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        version_keys = [version['Key'] for version in versions_list.get('Contents', [])]
        return version_keys

    def get_row_count(self, key):
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        df = pd.read_csv(response['Body'])
        return len(df.index)
    
    def get_last_updated_timestamp(self, dataset_type):
        prefix = f'{dataset_type.name}/'
        versions = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        for version in versions.get('Contents', []):
            print(f"{version['Key']}: Last Modified on {version['LastModified']}")
    
    def fetch_dataset(self, key):
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        df = pd.read_csv(response['Body'])
        return df
    
    def upload_dataset(self, dataset_type, file_path):
        file_name = datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
        key = f"{dataset_type.name}/{file_name}"
        self.s3_client.upload_file(file_path, self.bucket_name, key)
        print(f"Uploaded {file_path} as {key}")


if __name__ == '__main__':
    # Example usage:
    manager = S3DatasetManager()

    # Upload a Dataset-Type version
    # manager.upload_dataset(DatasetType.DS_SEC13F, '../../resources/SEC-13F_FY2023_Q4.csv')
    # manager.upload_dataset(DatasetType.DS_EOD, '../../resources/EODData_20231229.csv')

    # List all versions of a dataset type
    # versions = manager.list_dataset_versions(DatasetType.DS_EOD)
    # for each in versions:
    #     print(each)

    # Show row count of a selected Dataset-version
    # for each in versions:
    #     print(manager.get_row_count(each))

    # Show last updated timestamp of each Dataset-version copy
    manager.get_last_updated_timestamp(DatasetType.DS_SEC13F)
    manager.get_last_updated_timestamp(DatasetType.DS_EOD)

    # Fetch a Dataset-Type-version copy
    # for each in versions:
    #     df = manager.fetch_dataset(each)
    #     print(df.head())
