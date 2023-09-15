from typing import Union, Optional

import boto3
from src.utils.constants import s3_bucket_name


session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net'
)


def fetch_dataset(name: str, bucket: Optional[str] = None):
    get_object_response = s3.get_object(Bucket=bucket if bucket else s3_bucket_name,
                                        Key=name)
    print(f'Fetching dataset "{name}"')
    file = get_object_response['Body'].read()

    return file
