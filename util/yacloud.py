import boto3

s3 = boto3.resource(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net'
)
