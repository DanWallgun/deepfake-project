import os
import abc
import io

import boto3


class Storage(abc.ABC):
    @abc.abstractmethod
    def store_file(self, filename: str, data: bytes):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_file(self, filename: str) -> bytes:
        raise NotImplementedError()


class MemoryStorage(Storage):
    def __init__(self, root: str):
        self.root = root

    def store_file(self, filename: str, data: bytes):
        path = os.path.join(self.root, filename)
        with open(path, 'wb') as f:
            f.write(data)

    def load_file(self, filename: str) -> bytes:
        path = os.path.join(self.root, filename)
        with open(path, 'rb') as f:
            return f.read()


class S3BucketStorage(Storage):
    def __init__(self, boto3_bucket):
        self.bucket = boto3_bucket

    def store_file(self, filename: str, data: bytes):
        buf = io.BytesIO(data)
        self.bucket.upload_fileobj(Key=filename, Fileobj=buf)

    def load_file(self, filename: str) -> bytes:
        buf = io.BytesIO()
        self.bucket.download_fileobj(Key=filename, Fileobj=buf)
        return buf.getvalue()
