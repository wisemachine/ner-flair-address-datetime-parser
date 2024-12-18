import boto3
import tarfile
import os
from botocore.exceptions import ClientError
from flair.models import SequenceTagger
from flair.data import Sentence

class FlairMLModelInference:
    def __init__(self, bucket_name, object_key, download_path, extract_to):
        self.bucket_name = bucket_name
        self.object_key = object_key
        self.download_path = download_path
        self.extract_to = extract_to

        # Ensure the extract_to directory exists, but do not recreate it if it exists
        if not os.path.exists(self.extract_to):
            os.makedirs(self.extract_to)

    def list_objects_in_bucket(self, prefix=''):
        s3 = boto3.client('s3')
        try:
            response = s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' in response:
                for obj in response['Contents']:
                    print(obj['Key'])
            else:
                print(f"No objects found in bucket {self.bucket_name} with prefix {prefix}")
        except ClientError as e:
            print(f"Error listing objects in bucket: {e}")

    def download_model_from_s3(self):
        s3 = boto3.client('s3')
        try:
            if os.path.exists(self.download_path):
                os.remove(self.download_path)
            s3.download_file(self.bucket_name, self.object_key, self.download_path)
            print(f"Downloaded {self.object_key} to {self.download_path}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"The object {self.object_key} does not exist in bucket {self.bucket_name}")
            else:
                print(f"Error downloading object: {e}")

    def extract_tar_gz(self):
        with tarfile.open(self.download_path, 'r:gz') as tar:
            tar.extractall(path=self.extract_to)

    def load_model(self):
        model_path = os.path.join(self.extract_to, 'best-model.pt')
        self.model = SequenceTagger.load(model_path)

    def predict(self, text):
        sentence = Sentence(text)
        self.model.predict(sentence)
        results = [(entity.text, label.value, label.score) for entity in sentence.get_spans('ner') for label in entity.labels]
        return results
