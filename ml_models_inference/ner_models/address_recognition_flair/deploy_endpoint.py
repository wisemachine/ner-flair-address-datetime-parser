import os
import boto3
import sagemaker
from sagemaker.model import Model
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from a .env file if present

def main():
    # If you want to dynamically get the account and region from your AWS CLI / environment:
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.Session().region_name

    # Read from env or supply sensible defaults
    # e.g. "123456789012.dkr.ecr.us-east-1.amazonaws.com/flair-ner-inference:latest"
    flair_image_uri = os.getenv("FLAIR_IMAGE_URI", f"{account_id}.dkr.ecr.{region}.amazonaws.com/flair-ner-inference:latest")

    # Dummy account id below - e.g. "arn:aws:iam::123456789012:role/service-role/YourSageMakerRole"
    role = os.getenv("SAGEMAKER_EXECUTION_ROLE", "arn:aws:iam::123456789012:role/YourSageMakerRole")

    # If your model artifacts are in S3, specify the bucket/key via env
    s3_bucket = os.getenv("S3_BUCKET", "your-bucket")
    s3_key = os.getenv("S3_KEY", "flair_address_parsing/output/flair-ner-training/model.tar.gz")

    # e.g. "flair-ner-endpoint"
    endpoint_name = os.getenv("ENDPOINT_NAME", "flair-ner-endpoint")

    # Create a SageMaker Model
    model = Model(
        image_uri=flair_image_uri,
        role=role,
        env={
            'S3_BUCKET': s3_bucket,
            'S3_KEY': s3_key
        }
    )

    # Deploy the Model
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',  # can be parameterized via env if desired
        endpoint_name=endpoint_name
    )
    print(f"Model deployed. Endpoint name: {endpoint_name}")

if __name__ == "__main__":
    main()
