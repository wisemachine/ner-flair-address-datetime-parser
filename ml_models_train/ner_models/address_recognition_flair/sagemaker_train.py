import os
from dotenv import load_dotenv
import sagemaker
from sagemaker.estimator import Estimator

load_dotenv()  # load environment variables from .env if present

def create_estimator(image_uri, role, output_path):
    return Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.g4dn.xlarge',  # GPU instance
        volume_size=30,
        max_run=3600,      # 1 hour
        input_mode='File',
        output_path=output_path,
        base_job_name='flair-ner-training'
    )


def get_training_input(s3_path):
    return sagemaker.inputs.TrainingInput(
        s3_path,
        content_type='text/plain'
    )


def main():
    # Read from environment variables
    #Given here is an example dummy AWS account id
    image_uri = os.getenv("FLAIR_IMAGE_URI", "123456789012.dkr.ecr.us-east-1.amazonaws.com/flair-ner-trainer:latest")
    role = os.getenv("SAGEMAKER_EXECUTION_ROLE", "arn:aws:iam::123456789012:role/YourSageMakerRole")
    s3_training_path = os.getenv("S3_TRAINING_PATH", "s3://your-bucket/flair_address_parsing/train/")
    s3_output_path = os.getenv("S3_OUTPUT_PATH", "s3://your-bucket/flair_address_parsing/output")

    # Create estimator with no hard-coded account IDs
    estimator = create_estimator(image_uri, role, s3_output_path)

    # Create TrainingInput from S3
    train_input = get_training_input(s3_training_path)

    # Launch the training job
    estimator.fit({'training': train_input})


if __name__ == "__main__":
    main()
