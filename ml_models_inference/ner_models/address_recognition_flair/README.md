Here is the updated `README.md` file with the correct paths:

```markdown
# Building and Deploying a SageMaker Inference Endpoint with FLAIR NER Model built with AWS Sagamker and stored in AWS S3

## Overview
This directory contains scripts and configuration files to build, deploy, and test an Amazon SageMaker inference endpoint using a FLAIR NER model. The steps include setting up the SageMaker endpoint, building a Docker image, and testing the deployed model.

## Directory Contents
- **Dockerfile.inference**: Dockerfile for building the Docker image for the SageMaker inference endpoint.
- **inference.py**: The inference script that contains the logic for loading the FLAIR model and handling predictions.
- **deploy_endpoint.py**: Script to deploy the SageMaker endpoint.
- **requirements.txt**: List of Python dependencies for the inference script.
- **run_sagemaker_flair_inference.sh**: Shell script to build the Docker image, push it to Amazon ECR, and deploy the SageMaker endpoint.
  
## Support scripts to interact with and test Sagemkaer Endpoint
- **mltools/ml_models_inference/sagemaker_endpoint_client.py**: Python class for interacting with the SageMaker endpoint.
- **tests/ml_models/ner_models/address_recognition_flair/address_recognition_flair_sagemaker_endpoint_eval.py**: Script to test predictions from the SageMaker endpoint.

## Prerequisites
- AWS CLI configured with appropriate permissions.
- Docker installed and running.
- Python environment with `boto3` installed.

## Step-by-Step Instructions

### 1. Set Up Your Environment
Ensure your AWS CLI is configured:
```sh
aws configure
```

### 2. Build and Deploy the Docker Image
Run the `run_sagemaker_flair_inference.sh` script to build the Docker image, push it to Amazon ECR, and deploy the SageMaker endpoint.
```sh
chmod +x mltools/ml_models_inference/ner_models/address_recognition_flair/run_sagemaker_flair_inference.sh
./mltools/ml_models_inference/ner_models/address_recognition_flair/run_sagemaker_flair_inference.sh
```

### 3. Test the Deployed Model
Use the `address_recognition_flair_sagemaker_endpoint_eval.py` script to test the predictions from the SageMaker endpoint.
```sh
python tests/ml_models/ner_models/address_recognition_flair/address_recognition_flair_sagemaker_endpoint_eval.py
```

## File Descriptions

### Dockerfile.inference
Dockerfile for building the Docker image that will be used for the SageMaker inference endpoint. It installs necessary dependencies and copies the inference script.

### inference.py
The inference script contains the logic for loading the FLAIR NER model and handling predictions. It includes a Flask app to handle HTTP requests for model predictions.

### deploy_endpoint.py
Script to deploy the SageMaker endpoint. It sets up the necessary configurations and deploys the endpoint using the Docker image pushed to Amazon ECR.

### requirements.txt
List of Python dependencies required for the inference script. These dependencies are installed when building the Docker image.

### run_sagemaker_flair_inference.sh
Shell script to automate the process of building the Docker image, pushing it to Amazon ECR, and deploying the SageMaker endpoint. It includes steps for Docker authentication, image building, and endpoint deployment.

### sagemaker_endpoint_client.py
Python class to interact with the SageMaker endpoint. It includes methods to send input data to the endpoint and retrieve predictions.

### address_recognition_flair_sagemaker_endpoint_eval.py
Script to test the deployed SageMaker endpoint. It sends a sample input to the endpoint and prints the resulting predictions.

## Conclusion
By following these steps, you can deploy, and test a SageMaker inference endpoint for a FLAIR NER model that's stored in AWS S3. This setup allows for scalable and efficient deployment of NLP models in a production environment.
```
