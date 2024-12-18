#!/bin/bash

# Set variables
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
IMAGE_NAME=flair-ner-trainer
ECR_REPOSITORY=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}

# Authenticate Docker to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY}

# Build the Docker image
docker build -t ${IMAGE_NAME} .

# Tag the Docker image
docker tag ${IMAGE_NAME}:latest ${ECR_REPOSITORY}:latest

# Push the Docker image to ECR
docker push ${ECR_REPOSITORY}:latest

# Run the SageMaker training script
python sagemaker_train.py
