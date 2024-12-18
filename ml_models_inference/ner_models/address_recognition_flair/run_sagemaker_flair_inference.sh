#!/bin/bash

# Set variables
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
IMAGE_NAME=flair-ner-inference
ECR_REPOSITORY=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}

# Create the ECR repository if it does not exist
aws ecr describe-repositories --repository-names ${IMAGE_NAME} > /dev/null 2>&1

if [ $? -ne 0 ]; then
  aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION}
fi

# Authenticate Docker to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY}

# Build the Docker image for inference
docker build -t ${IMAGE_NAME} -f Dockerfile.inference .

# Tag the Docker image
docker tag ${IMAGE_NAME}:latest ${ECR_REPOSITORY}:latest

# Push the Docker image to ECR
docker push ${ECR_REPOSITORY}:latest

# Deploy the endpoint
python deploy_endpoint.py
