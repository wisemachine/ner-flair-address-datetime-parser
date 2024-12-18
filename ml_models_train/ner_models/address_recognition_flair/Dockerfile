FROM --platform=linux/amd64 python:3.8-slim-buster as build

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3-distutils \
    python3-setuptools \
    python3-pip \
    && apt-get clean

# Set the working directory
WORKDIR /opt/ml/code

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Define the entry point for the container
ENTRYPOINT ["python3", "train.py"]