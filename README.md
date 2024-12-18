Below is a **comprehensive README** that reflects the **current structure** of your repository. It highlights the **two main parts** (Training vs. Inference) as well as the **two standalone deterministic parsers** (Address & Datetime). It also includes usage notes and context around each file.

---

# NER Flair Address & Datetime Parser

This repository provides two main capabilities:

1. **Train and deploy a Flair-based NER model on SageMaker** for address recognition.  
2. **Standalone deterministic parsers** for addresses and datetime, completely independent of the Flair model.

Below is a **detailed breakdown** of the structure, scripts, and their usage.

---

## Repository Structure

```plaintext
ner-flair-address-datetime-parser/
├── ml_models_train/
│   └── ner_models/
│       └── address_recognition_flair/
│           ├── Dockerfile                     <-- Docker image for training
│           ├── requirements.txt              <-- Dependencies for training
│           ├── run_sagemaker_flair_training.sh <-- Shell script to build/push image & start SageMaker training
│           ├── sagemaker_train.py            <-- SageMaker Estimator script for training the Flair model
│           └── train.py                      <-- Local Flair training script
│
├── ml_models_inference/
│   ├── flair_ml_model_inference.py           <-- Generic Flair inference helper (loads model from S3, etc.)
│   ├── sagemaker_endpoint_client.py          <-- Client script to invoke the deployed SageMaker endpoint
│   └── ner_models/
│       └── address_recognition_flair/
│           ├── Dockerfile.inference          <-- Docker image for inference
│           ├── deploy_endpoint.py            <-- Script to create a SageMaker model/endpoint from ECR & model artifacts
│           ├── inference.py                  <-- Main Flask or API inference script (runs predictions)
│           ├── requirements.txt             <-- Dependencies for inference
│           └── run_sagemaker_flair_inference.sh <-- Shell script to build/push inference image & deploy on SageMaker
│
├── address_parser.py                         <-- Deterministic address parser (usaddress, pyap)
├── date_time_parser.py                       <-- Deterministic date/time parser (datefinder, dateutil)
├── README.md                                 <-- This common repository README
└── (other possible .env, Dockerfiles, etc.)
```

### 1. **Training Part** (`ml_models_train/ner_models/address_recognition_flair/`)

This folder houses **all training-related** components for the **Flair NER model** that recognizes address entities.

- **`Dockerfile`**  
  A Dockerfile used to create the **training container**. This container is typically pushed to ECR and used in SageMaker jobs.

- **`requirements.txt`**  
  Python libraries needed inside the training container (e.g., `flair`, `torch`, etc.).

- **`run_sagemaker_flair_training.sh`**  
  A shell script that might **build** the Docker image, **push** it to ECR, and **initiate** the SageMaker training job using `sagemaker_train.py`.  
  - Typically, you'd run: `bash run_sagemaker_flair_training.sh`

- **`sagemaker_train.py`**  
  A **SageMaker Estimator** script that sets up and runs a training job on AWS SageMaker, using your ECR Docker image.  
  - Reads environment variables or `.env` for S3 paths, roles, instance types, etc.  
  - **Output** model artifacts end up in S3.

- **`train.py`**  
  A **local training script** using Flair. You can run this script **locally** (without SageMaker) to train or finetune the NER model.  
  - It downloads or reads training data, sets up a Flair `SequenceTagger`, trains, and saves the model artifacts (e.g., `best-model.pt`).

---

### 2. **Inference Part** (`ml_models_inference/ner_models/address_recognition_flair/`)

This folder contains **inference-related** scripts and containers for the trained Flair NER model.

- **`Dockerfile.inference`**  
  A Dockerfile for **inference**. It typically installs `flair`, `torch`, and the code needed to load a trained model (`inference.py` or `flair_ml_model_inference.py`).  
  - Deployed on SageMaker or run locally as a container.

- **`deploy_endpoint.py`**  
  Creates a **SageMaker model and endpoint** from the ECR inference image & S3 model artifacts.  
  - This script might configure environment variables for your endpoint name, instance type, etc.  
  - After a successful run, a new SageMaker endpoint is live and can handle real-time predictions.

- **`inference.py`**  
  The main **inference logic** or **Flask app** code. Typically:
  - Defines API endpoints like `/ping` (health) and `/invocations` (prediction).
  - Loads the Flair model via `flair_ml_model_inference.py` or directly from S3.  
  - Processes requests, runs NER predictions, and returns structured JSON responses.

- **`requirements.txt`**  
  Dependencies needed during inference (e.g., `flair`, `torch`, `flask`, etc.).

- **`run_sagemaker_flair_inference.sh`**  
  Shell script that might build/push the **inference Docker image** to ECR and run `deploy_endpoint.py` to spin up a SageMaker endpoint.

- **`flair_ml_model_inference.py`**  
  A **generic helper** that handles **downloading** the `model.tar.gz` from S3, **extracting** the Flair model, and providing a `predict()` method for NER.  
  - Potentially invoked by `inference.py`.

- **`sagemaker_endpoint_client.py`**  
  A **client script** to invoke or test your deployed SageMaker endpoint.  
  - Calls the endpoint (usually via `boto3`) to submit inference payloads and parse responses.

---

### 3. **Deterministic Parsers**

At the root level, there are **two standalone scripts** that do not depend on Flair or SageMaker. They provide a purely **regex/library-based** approach to parsing addresses and datetimes.

1. **`address_parser.py`**  
   - Uses **usaddress** and **pyap** to parse address strings (e.g., "123 Main St, Springfield, IL 62704") into structured components (street, city, state_code, postal_code, etc.).  
   - Supports fallback logic, placeholders, and open-source–friendly code.  
   - Can run without SageMaker or any ML model.

2. **`date_time_parser.py`**  
   - A **deterministic** date/time parser using `datefinder`, `dateutil`, and regex to handle unstructured datetime strings.  
   - E.g., "We meet on 07/04 at 3 PM" => parsed `datetime` objects or dictionary format.

These scripts are **independent** from the Flair NER system. You can use them **alongside** or **separately**.

---

## Usage Overview

### 1. **Flair NER Training**

**Option A: Local Training**  
- Navigate to `ml_models_train/ner_models/address_recognition_flair/`.  
- Ensure environment variables or `.env` is set for S3, etc.  
- Run:
  ```bash
  python train.py
  ```
  This trains the Flair model locally using your data (train/test files).

**Option B: SageMaker Training**  
- Build & push Docker image using `Dockerfile`, then run:
  ```bash
  bash run_sagemaker_flair_training.sh
  ```
  This typically calls `sagemaker_train.py`, which creates a SageMaker training job.

---

### 2. **Flair NER Inference**

**Option A: Local Flask Inference**  
- Navigate to `ml_models_inference/ner_models/address_recognition_flair/`.  
- You can run `inference.py` directly if it sets up a Flask server:
  ```bash
  python inference.py
  ```
  Then POST text to the local endpoint.

**Option B: Deploy on SageMaker**  
- Build & push the inference Docker image using `Dockerfile.inference`.
- Run:
  ```bash
  bash run_sagemaker_flair_inference.sh
  ```
  This calls `deploy_endpoint.py` to create a SageMaker endpoint for real-time inference.

**Testing the Endpoint**  
- Use **`sagemaker_endpoint_client.py`** to invoke or send data to the newly created endpoint.

---

### 3. **Deterministic Address & Datetime Parsers**

**Usage Example**:

```python
# Deterministic address parsing
from address_parser import AddressParser

addr_parser = AddressParser()
parsed_address = addr_parser.parse_address("123 Main St, Springfield, IL 62704")
print(parsed_address)  # Structured output

# Deterministic datetime parsing
from date_time_parser import DateTimeParser

dt_parser = DateTimeParser()
parsed_dates = dt_parser.parse_datetime("Meeting on 07/15/2024 at 2 PM")
print(parsed_dates)
```

These are **purely regex/library** approaches with no ML.

---

**With this structure**:

- **`ml_models_train`** contains the code & Docker resources for training your Flair address model (local or SageMaker).  
- **`ml_models_inference`** contains the code & Docker resources for deploying/invoking the trained Flair model (local or SageMaker).  
- **`address_parser.py`** and **`date_time_parser.py`** are independent scripts for rule-based parsing.

---

**Happy parsing with Flair and deterministic approaches!**
