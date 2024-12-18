import os
import tarfile
import boto3
from botocore.exceptions import ClientError
from flair.models import SequenceTagger
from flair.data import Sentence
from flask import Flask, request, jsonify
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env if present

# Predefined dictionaries for US states and countries
US_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts',
    'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana',
    'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

COUNTRIES = {
    'US': 'United States', 'CA': 'Canada', 'GB': 'United Kingdom', 'FR': 'France', 'DE': 'Germany',
    'JP': 'Japan', 'CN': 'China', 'IN': 'India', 'BR': 'Brazil', 'RU': 'Russia'
    # Add other countries as needed
}

# Simple data classes for address components
class Country:
    def __init__(self, name: Optional[str], code: Optional[str]):
        self.name = name or ''
        self.code = code or ''

    def __str__(self):
        return f"{self.name} ({self.code})"

class State:
    def __init__(self, name: Optional[str], code: Optional[str], country: Optional[Country]):
        self.name = name or ''
        self.code = code or ''
        self.country = country or Country(None, None)

    def __str__(self):
        return f"{self.name} ({self.code}), {self.country}"

class Locality:
    def __init__(self, name: Optional[str], code: Optional[str], postal_code: Optional[str], state: Optional[State]):
        self.name = name or ''
        self.code = code or ''
        self.postal_code = postal_code or ''
        self.state = state or State(None, None, None)

    def __str__(self):
        return f"{self.name}, {self.state}"

class Address:
    def __init__(
        self, 
        address_line_1: Optional[str],
        address_line_2: Optional[str],
        locality: Optional[Locality],
        timezone: str,
        longitude: float,
        latitude: float,
        phone_numbers: List[str],
        emails: List[str],
        ref_numbers: List[str],
        recipient: Optional[str],
        contact: Optional[str]
    ):
        self.address_line_1 = address_line_1 or ''
        self.address_line_2 = address_line_2 or ''
        self.locality = locality or Locality(None, None, None, None)
        self.timezone = timezone or 'Unknown'
        self.longitude = longitude or 0.0
        self.latitude = latitude or 0.0
        self.phone_numbers = phone_numbers or []
        self.emails = emails or []
        self.ref_numbers = ref_numbers or []
        self.recipient = recipient or ''
        self.contact = contact or ''

    def __str__(self):
        return f"{self.address_line_1}, {self.address_line_2}, {self.locality}"

class FlairMLModelInference:
    def __init__(self, bucket_name, object_key, download_path, extract_to):
        self.bucket_name = bucket_name
        self.object_key = object_key
        self.download_path = download_path
        self.extract_to = extract_to

        if not os.path.exists(self.extract_to):
            os.makedirs(self.extract_to)

        self.model = None

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
        """
        Downloads the model file from S3 to the specified download path.
        """
        s3 = boto3.client('s3')
        try:
            if os.path.exists(self.download_path):
                os.remove(self.download_path)
            s3.download_file(self.bucket_name, self.object_key, self.download_path)
            print(f"Model downloaded to {self.download_path}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"The object {self.object_key} does not exist in bucket {self.bucket_name}")
            else:
                print(f"Error downloading object: {e}")

    def extract_tar_gz(self):
        """
        Extracts model.tar.gz into the extract_to directory.
        """
        with tarfile.open(self.download_path, 'r:gz') as tar:
            tar.extractall(path=self.extract_to)
        print(f"Model extracted to {self.extract_to}")

    def load_model(self):
        """
        Loads the best-model.pt file as a Flair SequenceTagger.
        """
        model_path = os.path.join(self.extract_to, 'best-model.pt')
        self.model = SequenceTagger.load(model_path)
        print("Model loaded successfully")

    def get_country_name(self, country_code):
        if not country_code:
            return ""
        return COUNTRIES.get(country_code.upper(), "")

    def get_state_name(self, state_code):
        if not state_code:
            return ""
        return US_STATES.get(state_code.upper(), "")

    def predict(self, text):
        sentence = Sentence(text)
        self.model.predict(sentence)

        # Initialize dictionary for storing entity predictions
        entity_dict = {
            "address_line_1": None,
            "address_line_2": None,
            "locality": None,
            "timezone": "Unknown",
            "longitude": 0.0,
            "latitude": 0.0,
            "state_name": None,
            "state_code": None,
            "country_name": None,
            "country_code": "US",  # Default
            "postal_code": None,
            "city": None,
            "phone_numbers": [],
            "emails": [],
            "ref_numbers": [],
            "recipient": None,
            "contact": None,
        }

        # Extract entities from the NER model
        for entity in sentence.get_spans('ner'):
            for label in entity.labels:
                if label.value == 'street':
                    if entity_dict["address_line_1"] is None:
                        entity_dict["address_line_1"] = entity.text
                    else:
                        entity_dict["address_line_2"] = entity.text
                elif label.value == 'city':
                    entity_dict["city"] = entity.text
                elif label.value == 'state_code':
                    entity_dict["state_code"] = entity.text
                elif label.value == 'postal_code':
                    entity_dict["postal_code"] = entity.text
                elif label.value == 'country_code':
                    entity_dict["country_code"] = entity.text
                elif label.value == 'phone_numbers':
                    entity_dict["phone_numbers"].append(entity.text)
                elif label.value == 'emails':
                    entity_dict["emails"].append(entity.text)
                elif label.value == 'ref_numbers':
                    entity_dict["ref_numbers"].append(entity.text)
                elif label.value == 'recipient':
                    entity_dict["recipient"] = entity.text
                elif label.value == 'contact':
                    entity_dict["contact"] = entity.text

        # Populate additional fields based on code lookups
        state_code = entity_dict.get("state_code", "")
        country_code = entity_dict.get("country_code", "US")
        entity_dict["state_name"] = self.get_state_name(state_code)
        entity_dict["country_name"] = self.get_country_name(country_code)

        # Construct domain objects
        country = Country(name=entity_dict["country_name"], code=entity_dict["country_code"])
        state = State(name=entity_dict["state_name"], code=entity_dict["state_code"], country=country)
        locality = Locality(
            name=entity_dict["city"],
            code=entity_dict["state_code"],
            postal_code=entity_dict["postal_code"],
            state=state
        )
        address = Address(
            address_line_1=entity_dict["address_line_1"],
            address_line_2=entity_dict["address_line_2"],
            locality=locality,
            timezone=entity_dict["timezone"],
            longitude=entity_dict["longitude"],
            latitude=entity_dict["latitude"],
            phone_numbers=entity_dict["phone_numbers"],
            emails=entity_dict["emails"],
            ref_numbers=entity_dict["ref_numbers"],
            recipient=entity_dict["recipient"],
            contact=entity_dict["contact"]
        )

        return address, entity_dict


# -------------------------------------
# Flask Application
# -------------------------------------
app = Flask(__name__)

# Load environment variables for the S3 model location
S3_BUCKET = os.getenv('S3_BUCKET', 'your-bucket')
S3_KEY = os.getenv('S3_MODEL_KEY', 'flair_address_parsing/output/model.tar.gz')
DOWNLOAD_PATH = os.getenv('DOWNLOAD_PATH', '/opt/ml/model/model.tar.gz')
EXTRACT_TO = os.getenv('EXTRACT_TO', '/opt/ml/model')

# Define the model inference object with environment variables
inference = FlairMLModelInference(
    bucket_name=S3_BUCKET,
    object_key=S3_KEY,
    download_path=DOWNLOAD_PATH,
    extract_to=EXTRACT_TO
)

# Download, extract, and load the model at startup
inference.download_model_from_s3()
inference.extract_tar_gz()
inference.load_model()

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy."""
    health = inference.model is not None  # You can add more checks here
    status = 200 if health else 404
    return jsonify({'status': 'ok' if health else 'not ready'}), status

@app.route('/invocations', methods=['POST'])
def invoke():
    """Perform inference on incoming text."""
    text = request.data.decode('utf-8')

    if not text:
        return jsonify({"error": "Invalid input format. Expecting plain text."}), 400

    try:
        address, entity_dict = inference.predict(text)
        response = {
            'address_line_1': address.address_line_1,
            'address_line_2': address.address_line_2,
            'locality': str(address.locality),
            'timezone': address.timezone,
            'longitude': address.longitude,
            'latitude': address.latitude,
            'phone_numbers': address.phone_numbers,
            'emails': address.emails,
            'ref_numbers': address.ref_numbers,
            'recipient': address.recipient,
            'contact': address.contact,
            'state_name': entity_dict.get('state_name'),
            'state_code': entity_dict.get('state_code'),
            'country_name': entity_dict.get('country_name'),
            'country_code': entity_dict.get('country_code'),
            'individual_components': entity_dict
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Adjust host/port if desired (e.g., host='0.0.0.0' for Docker container)
    app.run(host='0.0.0.0', port=8080)
