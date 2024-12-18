import usaddress
import pyap
import json
import warnings
import pycountry
import openai

class MlParsedAddress:
    """
    A data structure to hold parsed address information.
    """
    def __init__(
        self, 
        address_line_1=None, 
        address_line_2=None, 
        locality=None, 
        timezone="Unknown",
        longitude=0.0, 
        latitude=0.0, 
        state_name=None, 
        state_code=None, 
        country_name=None,
        country_code="US", 
        postal_code=None, 
        city=None, 
        phone_numbers=None, 
        emails=None,
        ref_numbers=None, 
        recipient=None, 
        contact=None
    ):
        self.address_line_1 = address_line_1
        self.address_line_2 = address_line_2
        self.locality = locality
        self.timezone = timezone
        self.longitude = longitude
        self.latitude = latitude
        self.state_name = state_name
        self.state_code = state_code
        self.country_name = country_name
        self.country_code = country_code
        self.postal_code = postal_code
        self.city = city
        self.phone_numbers = phone_numbers or []
        self.emails = emails or []
        self.ref_numbers = ref_numbers or []
        self.recipient = recipient
        self.contact = contact

    def __str__(self):
        return (
            f"Address Line 1: {self.address_line_1}\n"
            f"Address Line 2: {self.address_line_2}\n"
            f"Locality: {self.locality}\n"
            f"Timezone: {self.timezone}\n"
            f"Longitude: {self.longitude}\n"
            f"Latitude: {self.latitude}\n"
            f"State: {self.state_name} ({self.state_code})\n"
            f"Postal Code: {self.postal_code}\n"
            f"Country: {self.country_name} ({self.country_code})\n"
            f"City: {self.city}\n"
            f"Phone Numbers: {self.phone_numbers}\n"
            f"Emails: {self.emails}\n"
            f"Reference Numbers: {self.ref_numbers}\n"
            f"Recipient: {self.recipient}\n"
            f"Contact: {self.contact}\n"
        )

    def to_dict(self):
        return {
            "address_line_1": self.address_line_1,
            "address_line_2": self.address_line_2,
            "locality": self.locality,
            "timezone": self.timezone,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "state_name": self.state_name,
            "state_code": self.state_code,
            "country_name": self.country_name,
            "country_code": self.country_code,
            "postal_code": self.postal_code,
            "city": self.city,
            "phone_numbers": self.phone_numbers,
            "emails": self.emails,
            "ref_numbers": self.ref_numbers,
            "recipient": self.recipient,
            "contact": self.contact,
        }


class AddressParser:
    """
    Address parser with local parsing (usaddress, pyap) + LLM fallback using OpenAI Function Calling.
    """

    tag_mapping = {
        'Recipient': 'recipient',
        'AddressNumber': 'street',
        'AddressNumberPrefix': 'street',
        'AddressNumberSuffix': 'street',
        'StreetName': 'street',
        'StreetNamePreDirectional': 'street',
        'StreetNamePreModifier': 'street',
        'StreetNamePreType': 'street',
        'StreetNamePostDirectional': 'street',
        'StreetNamePostModifier': 'street',
        'StreetNamePostType': 'street',
        'CornerOf': 'street',
        'IntersectionSeparator': 'street',
        'LandmarkName': 'street',
        'USPSBoxGroupID': 'street',
        'USPSBoxGroupType': 'street',
        'USPSBoxID': 'street',
        'USPSBoxType': 'street',
        'BuildingName': 'street',
        'OccupancyType': 'street',
        'OccupancyIdentifier': 'street',
        'SubaddressIdentifier': 'street',
        'SubaddressType': 'street',
        'PlaceName': 'city',
        'StateName': 'state_code',
        'ZipCode': 'postal_code',
        'CountryName': 'country_code'
    }

    def __init__(self, openai_api_key=None):
        """
        Initialize the AddressParser with an optional OpenAI API key.
        If not provided here, ensure openai.api_key is set externally.
        """
        if openai_api_key:
            openai.api_key = openai_api_key

    def get_country_name(self, country_code):
        """
        Look up the country name by alpha-2 code (e.g., 'US' -> 'United States').
        """
        try:
            country = pycountry.countries.get(alpha_2=country_code.upper())
            return country.name if country else None
        except KeyError:
            return None

    def get_state_name(self, state_code, country_code):
        """
        Look up the state/province name by code (e.g., 'CA' in 'US' -> 'California').
        """
        try:
            country_code = country_code.upper()
            if country_code == "US":
                state = pycountry.subdivisions.get(code=f"US-{state_code.upper()}")
            else:
                state = pycountry.subdivisions.get(code=f"{country_code}-{state_code.upper()}")
            return state.name if state else None
        except KeyError:
            return None

    def create_full_address(self, address_dict):
        components_order = [
            "street", "city", "state_code", "state_long",
            "postal_code", "country_code", "country_long"
        ]
        address_components = [address_dict.get(component, '') for component in components_order]
        full_address = ', '.join(filter(None, address_components))
        address_dict['full_address'] = full_address
        return address_dict

    def rearrange_dict(self, input_dict, order):
        ordered_dict = {}
        for key in order:
            if key in input_dict:
                ordered_dict[key] = input_dict[key]
        for key, value in input_dict.items():
            if key not in ordered_dict:
                ordered_dict[key] = value
        return ordered_dict

    def map_pyap_address(self, address):
        address_dict = {
            "full_address": address.full_address,
            "street": address.full_street,
            "city": address.city,
            "state_code": address.region1,
            "postal_code": address.postal_code,
            "country_code": address.country_id
        }

        country_long = self.get_country_name(address_dict.get('country_code', ''))
        state_long = self.get_state_name(
            address_dict.get('state_code', ''), 
            address_dict.get('country_code', '')
        )

        if country_long:
            address_dict['country_long'] = country_long
        if state_long:
            address_dict['state_long'] = state_long

        return address_dict

    def parse_us_address(self, address_text):
        """
        Parses a US address from free-form text using usaddress, falling back to pyap if needed.
        Returns a JSON string with parsed info or {"error": "..."}.
        """
        try:
            tagged_address, _ = usaddress.tag(address_text, tag_mapping=AddressParser.tag_mapping)
            complete_tagged_address = self.create_full_address(tagged_address)

            if not complete_tagged_address.get('full_address'):
                return json.dumps({"error": "No address found"})

            # Enrich with country/state names
            country_long = self.get_country_name(complete_tagged_address.get('country_code', ''))
            state_long = self.get_state_name(
                complete_tagged_address.get('state_code', ''), 
                complete_tagged_address.get('country_code', '')
            )
            if country_long:
                complete_tagged_address['country_long'] = country_long
            if state_long:
                complete_tagged_address['state_long'] = state_long

            order = [
                "full_address", "recipient", "street", "city", 
                "state_code", "state_long", "postal_code", 
                "country_code", "country_long"
            ]
            result = self.rearrange_dict(complete_tagged_address, order)
            return json.dumps(result)

        except usaddress.RepeatedLabelError:
            # Fallback: pyap
            try:
                addresses = pyap.parse(address_text, country='US')
                if addresses:
                    address_dict = self.map_pyap_address(addresses[0])
                    return json.dumps(address_dict)
                else:
                    return json.dumps({"error": "No address found"})
            except Exception as e:
                return json.dumps({"error": str(e)})

    # -------------------------------
    # LLM-BASED ADDRESS PARSING BELOW
    # -------------------------------

    def get_messages(self, address):
        """
        Construct the conversation messages for an OpenAI function call.
        """
        return [
            {"role": "system", "content": "You are a helpful assistant that extracts address information."},
            {"role": "user", "content": f"Here is the address: {address}."}
        ]

    def get_function_parameters(self):
        """
        Defines the JSON schema for function-calling to parse address info.
        """
        return [
            {
                "name": "extract_address_info",
                "description": "Extract structured address info from raw text input.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address_line_1": {"type": "string"},
                        "address_line_2": {"type": "string", "nullable": True},
                        "locality": {"type": "string"},
                        "timezone": {"type": "string", "default": "Unknown"},
                        "longitude": {"type": "number", "default": 0.0},
                        "latitude": {"type": "number", "default": 0.0},
                        "state_name": {"type": "string"},
                        "state_code": {"type": "string"},
                        "country_name": {"type": "string"},
                        "country_code": {"type": "string", "default": "US"},
                        "postal_code": {"type": "string"},
                        "city": {"type": "string"},
                        "phone_numbers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "nullable": True
                        },
                        "emails": {
                            "type": "array",
                            "items": {"type": "string"},
                            "nullable": True
                        },
                        "ref_numbers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "nullable": True
                        },
                        "recipient": {"type": "string"},
                        "contact": {"type": "string", "nullable": True},
                    },
                    "required": [
                        "address_line_1", "locality", "state_code", 
                        "postal_code", "country_code", "recipient"
                    ]
                }
            }
        ]

    def call_openai_functions(self, messages, functions, temperature=0.0):
        """
        Generic function to call OpenAI ChatCompletion with function calls.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",  # or 'gpt-3.5-turbo-0613'
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=temperature
        )
        role = response.choices[0].message.role
        content = response.choices[0].message.content or ""
        function_call = response.choices[0].message.function_call

        if role == "assistant" and function_call:
            # Parse arguments if the function is called
            try:
                arguments = json.loads(function_call.arguments)
                return arguments
            except json.JSONDecodeError:
                warnings.warn("Invalid JSON in function call arguments.")
                return {}
        else:
            # If no function call was made, parse direct content
            try:
                content_json = json.loads(content)
                return content_json
            except json.JSONDecodeError:
                return {}

    def validate_recipient_contact(self, content, address):
        """
        Additional check: if 'recipient' is not at the start of the address, 
        swap 'recipient' and 'contact' to match certain heuristics.
        """
        recipient = content.get('recipient', '')
        contact = content.get('contact', '')
        if recipient and not address.startswith(recipient):
            # Swap recipient and contact
            recipient, contact = contact, recipient
        content['recipient'] = recipient
        content['contact'] = contact
        return content

    def call_llm(self, address, messages, functions, temperature=0.0):
        """
        Wrap the LLM function-call, apply post-processing.
        """
        content = self.call_openai_functions(messages, functions, temperature)
        content = self.validate_recipient_contact(content, address)

        # Return a dictionary with default values if keys are missing
        return {
            "address_line_1": content.get("address_line_1", ""),
            "address_line_2": content.get("address_line_2", ""),
            "locality": content.get("locality", ""),
            "timezone": content.get("timezone", "Unknown"),
            "longitude": content.get("longitude", 0.0),
            "latitude": content.get("latitude", 0.0),
            "state_name": content.get("state_name", ""),
            "state_code": content.get("state_code", ""),
            "country_name": content.get("country_name", ""),
            "country_code": content.get("country_code", "US"),
            "postal_code": content.get("postal_code", ""),
            "city": content.get("city", ""),
            "phone_numbers": content.get("phone_numbers", []),
            "emails": content.get("emails", []),
            "ref_numbers": content.get("ref_numbers", []),
            "recipient": content.get("recipient", ""),
            "contact": content.get("contact", ""),
        }

    def parse_address_with_llm(self, address_text):
        """
        Parses the address using OpenAI function calling to retrieve structured address data.
        Returns an MlParsedAddress object.
        """
        messages = self.get_messages(address_text)
        functions = self.get_function_parameters()
        parsed_address = self.call_llm(address_text, messages, functions)

        return MlParsedAddress(
            address_line_1=parsed_address.get('address_line_1'),
            address_line_2=parsed_address.get('address_line_2'),
            locality=parsed_address.get('locality'),
            timezone=parsed_address.get('timezone', "Unknown"),
            longitude=parsed_address.get('longitude', 0.0),
            latitude=parsed_address.get('latitude', 0.0),
            state_name=parsed_address.get('state_name'),
            state_code=parsed_address.get('state_code'),
            country_name=parsed_address.get('country_name'),
            country_code=parsed_address.get('country_code', "US"),
            postal_code=parsed_address.get('postal_code'),
            city=parsed_address.get('city'),
            phone_numbers=parsed_address.get('phone_numbers', []),
            emails=parsed_address.get('emails', []),
            ref_numbers=parsed_address.get('ref_numbers', []),
            recipient=parsed_address.get('recipient'),
            contact=parsed_address.get('contact'),
        )

    # --------------------------
    # PUBLIC METHODS / WORKFLOW
    # --------------------------

    def parse_address(self, address_text):
        """
        Generic method:
          1. Attempt local address parsing (usaddress -> pyap fallback).
          2. If that fails or yields incomplete results, try LLM parsing.
        """
        # 1. Local parse
        parsed_result_json = self.parse_us_address(address_text)
        parsed_result = json.loads(parsed_result_json)

        # If local parser fails, or if the result is missing key fields, fallback to LLM
        if "error" in parsed_result or not parsed_result.get("street") or not parsed_result.get("postal_code"):
            warnings.warn("Local parse incomplete. Falling back to LLM.")
            return self.parse_address_with_llm(address_text)

        # Otherwise, convert local parse to MlParsedAddress
        address_obj = MlParsedAddress(
            address_line_1=parsed_result.get('street'),
            address_line_2=parsed_result.get('full_address'),
            city=parsed_result.get('city'),
            state_code=parsed_result.get('state_code'),
            country_code=parsed_result.get('country_code', "US"),
            postal_code=parsed_result.get('postal_code'),
            recipient=parsed_result.get('recipient')
        )
        # Populate long names if available
        address_obj.country_name = parsed_result.get('country_long')
        address_obj.state_name = parsed_result.get('state_long')

        return address_obj


# ----------------------------------------------------------------------------
# Example usage:
# 
# parser = AddressParser(openai_api_key="YOUR_OPENAI_API_KEY")
# text = "John Smith, 123 Main St, Anytown, CA 90210"
# parsed_address = parser.parse_address(text)
# print(parsed_address)
# print(parsed_address.to_dict())
# ----------------------------------------------------------------------------
