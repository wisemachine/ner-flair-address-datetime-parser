import boto3

class SagemakerEndpointClient:
    def __init__(self, endpoint_name, region_name='us-east-1'):
        self.endpoint_name = endpoint_name
        self.runtime_client = boto3.client('sagemaker-runtime', region_name=region_name)
        print(f"Initialized SagemakerEndpointClient with endpoint: {self.endpoint_name}")

    def predict(self, text):
        # Send single input text
        input_data = text.strip()
        print(f"Sending input data to endpoint {self.endpoint_name}: {input_data}")

        try:
            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='text/plain',
                Body=input_data
            )
            print(f"Received response from endpoint: {response}")

            # Assuming the response is a JSON string
            result = response['Body'].read().decode()
            print(f"Decoded result: {result}")

            return result
        except Exception as e:
            print(f"Error during inference: {e}")
            raise
