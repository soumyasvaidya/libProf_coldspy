import boto3
import json
import os
# Replace 'your_function_name' with the actual name of your Lambda function
function_name = 'model_serving_custom'

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

# Create a Boto3 Lambda client
lambda_client = boto3.client('lambda', region_name='us-east-1', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)


# Replace 'your_input_payload' with the payload you want to send to the Lambda function
input_payload={
  "x": "The ambiance is magical. The food and service was nice! The lobster and cheese was to die for and our steaks were cooked perfectly.",
  "dataset_object_key": "reviews10mb.csv",
  "dataset_bucket": "modeltraining-dataset",
  "model_bucket": "modeltraining-model",
  "model_object_key": "lr_model.pk"
}

# Invoke the Lambda function
for i in range (0, 30):
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',  # Use 'Event' for asynchronous invocation
        Payload=json.dumps(input_payload)
    )

    # Extract and print the result
    result = json.loads(response['Payload'].read().decode('utf-8'))
    print(result)
