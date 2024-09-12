import json
import boto3
import joblib
import numpy as np

# If the model is stored in S3, download it and load it
def load_model_from_s3(bucket_name, model_key):
    s3 = boto3.client('s3')
    model_path = '/tmp/' + model_key
    s3.download_file(bucket_name, model_key, model_path)
    model = joblib.load(model_path)
    return model

# Initialize model (e.g., load from S3)
MODEL_BUCKET = "libprof-data-models"
MODEL_KEY = "logistic_model.joblib"
model = load_model_from_s3(MODEL_BUCKET, MODEL_KEY)

def lambda_handler(event, context):
    # Parse the input data from the event
    if 'body' in event:
        event = json.loads(event['body'])

    input_data = event['data']
    
    # Convert input_data to the format expected by the model (e.g., numpy array)
    data = np.array(input_data).reshape(1, -1)
    
    # Perform prediction
    prediction = model.predict(data)
    
    # Return the result as a JSON object
    return {
        'statusCode': 200,
        'body': json.dumps({
            'prediction': prediction.tolist()  # Convert prediction to list for JSON serialization
        })
    }
