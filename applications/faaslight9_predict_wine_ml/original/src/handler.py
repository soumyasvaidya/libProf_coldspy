import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle
import random
import logging
import boto3
import os
import json

# Setting up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variable for S3 bucket
s3_bucket = os.environ['s3_model_bucket']
logger.info("Bucket for model is: " + str(s3_bucket))

def handler(event, context):

    if 'body' in event:
        event = json.loads(event['body'])

    input_for_prediction = pd.DataFrame({k: [v] for k, v in event.items() if k != 'model name'})

    logger.info("Input data for prediction:")
    logger.info(str(input_for_prediction))

    model_name = event['model name']
    logger.info(" > Downloading model from S3 < ")
    temp_file_path = '/tmp/' + model_name

    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, model_name, temp_file_path)

    logger.info(" > Loading model to memory < ")
    with open(temp_file_path, 'rb') as f:
        model = pickle.load(f)

    logger.info(" > Predicting wine quality < ")
    predicted_wine_grade = model.predict(input_for_prediction)

    return str(round(predicted_wine_grade[0], 1))
