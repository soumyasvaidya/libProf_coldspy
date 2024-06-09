import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle
import logging
import boto3
import os
import urllib.request
import random

# Setting up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variable for S3 bucket
s3_bucket = os.environ['s3_model_bucket']
logger.info("Bucket for model is: " + str(s3_bucket))

def handler(event, context):
    model_name_prefix = 'model-optim-'
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    # Seeds for repeatability
    data_split_seed = 2
    random_forest_seed = 2

    logger.info(" > Importing Data < ")
    response = urllib.request.urlopen(dataset_url)
    data = np.genfromtxt(response, delimiter=';', skip_header=1)

    logger.info(" > Splitting in Training & Testing < ")
    np.random.seed(data_split_seed)
    
    split = np.random.rand(data.shape[0]) <= 0.90

    X_train = data[split, :-1]
    X_test = data[~split, :-1]

    y_train = data[split, -1]
    y_test = data[~split, -1]

    logger.info(" > Training Random Forest Model < ")
    regressor = RandomForestRegressor(max_depth=None, n_estimators=30, random_state=random_forest_seed)
    regressor.fit(X_train, y_train)

    logger.info(" > Saving model to S3 < ")
    model_name = model_name_prefix + str(random.randint(0, 100000))
    temp_file_path = '/tmp/' + model_name

    with open(temp_file_path, 'wb') as f:
        pickle.dump(regressor, f)

    with open(temp_file_path, 'rb') as f:
        model_data = f.read()

    s3 = boto3.resource('s3')
    s3.Object(s3_bucket, model_name).put(Body=model_data)

    logger.info("Model saved with name: " + model_name)

    logger.info(" > Evaluating the Model < ")
    y_predicted = regressor.predict(X_test)
    logger.info("Mean Absolute Error on full test set: " + str(round(metrics.mean_absolute_error(y_test, y_predicted), 3)))

    return model_name
