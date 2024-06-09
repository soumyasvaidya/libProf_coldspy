import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle
import random
import logging
import boto3
import os

# Setting up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variable for S3 bucket
s3_bucket = os.environ['s3_model_bucket']
logger.info("Bucket for model is: " + str(s3_bucket))

def handler(event, context):
    model_name_prefix = 'model-'
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    # Seeds for repeatability
    data_split_seed = 2
    random_forest_seed = 2

    logger.info(" > Importing Data < ")
    data = pd.read_csv(dataset_url, header='infer', na_values='?', sep=';')

    logger.info(" > Splitting in Training & Testing < ")
    np.random.seed(data_split_seed)

    data['split'] = np.random.randn(data.shape[0], 1)
    split = np.random.rand(len(data)) <= 0.90

    X_train = data[split].drop(['quality', 'split'], axis=1)
    X_test = data[~split].drop(['quality', 'split'], axis=1)

    y_train = data.quality[split]
    y_test = data.quality[~split].values

    logger.info(" > Training Random Forest Model < ")
    regressor = RandomForestRegressor(max_depth=None, n_estimators=30, random_state=random_forest_seed)
    regressor.fit(X_train, y_train)

    logger.info(" > Saving model to S3 < ")
    str(random.randint(0, 100000))
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