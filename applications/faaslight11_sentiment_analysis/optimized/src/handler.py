import time
init_st = time.time() * 1000
import os
import pickle
import gzip
# import pandas as pd
import numpy as np
from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer


import gzip
import pickle

CLASSES = {
    0: "negative",
    4: "positive"
}

MODEL_FILE = 'model.dat.gz'
with gzip.open(MODEL_FILE, 'rb') as f:
    MODEL = pickle.load(f, encoding='bytes')


init_ed = time.time() * 1000


def handler(event, context):
    """
        Validate parameters and call the recommendation engine
        @event: API Gateway's POST body;
        @context: LambdaContext instance;
    """
    fun_st = time.time() * 1000
    # input validation
    # assert event, "AWS Lambda event parameter not provided"
    # text = event.get("text")  # query text
    # assert isinstance(text, basestring)
    text = "In the gentle embrace of dawn, where the first rays of sunlight tenderly caress the earth, there exists a tranquil symphony of hope and renewal. Each morning, as the world awakens from the slumber of night, there's an unspoken promise lingering in the air—a promise of endless possibilities and the chance to begin anew. With each breath, there's a subtle reminder that yesterday's struggles need not define today's journey; that within every obstacle lies the seed of opportunity waiting to be nurtured. It's in these moments of quiet contemplation, amidst the soft whispers of the morning breeze, that one finds solace in the beauty of the world and the resilience of the human spirit. For even in the darkest of times, there remains a glimmer of light, a beacon of hope guiding us through the shadows. And as the sun rises above the horizon, painting the sky with hues of amber and gold, we're reminded that no matter how daunting the path may seem, there's always a reason to believe in the power of a new day."
    # text = json.loads(event)["test"]

    # call predicting function
    print(predict(text))
    fun_ed = time.time() * 1000
    return ",InitStart:{},".format(init_st)+"InitEnd:{},".format(init_ed)+"functionStart:{},".format(fun_st)+"functionEnd:{},".format(fun_ed)




def predict(text):
    """
        Predict the sentiment of a string
        @text: string - the string to be analyzed
    """
    
    x_vector = MODEL.vectorizer.transform([text])
    
    y_predicted = MODEL.predict(x_vector)

    return CLASSES.get(y_predicted[0])

# print(handler(json.dumps({"test":"This function is awesome"})))
# print(handler("",""))

