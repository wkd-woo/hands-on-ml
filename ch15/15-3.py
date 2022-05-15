# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"
# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras

assert tf.__version__ >= "2.0"
import numpy as np
import os
import pandas as pd

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deploy"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import googleapiclient.discovery

# read the file containing the pima indians diabities data set
data = pd.read_csv('../datasets/diabetes.csv', sep=',')
print("\ndata.head(): \n", data.head())
# describe the columns of the data set
data.describe()
# see if the data set has null values
data.info()
print("\n\nStep 2 - Prepare the data for the model building")
# extract the X and y from the imported data
X = data.values[:, 0:8]
y = data.values[:, 8]
# use MinMaxScaler to fit a scaler object
scaler = MinMaxScaler()
scaler.fit(X)
# min max scale the X data
X = scaler.transform(X)
# split the test set into the train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("\n\n###############################################################")
print("And now let's use saved_model_cli to make predictions \n"
      "for the instances we just saved:")
print("let's start by creating the query.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "myproject-1548545104898-dbb77a9b79ce.json"
# set the variables for the gcp ai platform
project_id = "myproject-1548545104898"
model_id = "my_pima_model"
model_path = "projects/{}/models/{}".format(project_id, model_id)
model_path += "/versions/v0001/"  # if you want to run a specific version
ml_resource = googleapiclient.discovery.build("ml", "v1").projects()
print("\nmodel_path: \n", model_path)


# function use to request a prediction from the web api of the model
# and get a reponse of the predctions
def predict(X):
    input_data_json = {"signature_name": "serving_default", "instances": X.tolist()}
    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    print("\nresponse: \n", response)
    if "error" in response:
        raise RuntimeError(response["error"])
    return np.array([pred['dense_2'] for pred in response["predictions"]])


print("\nX_test: \n", X_test)
Y_probas = predict(X_test[:3])
# predict the results for the test set
print("\n\npredict(X_test[:3]): \n", np.round(Y_probas, 2))
