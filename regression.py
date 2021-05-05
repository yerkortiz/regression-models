from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import os

def load_data(inputPath):
    cols = ["Country", "Unix", "Disaster", "Id", "Quantity"]
    df = pd.read_csv(inputPath, sep=",", header=None, names=cols)
    return df

def process_data(df, train, test):
	continuous = ["Country", "Disaster", "Quantity"]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])




# import the necessary packages
from tensorflow.keras.models import Model

def create_mlp(dim, regress=False):
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
	if regress:
		model.add(Dense(1, activation="linear"))
	return model

inputPath = "/Users/yerko/codes/twitter-traffic-predict/datasets/dataset.csv"
df = load_data(inputPath)
(train, test) = train_test_split(df, test_size=0.25, random_state=42)