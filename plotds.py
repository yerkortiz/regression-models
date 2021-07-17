import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
def load_data(inputPath):
    cols = ["Country", "Unix", "Disaster", "Quantity"]
    df = pd.read_csv(inputPath, sep=",", header=None, names=cols)
    df1 = df[['Unix', 'Quantity']]
    return df1

dataset_path = "/Users/yerko/codes/twitter-traffic-predict/datasets/dataset1n_60.csv"
raw_dataset = load_data(dataset_path)
dataset = raw_dataset.copy()
dataset.tail()

dataset.isna().sum()
dataset = dataset.dropna()

#print(dataset)

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["Unix", "Quantity"]], diag_kind="kde")
plt.show()