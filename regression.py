import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(inputPath):
    cols = ["Country", "Unix", "Disaster", "Quantity"]
    df = pd.read_csv(inputPath, sep=",", header=None, names=cols)
    return df

def get_metrics(Y, Y_pred):#metrics
    from sklearn import metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(Y, Y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y, Y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, Y_pred)))
    #print('R2 Score:', metrics.r2_score(Y, Y_pred))

datasetPath = 'datasets/dataset.csv'
df = load_data(datasetPath)
#plot dataframe (quantity, unix)
df.plot(x='Quantity', y='Unix', style='o')
plt.title('Unix, Quantity')
plt.xlabel('Quantity')
plt.ylabel('Unix')
plt.show()

#set train dataframe
df1 = df[['Unix', 'Quantity']]
print(df1.shape)
print(df1.head())
print(df1.describe())
X = df1.iloc[:, :-1].values.reshape(-1, 1)
Y = df1.iloc[:, 1].values.reshape(-1, 1)

def linearRegression(X, Y):
    print(X.shape, Y.shape)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
    linear_regressor = LinearRegression()  
    linear_regressor.fit(x_train, y_train)  
    Y_pred = linear_regressor.predict(x_test)
    plt.scatter(x_train, y_train)
    plt.plot(x_test, Y_pred, color='red')
    plt.show()
    get_metrics(y_test, Y_pred)
    #y = linear_regressor.predict(np.array([0]).reshape(-1,1))
    #print(y)
    
linearRegression(X, Y)

def logisticRegression(X, Y):
    X_train, X_test, y_train, y_test  = train_test_split(X, Y,train_size=0.80, random_state=1234)
    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()
    log_model = log_model.fit(X=X_train, y=y_train)    
    y_pred = log_model.predict(X_test)