import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
# importing the required libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout

dataset = pd.read_csv('data/SBIN.csv')
dataset = dataset.reindex(index = dataset.index[::-1]) # reverses the data to get the data in order
training_data = dataset.iloc[:,4:8].values # exctract the required colums from the data.In this case High, Low, Open, Close
#finding some features of the data 
ohcl_average = dataset.mean(axis = 1)
hcl_average = dataset[['High', 'Low', 'Close']].mean(axis = 1)
closing_values = dataset[['Close']]
indexes = np.arange(1,len(dataset) + 1, 1)
plt.plot(indexes, ohcl_average, 'b', label = 'OHCL Average')
plt.plot(indexes, hcl_average, 'y', label = 'HCL Average')
plt.plot(indexes, dataset[['Close']], label = 'Closing Price')
plt.legend(loc = 'upper left')
plt.show()
plt.figure(figsize = (15,6))
dataset['Close'].plot()
plt.show()


ma_timeframe = [10, 20, 50] 
for time_frame in ma_timeframe:
    column_name = f"Moving Average for {time_frame} days"
    dataset[column_name] = dataset['Close'].rolling(time_frame).mean();

dataset[['Close', 'Moving Average for 10 days', 'Moving Average for 20 days', 'Moving Average for 50 days']].plot()
plt.title('Tata beverages moving average')
plt.show()


sc = MinMaxScaler(feature_range = (0,1))
scaled_training_data = sc.fit_transform(training_data)

# preprocessing the data for training 
X,Y = [], []
for i in range(len(dataset)-59): # we take a timestep of 60 days 
    a = dataset[i:(i+60), 0]
    X.append(a)
    Y.append(dataset[i + 60, 0])

x_train = np.array(X)
y_train = np.array(Y)
x_train = np.reshape(xtrain, (x_train.shape[0], x_train.shape[1], 1))



