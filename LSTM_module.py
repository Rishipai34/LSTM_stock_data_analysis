from keras.layers.recurrent_v2 import lstm_with_backend_selection
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
# importing the required libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import preprocess

dataset = pd.read_csv('data/SBIN.csv')
training_data = dataset.iloc[:,[8]] # exctract the required colums from the data.In this case High, Low, Open, Close
#finding some features of the data 
ohcl_average = dataset.mean(axis = 1)
hcl_average = dataset[['High', 'Low', 'Close']].mean(axis = 1)
closing_values = dataset[['Close']]
indexes = np.arange(1,len(dataset) + 1, 1)
plt.plot(indexes, ohcl_average, 'b', label = 'OHCL Average')
plt.plot(indexes, hcl_average, 'y', label = 'HCL Average')
plt.plot(indexes, closing_values, label = 'Closing Price')
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

training_data = np.reshape(training_data.values, (len(training_data),1))
sc = MinMaxScaler(feature_range = (0,1))
scaled_training_closing = sc.fit_transform(training_data)

training_data_closing = int(len(scaled_training_closing) * 0.75)
testing_data_closing = len(scaled_training_closing) - training_data_closing

training_data_closing, testing_data_closing = scaled_training_closing[0:training_data_closing], scaled_training_closing[training_data_closing:len(scaled_training_closing)]
X,y_train = preprocess.preprocess(training_data_closing, 1)

x_train = np.reshape(X, (X.shape[0], X.shape[1], 1))
lstm_model = Sequential()
lstm_model.add(LSTM(128, return_sequences=True, input_shape = (x_train.shape[1],1)))
lstm_model.add(LSTM(50, return_sequences= False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss= 'mean_squared_error')

lstm_model.fit(x_train, y_train, epochs=10, batch_size= 32) # train the model

#generate the data for testing 
X,y_test = preprocess.preprocess(testing_data_closing, 1)
x_test = np.reshape(X, (X.shape[0], X.shape[1], 1))

#make predictions:
lstm_prediction = lstm_model.predict(x_test)
lstm_prediction = sc.inverse_transform(lstm_prediction)

data = dataset[['Close']]
train = data[:len(training_data_closing)]
valid = data[len(training_data_closing)+2:]
valid['Predictions'] = lstm_prediction
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()