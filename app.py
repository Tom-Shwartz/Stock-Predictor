import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import tensorflow as tf
from keras.models import load_model
import streamlit as st

start = dt.datetime(2010,1,1)
end =  dt.datetime(2019,12,31)

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
data = yf.download(user_input, start , end)

st.subheader('Data from 2010 - 2019')
st.write(data.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(data.Close, 'tab:blue')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(data.Close, 'tab:blue')
st.pyplot(fig)

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.7)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.7):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

test_dates = data.index[int(len(data) * 0.7):]

# Plot Predictions vs Original Prices with actual dates on x-axis
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(test_dates, y_test, 'b', label='Original Price')
plt.plot(test_dates, y_predicted, 'r', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)