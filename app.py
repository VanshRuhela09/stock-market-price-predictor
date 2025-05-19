import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from keras.models import load_model # type: ignore

model = load_model('C:\\Users\\Dell\\Desktop\\projects\\stock market price predictor\\Stock Predictions Model.keras')
st.header('Stock market predictor')
stock = st.text_input('Enter Stock Symbol' , 'GOOG')

start = '2013-01-01'
end = '2023-12-31'

data = yf.download(stock,start,end)

if data.empty:
    st.error(f"Failed to load stock data for symbol '{stock}'. Please check the stock symbol or try another.")
    st.stop()

st.subheader('stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) # used here to fit the data between 0 and 1

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days,data_test] , ignore_index = True) # adding the train dataset into the test data set
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price VS MA50') #moving average for 50 days
MA_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(MA_50_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price VS MA50 VS MA100') #moving average for 100 days
MA_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(MA_50_days,'r')
plt.plot(MA_100_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price VS MA100 VS MA200')
MA_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(MA_100_days,'r')
plt.plot(MA_200_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig3)


x = []
y = []
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x,y = np.array(x) , np.array(y)
y_predict = model.predict(x)

scale = 1/scaler.scale_
y_predict = y_predict*scale
y = y*scale

st.subheader('Original Price VS Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y_predict,'r',label='Original Price')
plt.plot(y,'g',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)