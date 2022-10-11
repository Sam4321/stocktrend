#Stock Web Trend App
#Step1 Import all the relavant libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#Step2 Loading the Data
start = '2010-07-01'
end = '2022-07-01'

st.title('Stock Trend Prediction App')

user_input = st.text_input('Enter Ticker Symbol: ','AAPL')
df = web.DataReader(user_input,'yahoo',start,end)

#Step3 Describing Data
st.subheader('Data from 2012 to 2022')
st.write(df.describe())

#Visualizing the Close Price
plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.show()

#Step4 Plot Visualizations
st.subheader('Closing Price vs Time Chart 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#Step5 Splitting Data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

#Step6 Scaling Down using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#Step7 Splitting data into x_train and y_train
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

#Step8 Load my model
from keras import models
model = load_model('keras_model.h5')

#Step9 Testing Part
past_100_days = data_training.tail(100)
final_df= past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)

#Step10 Making Predictions
y_predicted = model.predict(x_test)
scaler.scale_

scale_factor = 1/0.00716063
y_predicted = y_predicted*scale_factor

y_test = y_test*scale_factor

#Step11 Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, color='black', label=f"Original {company} price")
plt.plot(y_predicted, color= 'red', label=f"predicted {company} price")
plt.title(f"{company} share price")
plt.xlabel("time")
plt.ylabel(f"{company} share price")
plt.legend()
st.pyplot(fig2)