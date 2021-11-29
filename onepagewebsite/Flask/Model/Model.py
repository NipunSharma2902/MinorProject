# -*- coding: utf-8 -*-
"""MSFT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PekyOe-pSpK4gNqJblaDPAYQuADNRfBi
"""

### Data Collection
import pandas_datareader as pdr

df = pdr.get_data_tiingo('NFLX', api_key='42339adc748250f3771ee40550b91af60a4074f5')

df.to_csv('STOCK.csv')

import pandas as pd

df=pd.read_csv('STOCK.csv')

df1=df.reset_index()['close']

import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')



#model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

#model.save_weights("NFLX.h5")

model.load_weights("D:/Minor Project/Repository/MinorProject/onepagewebsite/Flask/Model/NFLX.h5")


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
#math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
#math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict



x_input=test_data[340:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        
        x_input=np.array(temp_input[1:])
       
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
      
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
      
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1

day_new=np.arange(1,100)
day_pred=np.arange(101,131)

import matplotlib.pyplot as plt

df3=df1.tolist()
df3.extend(lst_output)

df3=scaler.inverse_transform(df3).tolist()

df4=pd.DataFrame(df3, columns=['predict'])

df=pd.read_csv("STOCK.csv")

df=pd.concat([df,df4], axis=1)

df.to_csv("predict.csv")

import datetime
b=datetime.datetime.today()
a = b- datetime.timedelta(days=1259)

numdays = 1288
dateList = []
for x in range (0, numdays):
    dateList.append(a + datetime.timedelta(days=x))

import plotly.graph_objects as go
fig = go.Figure([go.Scatter(x=dateList, y=df['predict'])])
fig.write_html('D:/Minor Project/Repository/MinorProject/onepagewebsite/Flask/static/NFLX.html')
