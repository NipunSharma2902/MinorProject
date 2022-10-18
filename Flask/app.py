#modules needed for the model
from pandas.core.frame import DataFrame
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from numpy import array
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objects as go
#modules needed for webapp
import requests
import json
from flask import Flask, render_template



def model_train(df):

    global df1, scaler, X_train, X_test, test_data, train_data, model

    df.to_csv('Model Data/STOCK.csv')
    df=pd.read_csv('Model Data/STOCK.csv')
    df1=df.reset_index()['close']

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

    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    #model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


dateList = []


def model_predict(df):
    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)


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


    df3=df1.tolist()
    df3.extend(lst_output)
    df3=scaler.inverse_transform(df3).tolist()
    df4=pd.DataFrame(df3, columns=['predict'])
    df=pd.read_csv("Model Data/STOCK.csv")
    df=pd.concat([df,df4], axis=1)
    df.to_csv("Model Data/predict.csv")

    dateDf = pd.read_csv("Model Data/STOCK.csv")
    dateList=dateDf['date']



a=datetime.datetime.today()
for x in range (0, 30):
    dateList.append(a + datetime.timedelta(days=x))



#Google prediction model
### Data Collection
df = pdr.get_data_tiingo('GOOG', api_key='42339adc748250f3771ee40550b91af60a4074f5')

model_train(df)

#model.save_weights("Model Data/GOOG.h5")
model.load_weights("Model Data/GOOG.h5")

### Lets Do the prediction and check performance metrics
model_predict(df)

df=pd.read_csv("Model Data/predict.csv")
fig = go.Figure([go.Scatter(x=dateList, y=df['predict'])])
fig.write_html('Flask/static/GOOG.html')

#Google model end




#Microsoft prediction model
### Data Collection
df = pdr.get_data_tiingo('MSFT', api_key='42339adc748250f3771ee40550b91af60a4074f5')

model_train(df)

#model.save_weights("Model Data/MSFT.h5")
model.load_weights("Model Data/MSFT.h5")

### Lets Do the prediction and check performance metrics
model_predict(df)

df=pd.read_csv("Model Data/predict.csv")
fig = go.Figure([go.Scatter(x=dateList, y=df['predict'])])
fig.write_html('Flask/static/MSFT.html')

#Microsoft model end




#Netflix prediction model
### Data Collection
df = pdr.get_data_tiingo('NFLX', api_key='42339adc748250f3771ee40550b91af60a4074f5')

model_train(df)

#model.save_weights("Model Data/NFLX.h5")
model.load_weights("Model Data/NFLX.h5")

### Lets Do the prediction and check performance metrics
model_predict(df)

df=pd.read_csv("Model Data/predict.csv")
fig = go.Figure([go.Scatter(x=dateList, y=df['predict'])])
fig.write_html('Flask/static/NFLX.html')

#Netflix model end



#news api part
api_key = "pub_228565b25933ad33e2f1aae23c17b23b6705"
response = requests.get("https://newsdata.io/api/1/news?apikey=pub_228565b25933ad33e2f1aae23c17b23b6705&category=business&language=en")
print(response.status_code)
response = response.json()
results = response['results']

article_head=[]
article_description=[]
article_link=[]


for result in results:
    if(result['link'] == None):
        continue
    else:
        if(result['description'] == None):
            continue
        else:
            article_head.append(result['title'])
            article_description.append(result['description'])
            article_link.append(result['link'])


length=len(article_head)


#flask integration
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', article_head=article_head, article_description=article_description, article_link=article_link, length=length)

if __name__ == "__main__":
    app.run(debug=True)