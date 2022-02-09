# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:04:48 2022

@author: Admin
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from plotly import graph_objs as go
#import plotly.plotly as py
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from pylab import rcParams
import datetime


st.header('Load Prediction')
st.subheader('Model to forecast Guarenteed weight for different routes')

##reading the data
excel_file = 'Route Rate Per Ton Analysis 2020-21.xlsx'
sheet_name = 'data'
df = pd.read_excel(excel_file,sheet_name,header=1)


df_1 = df.groupby(['routedesc','DespatchDatetime'])['gauranteedwt'].sum().reset_index()


st.sidebar.header("Please Filter Here")
route= st.sidebar.text_input("Enter the route",'RUC-ZZZ')
period = st.sidebar.slider("Number of weeks",1,4,2)


week = st.sidebar.radio(
     "Choose number of weeks to get forecast",
     ('1 week', '2 weeks', '3 weeks'))

if week == '1 week':
     #st.write('You selected 1 week.')
     week_no = 1
if week == '2 weeks':
     #st.write('You selected 2 weeks.')
     week_no = 2
if week == '3 weeks':
     #st.write('You selected 3 weeks.')
     week_no = 3
else:
    week_no = 2

d_from = st.sidebar.date_input(
     "Select From Date",
     datetime.date(2019, 2, 9))

d_to = st.sidebar.date_input(
     "Select To Date",
     datetime.date(2022, 2, 15))

df_selection = df_1.query("routedesc == @route")
st.write("snapshot of the data")
st.dataframe(df_selection.head(5))

fig = px.line(df_selection, x="DespatchDatetime", y="gauranteedwt")
fig.layout.update(title_text='Time Series data representation', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)



def convert_to_date(x):
    return datetime.datetime.strptime(x + '-1', "%Y-%W-%w")


df_test = df_selection.sort_values(by='DespatchDatetime')
df_test  = df_test.reset_index(drop=True)
df_test['DespatchDatetime'] = pd.to_datetime(df_test['DespatchDatetime'])
df_test['dayOfWeek'] = df_test['DespatchDatetime'].dt.day_name()
df_test['Week_Number'] =  df_test['DespatchDatetime'].dt.week
df_test['year'] =df_test['DespatchDatetime'].dt.year
df_test_week=df_test.groupby(['year','Week_Number'])['gauranteedwt'].mean().reset_index()
df_test_week = df_test_week.sort_values(by=['year','Week_Number']).reset_index()
df_test_week['week_year'] = df_test_week['year'].astype('str')+'-'+df_test_week['Week_Number'].astype('str')

df_test_week['Date'] = pd.to_datetime(df_test_week['week_year'].apply(convert_to_date))
df_test_week = df_test_week.iloc[:len(df_test_week)-2]


df_train = df_test_week[['Date','gauranteedwt']]
df_train = df_train.rename(columns={'Date':'ds','gauranteedwt':'y'})


train_len = len(df_train) - period


df_train2 = df_train.iloc[:train_len]
df_test2 = df_train.iloc[train_len:]


model = SimpleExpSmoothing(df_train2['y'])
results = model.fit(smoothing_level=0.8, optimized=False)
predictions = results.forecast(steps=period)


fig = px.line(df_test_week, x=df_test_week.index, y="gauranteedwt")
fig.layout.update(title_text='weekly average demand', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)



fig = go.Figure()
fig.add_trace(go.Scatter( x=df_train2.index, y=df_train2["y"],mode='lines',name='train'))
fig.add_trace(go.Scatter( x=df_test2.index, y=df_test2["y"],mode='markers',name='test'))
fig.add_trace(go.Scatter( x=predictions.index, y=predictions.values,mode='markers',name='forecast'))
fig.add_trace(go.Scatter( x=results.fittedvalues.index, y=results.fittedvalues.values,mode='lines',name='fitted values'))
fig.layout.update(title_text='Forecasting on the weekly average demand', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)
actual = df_train2["y"].values
forecast = results.fittedvalues.values
mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
st.write("Mape -> "+ str(round(mape*100,2)))
ema_pred = pd.concat([pd.DataFrame(df_test2['y']).reset_index(drop=True),pd.Series(predictions.values)],axis = 1)
ema_pred.columns = ['Actual weekly average','Forecasted weekly average']
st.dataframe(ema_pred)





model = Prophet()
model.fit(df_train2)
future = pd.DataFrame(df_test2['ds'])
forecast = model.predict(future)
fig2= plot_plotly(model, forecast)
st.plotly_chart(fig2)



prophet_pred = pd.concat([pd.DataFrame(df_test2['y']).reset_index(drop=True),forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)],axis = 1)
st.dataframe(prophet_pred)


