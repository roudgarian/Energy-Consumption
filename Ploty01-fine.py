
# coding: utf-8

# In[26]:


import plotly.plotly as py
import plotly.graph_objs as go 
import plotly
plotly.tools.set_credentials_file(username='roudgarian', api_key='487wi04XQXVDhzZVfABJ')
from datetime import datetime
import pandas_datareader.data as web
import pandas as pd
import itertools
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from pylab import rcParams
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric
from numpy import split
from numpy import array
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt



data = read_csv('Power_consumption_Preprocessed.csv', header=0,
                infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])




df=data.resample('M').mean()



trace_gap = go.Scatter(x=list(df.index),
                       y=list(df.Global_active_power),
                          name='GAP',
                          line=dict(color='#B40431'))



trace_sub1 = go.Scatter(x=list(df.index),
                       y=list(df.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'))



trace_sub2 = go.Scatter(x=list(df.index),
                       y=list(df.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'))




trace_sub3 = go.Scatter(x=list(df.index),
                       y=list(df.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'))



trace_sub4 = go.Scatter(x=list(df.index),
                       y=list(df.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'))



data = [trace_gap, trace_sub1, trace_sub2, trace_sub3, trace_sub4 ]




gap_annotations=[dict(x=df.Global_active_power.idxmax(),
                       y=df.Global_active_power.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Global_active_power.max()),
                       ax=0, ay=-40)]


sub1_annotations=[dict(x=df.Sub_metering_1.idxmax(),
                       y=df.Sub_metering_1.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Sub_metering_1.max()),
                       ax=0, ay=-40)]



sub2_annotations=[dict(x=df.Sub_metering_2.idxmax(),
                       y=df.Sub_metering_2.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Sub_metering_2.max()),
                       ax=0, ay=-40)]



sub3_annotations=[dict(x=df.Sub_metering_3.idxmax(),
                       y=df.Sub_metering_3.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Sub_metering_3.max()),
                       ax=0, ay=-40)]


sub4_annotations=[dict(x=df.Unmetered.idxmax(),
                       y=df.Unmetered.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Unmetered.max()),
                       ax=0, ay=40)]


updatemenus = list([
    dict(active=-1,
         buttons=list([   
            dict(label = 'GAP',
                 method = 'update',
                 args = [{'visible': [True, False, False, False, False, False]},
                         {'title': 'Global Active Power'}]),
            dict(label = 'Kitchen',
                 method = 'update',
                 args = [{'visible': [False, True, False, False, False, False]},
                         {'title': 'Kitchen',
                          'annotations': sub1_annotations}]),
            dict(label = 'Laundry.R',
                 method = 'update',
                 args = [{'visible': [False, False, True, False, False, False]},
                         {'title': 'Laundry Room',
                          'annotations': sub2_annotations}]),
            dict(label = 'Heating_AC',
                 method = 'update',
                 args = [{'visible': [False, False, False, True, False, False]},
                         {'title': 'Heating_AC',
                          'annotations': sub3_annotations}]),
            dict(label = 'Unmetered',
                 method = 'update',
                 args = [{'visible': [False, False, False, False, True, False]},
                         {'title': 'Unmetered',
                          'annotations': sub4_annotations}]),
             dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, True, True, True, True]},
                         {'title': 'All Attribude',
                          'annotations': gap_annotations+sub1_annotations+sub2_annotations+sub3_annotations+sub4_annotations}])
        ]),
    )
])


layout = dict(title='Energy Consumption',height=700, width=1000, showlegend=False,
              updatemenus=updatemenus)



fig = dict(data=data, layout=layout)
py.iplot(fig, filename='update_dropdown')

