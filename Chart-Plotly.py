
# coding: utf-8

# In[1]:


import plotly.plotly as py
import plotly.graph_objs as go 
import plotly
plotly.tools.set_credentials_file(username='', api_key='')
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
from plotly import tools



data = read_csv('Power_consumption_Preprocessed.csv', header=0,
                infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])


week_min=data['2008-08-18':'2008-08-24']
day_min=data['2008-08-23':'2008-08-23']
week_max=data['2007-12-24':'2007-12-30']
day_max=data['2007-12-28':'2007-12-28']



week_max=week_max.resample('h').mean()


trace_gap = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Global_active_power),
                       legendgroup= 'group1',
                          name='GAP',
                          line=dict(color='#B40431'),
                       showlegend= True,
                      visible ='legendonly')



trace_sub1 = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'),
                       legendgroup= 'group2',
                       showlegend= True,
                      visible ='legendonly')



trace_sub2 = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'),
                       legendgroup= 'group3',
                       showlegend= True,
                      visible ='legendonly')



trace_sub3 = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'),
                       legendgroup= 'group4',
                       showlegend= True,
                      visible ='legendonly')


trace_sub4 = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'),
                       legendgroup= 'group5',
                       showlegend= True,
                      visible ='legendonly')



data = [trace_gap, trace_sub1, trace_sub2, trace_sub3, trace_sub4 ]


day_max=day_max.resample('h').mean()


trace2_gap = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Global_active_power),
                        legendgroup= 'group1',
                          name='GAP',
                          line=dict(color='#B40431'),
                       showlegend= False,
                      visible ='legendonly')



trace2_sub1 = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'),
                       legendgroup= 'group2',
                       showlegend= False,
                      visible ='legendonly')



trace2_sub2 = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'),
                       legendgroup= 'group3',
                       showlegend= False,
                      visible ='legendonly')


trace2_sub3 = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'),
                       legendgroup= 'group4',
                       showlegend= False,
                      visible ='legendonly')




trace2_sub4 = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'),
                       legendgroup= 'group5',
                       showlegend= False,
                      visible ='legendonly')




week_min=week_min.resample('h').mean()


trace3_gap = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Global_active_power),
                        legendgroup= 'group1',
                          name='GAP',
                          line=dict(color='#B40431'),
                       showlegend= False,
                      visible ='legendonly')



trace3_sub1 = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'),
                       legendgroup= 'group2',
                       showlegend= False,
                      visible ='legendonly')



trace3_sub2 = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'),
                       legendgroup= 'group3',
                       showlegend= False,
                      visible ='legendonly')



trace3_sub3 = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'),
                       legendgroup= 'group4',
                       showlegend= False,
                      visible ='legendonly')



trace3_sub4 = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'),
                       legendgroup= 'group5',
                       showlegend= False,
                      visible ='legendonly')



day_min=day_min.resample('h').mean()


trace4_gap = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Global_active_power),
                        legendgroup= 'group1',
                          name='GAP',
                          line=dict(color='#B40431'),
                       showlegend= False,
                      visible ='legendonly')



trace4_sub1 = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'),
                       legendgroup= 'group2',
                       showlegend= False,
                      visible ='legendonly')



trace4_sub2 = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'),
                       legendgroup= 'group3',
                       showlegend= False,
                      visible ='legendonly')



trace4_sub3 = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'),
                       legendgroup= 'group4',
                       showlegend= False,
                      visible ='legendonly')



trace4_sub4 = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'),
                       legendgroup= 'group5',
                       showlegend= False,
                      visible ='legendonly')



fig = tools.make_subplots(rows=2, cols=2,
                          subplot_titles=('High Energy Consumption Week(Dec Last Week )',
                                          'High Energy Consumption Day',
                                                          'Low Energy Consumption Week(Aug 3th Week)',
                                          'Low Energy Consumption Day'))




fig.append_trace(trace_gap, 1, 1)
fig.append_trace(trace_sub1, 1, 1)
fig.append_trace(trace_sub2, 1, 1)
fig.append_trace(trace_sub3, 1, 1)
fig.append_trace(trace_sub4, 1, 1)
fig.append_trace(trace2_gap, 1, 2)
fig.append_trace(trace2_sub1, 1, 2)
fig.append_trace(trace2_sub2, 1, 2)
fig.append_trace(trace2_sub3, 1, 2)
fig.append_trace(trace2_sub4, 1, 2)
fig.append_trace(trace3_gap, 2, 1)
fig.append_trace(trace3_sub1, 2, 1)
fig.append_trace(trace3_sub2, 2, 1)
fig.append_trace(trace3_sub3, 2, 1)
fig.append_trace(trace3_sub4, 2, 1)
fig.append_trace(trace4_gap, 2, 2)
fig.append_trace(trace4_sub1, 2, 2)
fig.append_trace(trace4_sub2, 2, 2)
fig.append_trace(trace4_sub3, 2, 2)
fig.append_trace(trace4_sub4, 2, 2)



fig['layout'].update(height=700, width=800, title='Multiple Subplots' +
                                                  ' with Titles')



py.iplot(fig, filename='make-subplots-multiple-with-titles')

