
# coding: utf-8

# In[92]:


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
from plotly import tools


# In[40]:


data= pd.read_table('household_power_consumption.txt', sep=';', 
                 parse_dates={'DateTime' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, index_col='DateTime')


# # preprocessing

# In[41]:


data.isnull().sum().sum()


# In[42]:


data.replace('?', nan, inplace=True)
data = data.astype('float64')


# In[43]:


def fill_missing(values):
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - 1, col]


# In[44]:


fill_missing(data.values)


# In[46]:


data.Global_active_power=(data.Global_active_power*1000)/60


# In[47]:


data["Unmetered"]= data.Global_active_power - data.Sub_metering_1 - data.Sub_metering_2 - data.Sub_metering_3


# In[48]:


data.info()


# In[49]:


data.to_csv('Power_consumption_Preprocessed.csv')


# # Investigation and Visualization

# In[50]:


#checkpoint to back here without doing data-preprocess step
data = read_csv('Power_consumption_Preprocessed.csv', header=0,
                infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])


# In[51]:


df=data.resample('M').mean()


# In[52]:


trace_gap = go.Scatter(x=list(df.index),
                       y=list(df.Global_active_power),
                          name='GAP',
                          line=dict(color='#B40431'))


# In[53]:


trace_sub1 = go.Scatter(x=list(df.index),
                       y=list(df.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'))


# In[54]:


trace_sub2 = go.Scatter(x=list(df.index),
                       y=list(df.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'))


# In[55]:


trace_sub3 = go.Scatter(x=list(df.index),
                       y=list(df.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'))


# In[56]:


trace_sub4 = go.Scatter(x=list(df.index),
                       y=list(df.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'))


# In[57]:


#data = [trace_gap, trace_sub1, trace_sub2, trace_sub3, trace_sub4 ]


# In[58]:


gap_annotations=[dict(x=df.Global_active_power.idxmax(),
                       y=df.Global_active_power.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Global_active_power.max()),
                       ax=0, ay=-40)]


# In[59]:


sub1_annotations=[dict(x=df.Sub_metering_1.idxmax(),
                       y=df.Sub_metering_1.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Sub_metering_1.max()),
                       ax=0, ay=-40)]


# In[60]:


sub2_annotations=[dict(x=df.Sub_metering_2.idxmax(),
                       y=df.Sub_metering_2.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Sub_metering_2.max()),
                       ax=0, ay=-40)]


# In[61]:


sub3_annotations=[dict(x=df.Sub_metering_3.idxmax(),
                       y=df.Sub_metering_3.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Sub_metering_3.max()),
                       ax=0, ay=-40)]


# In[62]:


sub4_annotations=[dict(x=df.Unmetered.idxmax(),
                       y=df.Unmetered.max(),
                       xref='x', yref='y',
                       text='Max:<br>'+str(df.Unmetered.max()),
                       ax=0, ay=40)]


# In[63]:


updatemenus = list([
    dict(active=-1,
         buttons=list([   
            dict(label = 'GAP',
                 method = 'update',
                 args = [{'visible': [True, False, False, False, False, False]},
                         {'title': 'Global Active Power',
                          'annotations': gap_annotations}]),
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


# In[64]:


layout = dict(title='Energy Consumption',height=700, width=1000, showlegend=False,
              updatemenus=updatemenus)


# In[36]:


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='update_dropdown')


# In[66]:


week_min=data['2008-08-18':'2008-08-24']
day_min=data['2008-08-23':'2008-08-23']
week_max=data['2007-12-24':'2007-12-30']
day_max=data['2007-12-28':'2007-12-28']


# In[67]:


week_max=week_max.resample('h').mean()


# In[68]:


trace_gap = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Global_active_power),
                       legendgroup= 'group1',
                          name='GAP',
                          line=dict(color='#B40431'),
                       showlegend= True,
                      visible ='legendonly')


# In[69]:


trace_sub1 = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'),
                       legendgroup= 'group2',
                       showlegend= True,
                      visible ='legendonly')


# In[70]:


trace_sub2 = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'),
                       legendgroup= 'group3',
                       showlegend= True,
                      visible ='legendonly')


# In[71]:


trace_sub3 = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'),
                       legendgroup= 'group4',
                       showlegend= True,
                      visible ='legendonly')


# In[72]:


trace_sub4 = go.Scatter(x=list(week_max.index),
                       y=list(week_max.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'),
                       legendgroup= 'group5',
                       showlegend= True,
                      visible ='legendonly')


# In[73]:


day_max=day_max.resample('h').mean()


# In[74]:


trace2_gap = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Global_active_power),
                        legendgroup= 'group1',
                          name='GAP',
                          line=dict(color='#B40431'),
                       showlegend= False,
                      visible ='legendonly')


# In[75]:


trace2_sub1 = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'),
                       legendgroup= 'group2',
                       showlegend= False,
                      visible ='legendonly')


# In[76]:


trace2_sub2 = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'),
                       legendgroup= 'group3',
                       showlegend= False,
                      visible ='legendonly')


# In[77]:


trace2_sub3 = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'),
                       legendgroup= 'group4',
                       showlegend= False,
                      visible ='legendonly')


# In[78]:


trace2_sub4 = go.Scatter(x=list(day_max.index),
                       y=list(day_max.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'),
                       legendgroup= 'group5',
                       showlegend= False,
                      visible ='legendonly')


# In[79]:


week_min=week_min.resample('h').mean()


# In[80]:


trace3_gap = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Global_active_power),
                        legendgroup= 'group1',
                          name='GAP',
                          line=dict(color='#B40431'),
                       showlegend= False,
                      visible ='legendonly')


# In[81]:


trace3_sub1 = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'),
                       legendgroup= 'group2',
                       showlegend= False,
                      visible ='legendonly')


# In[82]:


trace3_sub2 = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'),
                       legendgroup= 'group3',
                       showlegend= False,
                      visible ='legendonly')


# In[83]:


trace3_sub3 = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'),
                       legendgroup= 'group4',
                       showlegend= False,
                      visible ='legendonly')


# In[84]:


trace3_sub4 = go.Scatter(x=list(week_min.index),
                       y=list(week_min.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'),
                       legendgroup= 'group5',
                       showlegend= False,
                      visible ='legendonly')


# In[85]:


day_min=day_min.resample('h').mean()


# In[86]:


trace4_gap = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Global_active_power),
                        legendgroup= 'group1',
                          name='GAP',
                          line=dict(color='#B40431'),
                       showlegend= False,
                      visible ='legendonly')


# In[87]:


trace4_sub1 = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Sub_metering_1),
                       name='Kitchen',
                       line=dict(color='#2d7f5e'),
                       legendgroup= 'group2',
                       showlegend= False,
                      visible ='legendonly')


# In[88]:


trace4_sub2 = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Sub_metering_2),
                       name='Laundry.R',
                       line=dict(color='#0080FF'),
                       legendgroup= 'group3',
                       showlegend= False,
                      visible ='legendonly')


# In[89]:


trace4_sub3 = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Sub_metering_3),
                       name='Heating_AC',
                       line=dict(color='#FFBF00'),
                       legendgroup= 'group4',
                       showlegend= False,
                      visible ='legendonly')


# In[90]:


trace4_sub4 = go.Scatter(x=list(day_min.index),
                       y=list(day_min.Unmetered),
                       name='Unmetered',
                       line=dict(color='#3A2F0B'),
                       legendgroup= 'group5',
                       showlegend= False,
                      visible ='legendonly')


# In[93]:


fig = tools.make_subplots(rows=2, cols=2,
                          subplot_titles=('High Energy Consumption Week(Dec Last Week )',
                                          'High Energy Consumption Day',
                                                          'Low Energy Consumption Week(Aug 3th Week)',
                                          'Low Energy Consumption Day'))


# In[94]:


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


# In[95]:


fig['layout'].update(height=700, width=800, title='Multiple Subplots' +
                                                  ' with Titles')


# In[96]:


py.iplot(fig, filename='make-subplots-multiple-with-titles')


# In[38]:


Series= data['Global_active_power'].resample('W').mean()
decomposition = sm.tsa.seasonal_decompose(Series,freq=50, model='additive')
fig = decomposition.plot()
#plt.savefig('Trend-Seasonal.png', dpi = 1200)
#print(decomposition.trend)
#print(decomposition.seasonal)
#print(decomposition.resid)
#print(decomposition.observed)
plt.show()


# # Training and Test Set

# In[23]:


dataset = read_csv('Power_consumption_Preprocessed.csv',
                   header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])


# In[24]:


dataset['Global_active_power']=dataset['Global_active_power']*0.06


# In[25]:


arima_Series=dataset['Global_active_power'].resample('M').mean()


# In[26]:


p = range(1,9)
d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[ ]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(arima_Series,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = model.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[27]:


model = sm.tsa.statespace.SARIMAX(arima_Series,
                                order=(7, 1, 1),
                                seasonal_order=(2, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = model.fit()
print(results.summary().tables[1])


# results.plot_diagnostics(figsize=(16, 8),lags=10)
# plt.show()

# In[28]:


pred = results.get_prediction(start=pd.to_datetime('2009-12-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = arima_Series['2007':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.25)
ax.set_xlabel('Date')
ax.set_ylabel('Average Global Active Power')
plt.savefig('Validating forecasts.png', dpi = 1200)
plt.legend()


# In[29]:


arima_Series_forecasted = pred.predicted_mean
arima_Series_truth = arima_Series['2006-12-31':]
mse = ((arima_Series_forecasted - arima_Series_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[30]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# In[42]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = arima_Series.plot(label='observed', figsize=(14, 5))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Average Global Active Power')
ax.set_title('RMSE {}'.format(round(np.sqrt(mse), 2)),color='red')
plt.legend()
#plt.savefig('forecasts.png', dpi = 1200)
plt.show()

