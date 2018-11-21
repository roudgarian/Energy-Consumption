#!/usr/bin/env python
# coding: utf-8



import plotly.plotly as py
import plotly.graph_objs as go 
import plotly
import IPython
from pylab import rcParams
#rcParams['figure.figsize'] = 18, 8
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
from IPython.display import Image
from IPython.display import IFrame
import warnings
from IPython.display import HTML
from traitlets.config.manager import BaseJSONConfigManager
from IPython.core.display import display,HTML





display(HTML("<style>.container { width:100% !important; }</style>"))


# path = "C:/Users/Saeed/AppData/Roaming/Python/etc/jupyter"
# cm = BaseJSONConfigManager(config_dir=path)
# cm.update("livereveal", {
#               "transition": "zoom",
#     "width": 1280,
#     "height": 960,
#               "start_slideshow_at": "selected",
# })




HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')





#display(HTML('<style>.prompt{width: 0px; min-width: 0px; visibility: collapse}</style>'))#disable
#display(HTML('<style>.prompt{width: 0px; min-width: 0px}</style>'))#enable




warnings.filterwarnings("ignore") 


# # Power Consumtion Exploratory Analysis
# ###### Powered By:



Image(filename='jupyter-icon.jpg') 


# ##Saeed Roudgarian
# 
# ###20.11.2018




Image(filename='Process.png')





data= pd.read_table('household_power_consumption.txt', sep=';', 
                 parse_dates={'DateTime' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, index_col='DateTime')


# # preprocessing



data.isnull().sum().sum()





data.replace('?', nan, inplace=True)
data = data.astype('float64')





def fill_missing(values):
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - 1, col]





fill_missing(data.values)





data.Global_active_power=(data.Global_active_power*1000)/60





data["Unmetered"]= data.Global_active_power - data.Sub_metering_1 - data.Sub_metering_2 - data.Sub_metering_3





data.info()





data.to_csv('Power_consumption_Preprocessed.csv')


# # Investigation and Visualization



#checkpoint to back here without doing data-preprocess step
data_df = read_csv('Power_consumption_Preprocessed.csv', header=0,
                infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])





data_df=data_df['2007-01-01':'2010-12-26']





df=data_df.resample('M').mean()





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
                       name='W.Heater_AC',
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
            dict(label = 'W.Heater_AC',
                 method = 'update',
                 args = [{'visible': [False, False, False, True, False, False]},
                         {'title': 'Electrical Water Heater and an AC',
                          'annotations': sub3_annotations}]),
            dict(label = 'Unmetered',
                 method = 'update',
                 args = [{'visible': [False, False, False, False, True, False]},
                         {'title': 'Unmetered',
                          'annotations': sub4_annotations}]),
             dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, True, True, True, True]},
                         {'title': 'Average monthly energy consumption',
                          'annotations': gap_annotations+sub1_annotations+sub2_annotations+sub3_annotations+sub4_annotations}])
        ]),
    )
])




layout = dict(title='Average monthly energy consumption', showlegend= False,height=650, width=800,
              updatemenus=updatemenus)





fig = dict(data=data, layout=layout)
py.iplot(fig, name=('Energy'))





week_min=data_df['2008-08-18':'2008-08-24']
#day_min=data_df['2008-08-23':'2008-08-23']
week_max=data_df['2007-12-24':'2007-12-30']
#day_max=data_df['2007-12-28':'2007-12-28']





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
                       name='W.Heater_AC',
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
                       name='W.Heater_AC',
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





fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=('High Energy Consumption Week(Dec Last Week )',
                                          'Low Energy Consumption Week(Aug 3th Week)'))





fig.append_trace(trace_gap, 1, 1)
fig.append_trace(trace_sub1, 1, 1)
fig.append_trace(trace_sub2, 1, 1)
fig.append_trace(trace_sub3, 1, 1)
fig.append_trace(trace_sub4, 1, 1)
fig.append_trace(trace3_gap, 2, 1)
fig.append_trace(trace3_sub1, 2, 1)
fig.append_trace(trace3_sub2, 2, 1)
fig.append_trace(trace3_sub3, 2, 1)
fig.append_trace(trace3_sub4, 2, 1)





fig['layout'].update(height=680, width=750)
py.iplot(fig)


# ###### Time Series Decomposition




rcParams['figure.figsize'] = 13, 8
Series= data_df['Global_active_power'].resample('W').mean()
decomposition = sm.tsa.seasonal_decompose(Series,freq=50, model='additive')
fig = decomposition.plot()


# ###### The plot clearly shows that the energy consumption is unstable, along with its obvious seasonality.

# ## Forcasting the Time Series

# ##### ARIMA method (Autoregressive Integrated Moving Average)

# ##### With parameter combinations for Seasonal ARIMA(p,d,q):
# ##### SARIMAX: (7, 1, 1) x (2, 1, 0, 12)



dataset = read_csv('Power_consumption_Preprocessed.csv',
                   header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])





#dataset['Global_active_power']=dataset['Global_active_power']*0.06





arima_Series=dataset['Global_active_power'].resample('M').mean()





p = range(1,9)
d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             model = sm.tsa.statespace.SARIMAX(arima_Series,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#             results = model.fit()
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue



model = sm.tsa.statespace.SARIMAX(arima_Series,
                                order=(3, 0, 1),
                                seasonal_order=(2, 0, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = model.fit()
print(results.summary().tables[1])


# results.plot_diagnostics(figsize=(16, 8),lags=10)
# plt.show()

# #### Validating forecast
# ###### understand the accuracy
# ###### compare predicted consumed engery to real energy consumption




pred = results.get_prediction(start=pd.to_datetime('2009-12-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = arima_Series['2006':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.9, figsize=(16, 6))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.25)
ax.set_xlabel('Date')
ax.set_ylabel('Average Global Active Power')
plt.grid(color='gray', linestyle='--', linewidth=0.3)
plt.legend()
plt.show()


# ###### our forecasts align with the true values well , it starts from the beginning of the year and captured the seasonality toward the end of the year

# #### Producing and visualizing forecasts




arima_Series_forecasted = pred.predicted_mean
arima_Series_truth = arima_Series['2006-12-31':]
mse = ((arima_Series_forecasted - arima_Series_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))





print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))





pred_uc = results.get_forecast(steps=50)
pred_ci = pred_uc.conf_int()
ax = arima_Series.plot(label='observed', figsize=(16, 6))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Average Global Active Power')
#ax.set_title('RMSE {}'.format(round(np.sqrt(mse), 2)),color='red')
plt.grid(color='gray', linestyle='--', linewidth=0.3)
plt.legend()
plt.show()


# ###### . captured energy consumption seasonality
# ###### . confidence intervals

# ###### Our model captured energy consumption seasonality.Forecast further out into the Energy Consumption,then less confident.
# ###### This is reflected by the confidence intervals generated by our model, which grow larger as we move further out into the future.

# # Conclusion and Recommendation

# ###### . To change Family consumption behaviour, Creating an app connected to the Sub-meters to send report and notification about Energy consumption is recommended.
# ###### . To reduce the cost of installing 4 Sub-meters, use one submeter with new technology to control all sections
# ###### . It seems to be enough to use only one Sub-meter for the kitchen and the laundry room.
# ###### . According to the amount of consumed energy in missed part, install a Sub-meter in this part is recommended 

# # Thank You!






