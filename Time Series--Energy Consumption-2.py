
# coding: utf-8

# In[221]:


import pandas as pd
import itertools
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 14, 5
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


# In[222]:


data= pd.read_table('household_power_consumption.txt', sep=';', 
                 parse_dates={'DateTime' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, index_col='DateTime')


# # preprocessing

# In[223]:


data.isnull().sum().sum()


# In[224]:


data.replace('?', nan, inplace=True)
data = data.astype('float64')


# In[225]:


def fill_missing(values):
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - 1, col]


# In[226]:


fill_missing(data.values)


# In[227]:


data.index#data=data['2007-01-01':'2010-11-21']

#data = data.astype('float64')
#data["Global_active_power"] = pd.to_numeric(data["Global_active_power"], errors='coerce')
#data["Global_reactive_power"] = pd.to_numeric(data["Global_reactive_power"], errors='coerce')
#data["Voltage"] = pd.to_numeric(data["Voltage"], errors='coerce')
#data["Global_intensity"] = pd.to_numeric(data["Global_intensity"], errors='coerce')
#data["Sub_metering_1"] = pd.to_numeric(data["Sub_metering_1"], errors='coerce')
#data["Sub_metering_2"] = pd.to_numeric(data["Sub_metering_2"], errors='coerce')
# In[228]:


data.Global_active_power=(data.Global_active_power*1000)/60

data.Global_reactive_power=(data.Global_reactive_power*1000)/60
# In[229]:


data["Unmetered"]= data.Global_active_power - data.Sub_metering_1 - data.Sub_metering_2 - data.Sub_metering_3

decimals = pd.Series([2,2,2], index=['Global_active_power','Global_reactive_power', 'Unmetered'])
data.round(decimals)
# In[230]:


data.info()


# In[231]:


# save updated dataset
#data.to_csv('household_power_consumption.csv')

# data=data.dropna(how='any')#data.Global_active_power.resample('D').mean().plot(title='Daily Global Power', color='gray')#data_M =data.resample('M').mean()
#we=data_sorted.sort_values(by='Global_active_power', ascending=False)
#data_M.head()#data['Global_active_power'].resample('M').mean().plot(kind='bar',color='blue',subplots = True)#data['Unmetered'].resample('M').mean().plot(kind='bar',color='r',subplots = True)
#data['Sub_metering_1'].resample('M').mean().plot(kind='bar',color='red',subplots = True)#data_sorted =data.resample('Y').mean()
#we=data_sorted.sort_values(by='Global_active_power', ascending=False, na_position='first')
#we.head()
#y = data['Global_active_power'].resample('MS').sum()
#y['2007']
#y['2008']
#y['2009']
#y['2010']
#y.plot(figsize=(19, 9))
# In[232]:


data.to_csv('Power_consumption_Preprocessed.csv')


# # Investigation and Visualization

# In[233]:


#checkpoint to back here without doing data-preprocess step
data = read_csv('Power_consumption_Preprocessed.csv', header=0,
                infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])


# In[234]:


data.Global_active_power.resample('Y').sum().plot(color='#B40431', legend=True,figsize=(8, 5))
data.Unmetered.resample('Y').sum().plot(color='#3A2F0B', legend=True,figsize=(8, 5))
data.Sub_metering_3.resample('Y').sum().plot(color='#FFBF00', legend=True,figsize=(8, 5))
data.Sub_metering_2.resample('Y').sum().plot(color='#0080FF', legend=True,figsize=(8, 5))
data.Sub_metering_1.resample('Y').sum().plot( color='#2d7f5e', legend=True,figsize=(8, 5))
#data.Global_intensity.resample('Y').sum().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper left')
#plt.savefig('Anual.png', dpi = 2400)
plt.show()
#plot(figsize=(19, 9))


# In[235]:


#fig, axes = plt.subplots(ncols=2)

a=data.Global_active_power.resample('Q').mean().plot(color='#B40431',stacked=True,legend=True,figsize=(12, 5))
b=data.Unmetered.resample('Q').mean().plot(color='#3A2F0B',stacked=True,legend=True,figsize=(12, 5))
c=data.Sub_metering_3.resample('Q').mean().plot(color='#FFBF00',stacked=True,legend=True,figsize=(12, 5))
d=data.Sub_metering_2.resample('Q').mean().plot(color='#0080FF',stacked=True,legend=True,figsize=(12, 5))
e=data.Sub_metering_1.resample('Q').mean().plot( color='#2d7f5e',stacked=True,legend=True,figsize=(12, 5))
#data.Global_intensity.resample('M').mean().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper right')
#labels = [item.get_text() for item in b.get_yticklabels()]
#labels[1] = '0'
#labels[2] = '500'
#labels[3] = '1000'
#labels[4] = '1500'
#labels[5] = '2000'
#labels[6] = '2500'
#labels[7] = '3000'
#a.set_yticklabels(labels)
plt.xlabel("")
plt.ylabel("K Watt/h")
#plt.savefig('Seasonal.png', dpi = 4800)
plt.show()


# In[236]:


#data_sorted =data.resample('M').mean()
#month=data_sorted.sort_values(by='Global_active_power', ascending=False, na_position='first')
#data_sorted.head(50)
#data_sorted.to_excel('exampleResult.xls', sheet_name='Sheet1')


# In[237]:


a=data.Global_active_power.resample('M').mean().plot(color='#B40431',stacked=True,figsize=(12, 5))
b=data.Unmetered.resample('M').mean().plot(color='#3A2F0B',stacked=True,figsize=(12, 5))
c=data.Sub_metering_3.resample('M').mean().plot(color='#FFBF00',stacked=True,figsize=(12, 5))
d=data.Sub_metering_2.resample('M').mean().plot(color='#0080FF',stacked=True,figsize=(12, 5))
e=data.Sub_metering_1.resample('M').mean().plot( color='#2d7f5e',stacked=True,figsize=(12, 5))
#data.Global_intensity.resample('M').mean().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
#plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper right')
#labels = [item.get_text() for item in b.get_yticklabels()]
#labels[1] = '0'
#labels[2] = '500'
#labels[3] = '1000'
#labels[4] = '1500'
#labels[5] = '2000'
#labels[6] = '2500'
#labels[7] = '3000'
#a.set_yticklabels(labels)
plt.xlabel("")
#plt.ylabel("K Watt/h")
#plt.savefig('Monthly.png', dpi = 2400)
plt.show()


# In[238]:


#we.plot(color='blue',legend=True)


# In[239]:


#sns.heatmap(data, xticklabels=data.Global_active_power, yticklabels=data.Unmetered, vmin=-1, vmax=1)
#plt.show() 


# data_2007=data['2007-01-01':'2007-12-31']
# data_2008=data['2008-01-01':'2008-12-31']
# data_2009=data['2009-01-01':'2009-12-31']
# data_2010=data['2010-01-01':'2010-12-31']

# In[241]:


data_sorted_max =data.resample('M').mean()
Max=data_sorted_max.sort_values(by='Global_active_power', ascending=False, na_position='first')
Max.head()


# In[242]:


data_sorted_min =data.resample('M').mean()
Min=data_sorted_min.sort_values(by='Global_active_power')
Min.head()


# In[243]:


week_min=data['2008-08-18':'2008-08-24']
day_min=data['2008-08-23':'2008-08-23']
week_max=data['2007-12-24':'2007-12-30']
day_max=data['2007-12-28':'2007-12-28']
#jan=data['2007-01-01':'2007-01-31']
#Feb=data['2007-02-01':'2007-02-28']


# In[244]:


#data_sorted_w01 =week01.resample('Min').mean()
#data_sorted_w01.head(7)
#data_sorted_w01.to_excel('data_sorted_w01.xls', sheet_name='Sheet1')


# In[245]:


week_min.Global_active_power.resample('h').mean().plot(color='#B40431',stacked=True,figsize=(120, 5))
week_min.Unmetered.resample('h').mean().plot(color='#3A2F0B',stacked=True,figsize=(12, 5))
week_min.Sub_metering_3.resample('h').mean().plot(color='#FFBF00',stacked=True,figsize=(12, 5))
week_min.Sub_metering_2.resample('h').mean().plot(color='#0080FF',stacked=True,figsize=(12, 5))
week_min.Sub_metering_1.resample('h').mean().plot(color='#2d7f5e',stacked=True,figsize=(12, 5))
#plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper right')
plt.xlabel('18Aug-24Aug 2008')
#plt.ylabel('Kw/h')
#plt.savefig('Min-Weekly.png', dpi = 2400)
plt.show()


# In[246]:


#data_sorted_d01 =day01.resample('Min').mean()
#data_sorted_w01.head(7)
#data_sorted_d01.to_excel('data_sorted_d01.xls', sheet_name='Sheet1')


# In[247]:


day_min.Global_active_power.resample('h').mean().plot(color='#B40431',stacked=True,figsize=(12, 5))
day_min.Unmetered.resample('h').mean().plot(color='#3A2F0B',stacked=True,figsize=(12, 5))
day_min.Sub_metering_3.resample('h').mean().plot(color='#FFBF00',stacked=True,figsize=(12, 5))
day_min.Sub_metering_2.resample('h').mean().plot(color='#0080FF',stacked=True,figsize=(12, 5))
day_min.Sub_metering_1.resample('h').mean().plot(color='#2d7f5e',stacked=True,figsize=(12, 5))
#plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper right')
plt.xlabel('23-Aug 2008')
#plt.ylabel('Kw/h')
#plt.savefig('Min-Daily.png', dpi = 2400)
plt.show()


# In[248]:


#data_sorted_w02 =week02.resample('Min').mean()
#data_sorted_w01.head(7)
#data_sorted_w02.to_excel('data_sorted_w02.xls', sheet_name='Sheet1')


# In[249]:


week_max.Global_active_power.resample('h').mean().plot(color='#B40431',stacked=True,figsize=(12, 5))
week_max.Unmetered.resample('h').mean().plot(color='#3A2F0B',stacked=True,figsize=(12, 5))
week_max.Sub_metering_3.resample('h').mean().plot(color='#FFBF00',stacked=True,figsize=(12, 5))
week_max.Sub_metering_2.resample('h').mean().plot(color='#0080FF',stacked=True,figsize=(12, 5))
week_max.Sub_metering_1.resample('h').mean().plot(color='#2d7f5e',stacked=True,figsize=(12, 5))
#plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper left')
plt.xlabel('24-Dec 30-Dec 2007')
#plt.ylabel('Kw/h')
#plt.savefig('Max-Weekly.png', dpi = 2400)
plt.show()


# In[250]:


#data_sorted_d02 =day02.resample('Min').mean()
#data_sorted_w01.head(7)
#data_sorted_d02.to_excel('data_sorted_d02.xls', sheet_name='Sheet1')


# In[251]:


#fig, ax =plt.subplots(1)
day_max.Global_active_power.resample('h').mean().plot(color='#B40431',stacked=True,legend=True,figsize=(12, 5))
day_max.Unmetered.resample('h').mean().plot(color='#3A2F0B',stacked=True,legend=True,figsize=(12, 5))
day_max.Sub_metering_3.resample('h').mean().plot(color='#FFBF00',stacked=True,legend=True,figsize=(12, 5))
day_max.Sub_metering_2.resample('h').mean().plot(color='#0080FF',stacked=True,legend=True,figsize=(12, 5))
day_max.Sub_metering_1.resample('h').mean().plot(color='#2d7f5e',stacked=True,legend=True,figsize=(12, 5))
plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper left')
plt.xlabel('28-Dec 2007')
#plt.ylabel('Kw/h')
#plt.savefig('Max-Daily.png', dpi = 2400)
plt.show()


# In[252]:


#plt.matshow(data.corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
#plt.title('without resampling', size=10)
#plt.colorbar()
#plt.show()
#autocorrelation_plot(data)
#pyplot.show()


# In[253]:


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

# In[254]:


dataset = read_csv('Power_consumption_Preprocessed.csv',
                   header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])


# In[255]:


dataset['Global_active_power']=dataset['Global_active_power']*0.06


# In[256]:


gap_ts=dataset['Global_active_power'].resample('M').mean()


# In[257]:


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
#             model = sm.tsa.statespace.SARIMAX(gap_ts,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#             results = model.fit()
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue

# In[258]:


model = sm.tsa.statespace.SARIMAX(gap_ts,
                                order=(7, 1, 1),
                                seasonal_order=(2, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = model.fit()
print(results.summary().tables[1])


# results.plot_diagnostics(figsize=(16, 8),lags=10)
# plt.show()

# In[263]:


pred = results.get_prediction(start=pd.to_datetime('2009-12-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = gap_ts['2007':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.25)
ax.set_xlabel('Date')
ax.set_ylabel('Average Global Active Power')
plt.savefig('Validating forecasts.png', dpi = 1200)
plt.legend()


# In[260]:


gap_ts_forecasted = pred.predicted_mean
gap_ts_truth = gap_ts['2006-12-31':]
mse = ((gap_ts_forecasted - gap_ts_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[261]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# In[264]:


pred_uc = results.get_forecast(steps=70)
pred_ci = pred_uc.conf_int()
ax = gap_ts.plot(label='observed', figsize=(14, 5))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Average Global Active Power')
ax.set_title('RMSE {}'.format(round(np.sqrt(mse), 2)),color='red')
plt.legend()
plt.savefig('forecasts.png', dpi = 1200)
plt.show()

