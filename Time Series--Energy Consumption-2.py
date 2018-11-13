
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

decimals = pd.Series([2,2,2], index=['Global_active_power','Global_reactive_power', 'Unmetered'])
data.round(decimals)


data.to_csv('Power_consumption_Preprocessed.csv')


# # Investigation and Visualization

#checkpoint to back here without doing data-preprocess step
data = read_csv('Power_consumption_Preprocessed.csv', header=0,
                infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])


data.Global_active_power.resample('Y').sum().plot(color='#B40431', legend=True,figsize=(8, 5))
data.Unmetered.resample('Y').sum().plot(color='#3A2F0B', legend=True,figsize=(8, 5))
data.Sub_metering_3.resample('Y').sum().plot(color='#FFBF00', legend=True,figsize=(8, 5))
data.Sub_metering_2.resample('Y').sum().plot(color='#0080FF', legend=True,figsize=(8, 5))
data.Sub_metering_1.resample('Y').sum().plot( color='#2d7f5e', legend=True,figsize=(8, 5))
plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper left')
#plt.savefig('Anual.png', dpi = 2400)
plt.show()



a=data.Global_active_power.resample('Q').mean().plot(color='#B40431',stacked=True,legend=True,figsize=(12, 5))
b=data.Unmetered.resample('Q').mean().plot(color='#3A2F0B',stacked=True,legend=True,figsize=(12, 5))
c=data.Sub_metering_3.resample('Q').mean().plot(color='#FFBF00',stacked=True,legend=True,figsize=(12, 5))
d=data.Sub_metering_2.resample('Q').mean().plot(color='#0080FF',stacked=True,legend=True,figsize=(12, 5))
e=data.Sub_metering_1.resample('Q').mean().plot( color='#2d7f5e',stacked=True,legend=True,figsize=(12, 5))
plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper right')
plt.xlabel("")
plt.ylabel("K Watt/h")
#plt.savefig('Seasonal.png', dpi = 2400)
plt.show()


a=data.Global_active_power.resample('M').mean().plot(color='#B40431',stacked=True,figsize=(12, 5))
b=data.Unmetered.resample('M').mean().plot(color='#3A2F0B',stacked=True,figsize=(12, 5))
c=data.Sub_metering_3.resample('M').mean().plot(color='#FFBF00',stacked=True,figsize=(12, 5))
d=data.Sub_metering_2.resample('M').mean().plot(color='#0080FF',stacked=True,figsize=(12, 5))
e=data.Sub_metering_1.resample('M').mean().plot( color='#2d7f5e',stacked=True,figsize=(12, 5))
plt.xlabel("")
#plt.savefig('Monthly.png', dpi = 2400)
plt.show()

data_sorted_max =data.resample('M').mean()
Max=data_sorted_max.sort_values(by='Global_active_power', ascending=False, na_position='first')
Max.head()

data_sorted_min =data.resample('M').mean()
Min=data_sorted_min.sort_values(by='Global_active_power')
Min.head()


week_min=data['2008-08-18':'2008-08-24']
day_min=data['2008-08-23':'2008-08-23']
week_max=data['2007-12-24':'2007-12-30']
day_max=data['2007-12-28':'2007-12-28']

week_min.Global_active_power.resample('h').mean().plot(color='#B40431',stacked=True,figsize=(120, 5))
week_min.Unmetered.resample('h').mean().plot(color='#3A2F0B',stacked=True,figsize=(12, 5))
week_min.Sub_metering_3.resample('h').mean().plot(color='#FFBF00',stacked=True,figsize=(12, 5))
week_min.Sub_metering_2.resample('h').mean().plot(color='#0080FF',stacked=True,figsize=(12, 5))
week_min.Sub_metering_1.resample('h').mean().plot(color='#2d7f5e',stacked=True,figsize=(12, 5))
plt.xlabel('18Aug-24Aug 2008')
#plt.savefig('Min-Weekly.png', dpi = 2400)
plt.show()


day_min.Global_active_power.resample('h').mean().plot(color='#B40431',stacked=True,figsize=(12, 5))
day_min.Unmetered.resample('h').mean().plot(color='#3A2F0B',stacked=True,figsize=(12, 5))
day_min.Sub_metering_3.resample('h').mean().plot(color='#FFBF00',stacked=True,figsize=(12, 5))
day_min.Sub_metering_2.resample('h').mean().plot(color='#0080FF',stacked=True,figsize=(12, 5))
day_min.Sub_metering_1.resample('h').mean().plot(color='#2d7f5e',stacked=True,figsize=(12, 5))
plt.xlabel('23-Aug 2008')
#plt.savefig('Min-Daily.png', dpi = 2400)
plt.show()

week_max.Global_active_power.resample('h').mean().plot(color='#B40431',stacked=True,figsize=(12, 5))
week_max.Unmetered.resample('h').mean().plot(color='#3A2F0B',stacked=True,figsize=(12, 5))
week_max.Sub_metering_3.resample('h').mean().plot(color='#FFBF00',stacked=True,figsize=(12, 5))
week_max.Sub_metering_2.resample('h').mean().plot(color='#0080FF',stacked=True,figsize=(12, 5))
week_max.Sub_metering_1.resample('h').mean().plot(color='#2d7f5e',stacked=True,figsize=(12, 5))
plt.xlabel('24-Dec 30-Dec 2007')
#plt.savefig('Max-Weekly.png', dpi = 2400)
plt.show()


day_max.Global_active_power.resample('h').mean().plot(color='#B40431',stacked=True,legend=True,figsize=(12, 5))
day_max.Unmetered.resample('h').mean().plot(color='#3A2F0B',stacked=True,legend=True,figsize=(12, 5))
day_max.Sub_metering_3.resample('h').mean().plot(color='#FFBF00',stacked=True,legend=True,figsize=(12, 5))
day_max.Sub_metering_2.resample('h').mean().plot(color='#0080FF',stacked=True,legend=True,figsize=(12, 5))
day_max.Sub_metering_1.resample('h').mean().plot(color='#2d7f5e',stacked=True,legend=True,figsize=(12, 5))
plt.legend(['GAP', 'Unmetered', 'WhA', 'Laundry.R', 'Kitchen'], loc='upper left')
plt.xlabel('28-Dec 2007')
#plt.savefig('Max-Daily.png', dpi = 2400)
plt.show()

##Decomposition

Series= data['Global_active_power'].resample('W').mean()
decomposition = sm.tsa.seasonal_decompose(Series,freq=50, model='additive')
fig = decomposition.plot()
plt.show()


dataset['Global_active_power']=dataset['Global_active_power']*0.06



### Find ARIMA parameteres p,d,q

p = range(1,9)
d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


for param in pdq:
     for param_seasonal in seasonal_pdq:
        try:
             model = sm.tsa.statespace.SARIMAX(Series,
                                             order=param,
                                             seasonal_order=param_seasonal,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
             results = model.fit()
             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
         except:
             continue

###Model ARIMA with (7,1,1)

model = sm.tsa.statespace.SARIMAX(Series,
                                order=(7, 1, 1),
                                seasonal_order=(2, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = model.fit()
print(results.summary().tables[1])


# results.plot_diagnostics(figsize=(16, 8),lags=10)
# plt.show()

###forcasts Validation

pred = results.get_prediction(start=pd.to_datetime('2009-12-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = Series['2007':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.25)
ax.set_xlabel('Date')
ax.set_ylabel('Average Global Active Power')
#plt.savefig('Validating forecasts.png', dpi = 1200)
plt.legend()

###MSE  
Series_forecasted = pred.predicted_mean
Series_truth = Series['2006-12-31':]
mse = ((Series_forecasted - Series_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

### RMSE
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))



###Prediction & Visualisation

pred_uc = results.get_forecast(steps=70)
pred_ci = pred_uc.conf_int()
ax = Series.plot(label='observed', figsize=(14, 5))
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

