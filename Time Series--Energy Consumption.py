
# coding: utf-8

# In[176]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
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


# In[177]:


data= pd.read_table('household_power_consumption.txt', sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, index_col='dt')


# # preprocessing

# In[178]:


data.isnull().sum().sum()


# In[179]:


data.replace('?', nan, inplace=True)
data = data.astype('float64')


# In[180]:


def fill_missing(values):
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - 1, col]


# In[181]:


fill_missing(data.values)


# In[182]:


#data=data['2007-01-01':'2010-11-21']

#data = data.astype('float64')
#data["Global_active_power"] = pd.to_numeric(data["Global_active_power"], errors='coerce')
#data["Global_reactive_power"] = pd.to_numeric(data["Global_reactive_power"], errors='coerce')
#data["Voltage"] = pd.to_numeric(data["Voltage"], errors='coerce')
#data["Global_intensity"] = pd.to_numeric(data["Global_intensity"], errors='coerce')
#data["Sub_metering_1"] = pd.to_numeric(data["Sub_metering_1"], errors='coerce')
#data["Sub_metering_2"] = pd.to_numeric(data["Sub_metering_2"], errors='coerce')
# In[183]:


data.Global_active_power=(data.Global_active_power*1000)/60

data.Global_reactive_power=(data.Global_reactive_power*1000)/60
# In[184]:


data["Unmetered"]= data.Global_active_power - data.Sub_metering_1 - data.Sub_metering_2 - data.Sub_metering_3

decimals = pd.Series([2,2,2], index=['Global_active_power','Global_reactive_power', 'Unmetered'])
data.round(decimals)
# In[185]:


data.info()


# In[186]:


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
# In[187]:


data.to_csv('Power_consumption_Preprocessed.csv')


# # Investigation and Visualization

# In[188]:


#checkpoint to back here without doing data-preprocess step
data = read_csv('Power_consumption_Preprocessed.csv', header=0,
                infer_datetime_format=True, parse_dates=['dt'], index_col=['dt'])


# In[189]:


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


# In[190]:


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


# In[191]:


#data_sorted =data.resample('M').mean()
#month=data_sorted.sort_values(by='Global_active_power', ascending=False, na_position='first')
#data_sorted.head(50)
#data_sorted.to_excel('exampleResult.xls', sheet_name='Sheet1')


# In[192]:


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


# In[193]:


#we.plot(color='blue',legend=True)


# In[194]:


#sns.heatmap(data, xticklabels=data.Global_active_power, yticklabels=data.Unmetered, vmin=-1, vmax=1)
#plt.show() 


# In[195]:


data_2007=data['2007-01-01':'2007-12-31']
data_2008=data['2008-01-01':'2008-12-31']
data_2009=data['2009-01-01':'2009-12-31']
data_2010=data['2010-01-01':'2010-12-31']


# In[196]:


data_sorted_max =data.resample('M').mean()
Max=data_sorted_max.sort_values(by='Global_active_power', ascending=False, na_position='first')
Max.head()


# In[197]:


data_sorted_min =data.resample('M').mean()
Min=data_sorted_min.sort_values(by='Global_active_power')
Min.head()


# In[198]:


week_min=data['2008-08-18':'2008-08-24']
day_min=data['2008-08-23':'2008-08-23']
week_max=data['2007-12-24':'2007-12-30']
day_max=data['2007-12-28':'2007-12-28']
#jan=data['2007-01-01':'2007-01-31']
#Feb=data['2007-02-01':'2007-02-28']


# In[199]:


#data_sorted_w01 =week01.resample('Min').mean()
#data_sorted_w01.head(7)
#data_sorted_w01.to_excel('data_sorted_w01.xls', sheet_name='Sheet1')


# In[200]:


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


# In[201]:


#data_sorted_d01 =day01.resample('Min').mean()
#data_sorted_w01.head(7)
#data_sorted_d01.to_excel('data_sorted_d01.xls', sheet_name='Sheet1')


# In[202]:


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


# In[203]:


#data_sorted_w02 =week02.resample('Min').mean()
#data_sorted_w01.head(7)
#data_sorted_w02.to_excel('data_sorted_w02.xls', sheet_name='Sheet1')


# In[204]:


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


# In[205]:


#data_sorted_d02 =day02.resample('Min').mean()
#data_sorted_w01.head(7)
#data_sorted_d02.to_excel('data_sorted_d02.xls', sheet_name='Sheet1')


# In[206]:


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


# In[207]:


#plt.matshow(data.corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
#plt.title('without resampling', size=10)
#plt.colorbar()
#plt.show()
#autocorrelation_plot(data)
#pyplot.show()


# In[208]:


Series= data['Global_active_power'].resample('W').mean()
decomposition = sm.tsa.seasonal_decompose(Series,freq=50, model='additive')
fig = decomposition.plot()
#plt.savefig('Trend-Seasonal.png', dpi = 1200)
#print(decomposition.trend)
#print(decomposition.seasonal)
#print(decomposition.resid)
#print(decomposition.observed)
plt.show()


# In[209]:


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores


# # Training and Test Set

# In[210]:


dataset = read_csv('Power_consumption_Preprocessed.csv',
                   header=0, infer_datetime_format=True, parse_dates=['dt'], index_col=['dt'])


# In[213]:


dataset.head()


# In[212]:


dataset.Global_active_power=(dataset.Global_active_power*60)/1000


# In[214]:


daily_data =dataset.resample('D').sum()


# In[215]:


print(daily_data.shape)
print(daily_data.head())


# In[216]:


daily_data.to_csv('Daily_Power_consumption.csv')


# In[217]:


# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[2:-334], data[-334:-5]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test


# In[218]:


dataset = read_csv('Daily_Power_consumption.csv', 
                   header=0, infer_datetime_format=True, parse_dates=['dt'], index_col=['dt'])


# In[219]:


train, test = split_dataset(dataset.values)


# In[220]:


print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])


# In[221]:


print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])


# # Model Evaluation

# In[222]:


# evaluate a single model
def evaluate_model(model_func, train, test):
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = model_func(history)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores


# In[223]:


# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))


# In[224]:


# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:, 0] for week in data]
	# flatten into a single series
	series = array(series).flatten()
	return series


# In[225]:


# load the new file
dataset = read_csv('Daily_Power_consumption.csv',
                   header=0, infer_datetime_format=True, parse_dates=['dt'], index_col=['dt'])


# In[226]:


# split into train and test
train, test = split_dataset(dataset.values)
# convert training data into a series
series = to_series(train)


# In[227]:


# plots
pyplot.figure(figsize=(15, 8))
lags = 70
# acf
axis = pyplot.subplot(2, 1, 1)
plot_acf(series, ax=axis, lags=lags,)
# pacf
axis = pyplot.subplot(2, 1, 2)
plot_pacf(series, ax=axis, lags=lags)
# show plot
pyplot.show()


# # Autoregression Model

# In[228]:


# arima forecast
def arima_forecast(history):
	# convert history into a univariate series
	series = to_series(history)
	# define the model
	model = ARIMA(series, order=(7,0,0))
	# fit the model
	model_fit = model.fit(disp=False)
	# make forecast
	yhat = model_fit.predict(len(series), len(series)+6)
	return yhat


# In[229]:


# load the new file
dataset = read_csv('Daily_Power_consumption.csv',
                   header=0, infer_datetime_format=True, parse_dates=['dt'], index_col=['dt'])


# In[230]:


# split into train and test
train, test = split_dataset(dataset.values)


# In[231]:


# define the names and functions for the models we wish to evaluate
models = dict()
models['arima'] = arima_forecast


# In[232]:


# evaluate each model
days = [ 'mon', 'tue', 'wed', 'thr', 'fri', 'sat','sun']


# In[233]:


for name, func in models.items():
	# evaluate and get scores
	score, scores = evaluate_model(func, train, test)
	# summarize scores
	summarize_scores(name, score, scores)
 


# In[234]:


# plot scores
pyplot.plot(days, scores, marker='o', label=name)
# show plot
pyplot.legend()
pyplot.show()

