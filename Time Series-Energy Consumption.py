# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 21:12:53 2018

@author: Saeed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

data= pd.read_table('household_power_consumption.txt',
                    sep=';', parse_dates={'dt' : ['Date', 'Time']},
                    infer_datetime_format=True, low_memory=False, index_col='dt')
data.head()
data["Global_active_power"] = data.Global_active_power.convert_objects(convert_numeric=True)
data["Global_reactive_power"] = data.Global_reactive_power.convert_objects(convert_numeric=True)
data["Voltage"] = data.Voltage.convert_objects(convert_numeric=True)
data["Global_intensity"] = data.Global_intensity.convert_objects(convert_numeric=True)
data["Sub_metering_1"] = data.Sub_metering_1.convert_objects(convert_numeric=True)
data["Sub_metering_2"] = data.Sub_metering_2.convert_objects(convert_numeric=True)
data.Global_active_power=(data.Global_active_power*1000)/60
data.Global_reactive_power=(data.Global_reactive_power*1000)/60
data["Sub_metering_4"]= data.Global_active_power - data.Sub_metering_1 - data.Sub_metering_2 - data.Sub_metering_3
data.head()
decimals = pd.Series([2,2], index=['Global_active_power', 'Sub_metering_4'])
data.round(decimals)
data.info()
data.isnull().sum()
data=data.dropna(how="any")
data.Global_active_power.resample('D').mean().plot(title='Daily Global Power', color='gray')
data['Global_active_power'].resample('M').mean().plot(kind='bar',color='blue',subplots = True)
data['Sub_metering_4'].resample('M').mean().plot(kind='bar',color='r',subplots = True)
#data['Sub_metering_1'].resample('M').mean().plot(kind='bar',color='red',subplots = True)

data.Global_active_power.resample('Y').sum().plot(color='b', legend=True)
data.Sub_metering_1.resample('Y').sum().plot( color='purple', legend=True)
data.Sub_metering_2.resample('Y').sum().plot(color='y', legend=True)
data.Sub_metering_3.resample('Y').sum().plot(color='r', legend=True)
data.Sub_metering_4.resample('Y').sum().plot(color='black', legend=True)
#data.Global_intensity.resample('Y').sum().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.show()

data.Global_active_power.resample('Q').mean().plot(color='b', legend=True)
data.Sub_metering_1.resample('Q').mean().plot( color='purple', legend=True)
data.Sub_metering_2.resample('Q').mean().plot(color='y', legend=True)
data.Sub_metering_3.resample('Q').mean().plot(color='r', legend=True)
data.Sub_metering_4.resample('Q').mean().plot(color='black', legend=True)
#data.Global_intensity.resample('M').mean().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.show()

data.Global_active_power.resample('M').sum().plot(color='b', legend=True)
data.Sub_metering_1.resample('M').sum().plot( color='purple', legend=True)
data.Sub_metering_2.resample('M').sum().plot(color='y', legend=True)
data.Sub_metering_3.resample('M').sum().plot(color='r', legend=True)
data.Sub_metering_4.resample('M').sum().plot(color='black', legend=True)
#data.Global_intensity.resample('M').mean().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.show()

data_sorted =data.sort_values(by='Global_active_power', ascending=False, na_position='first')
data_sorted.head()









