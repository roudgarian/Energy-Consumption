
# coding: utf-8

# In[]:

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


# In[]:


data= pd.read_table('household_power_consumption.txt', sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, index_col='dt')


# In[]:


data=data['2007-01-01':'2010-11-27']


# In[]:


data["Global_active_power"] = data.Global_active_power.convert_objects(convert_numeric=True)
data["Global_reactive_power"] = data.Global_reactive_power.convert_objects(convert_numeric=True)
data["Voltage"] = data.Voltage.convert_objects(convert_numeric=True)
data["Global_intensity"] = data.Global_intensity.convert_objects(convert_numeric=True)
data["Sub_metering_1"] = data.Sub_metering_1.convert_objects(convert_numeric=True)
data["Sub_metering_2"] = data.Sub_metering_2.convert_objects(convert_numeric=True)


# In[]:


data.Global_active_power=(data.Global_active_power*1000)/60


# In[]:


data.Global_reactive_power=(data.Global_reactive_power*1000)/60


# In[]:


data["Sub_metering_4"]= data.Global_active_power - data.Sub_metering_1 - data.Sub_metering_2 - data.Sub_metering_3


# In[]:


data.head()


# In[]:


decimals = pd.Series([2,2,2], index=['Global_active_power','Global_reactive_power', 'Sub_metering_4'])
data.round(decimals)


# In[]:


data.info()


# In[]:


data.isnull().sum()


# In[]:


data=data.dropna(how='any')


# In[]:


data.Global_active_power.resample('D').mean().plot(title='Daily Global Power', color='gray')


# In[]:


data['Global_active_power'].resample('M').mean().plot(kind='bar',color='blue',subplots = True)


# In[]:


data['Sub_metering_4'].resample('M').mean().plot(kind='bar',color='r',subplots = True)
#data['Sub_metering_1'].resample('M').mean().plot(kind='bar',color='red',subplots = True)


# In[]:


data_sorted =data.resample('Y').mean()
we=data_sorted.sort_values(by='Global_active_power', ascending=False, na_position='first')
we.head()


# In[]:


data.Global_active_power.resample('Y').mean().plot(color='b', legend=True)
data.Sub_metering_1.resample('Y').mean().plot( color='purple', legend=True)
data.Sub_metering_2.resample('Y').mean().plot(color='y', legend=True)
data.Sub_metering_3.resample('Y').mean().plot(color='r', legend=True)
data.Sub_metering_4.resample('Y').mean().plot(color='black', legend=True)
#data.Global_intensity.resample('Y').sum().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.show()


# In[]:


data.Global_active_power.resample('Q').mean().plot(color='b', legend=True)
data.Sub_metering_1.resample('Q').mean().plot( color='purple', legend=True)
data.Sub_metering_2.resample('Q').mean().plot(color='y', legend=True)
data.Sub_metering_3.resample('Q').mean().plot(color='r', legend=True)
data.Sub_metering_4.resample('Q').mean().plot(color='black', legend=True)
#data.Global_intensity.resample('M').mean().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.show()


# In[]:


data.Global_active_power.resample('M').mean().plot(color='b', legend=True)
data.Sub_metering_1.resample('M').mean().plot( color='purple', legend=True)
data.Sub_metering_2.resample('M').mean().plot(color='y', legend=True)
data.Sub_metering_3.resample('M').mean().plot(color='r', legend=True)
data.Sub_metering_4.resample('M').mean().plot(color='black', legend=True)
#data.Global_intensity.resample('M').mean().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.show()


# In[]:


#we.plot(color='blue',legend=True)


# In[]:


#sns.heatmap(data, xticklabels=data.Global_active_power, yticklabels=data.Sub_metering_4, vmin=-1, vmax=1)
#plt.show() 


# In[]:


data_2007=data['2007-01-01':'2007-12-31']
data_2008=data['2008-01-01':'2008-12-31']
data_2009=data['2009-01-01':'2009-12-31']
data_2010=data['2010-01-01':'2010-12-31']


# In[]:


data_sorted =data_2009.resample('M').mean()
Max=data_sorted.sort_values(by='Global_active_power', ascending=False, na_position='first')
Max.head()


# In[]:


data_sorted =data_2009.resample('M').mean()
Min=data_sorted.sort_values(by='Global_active_power', ascending=True, na_position='first')
Min.head()


# In[]:


data.Sub_metering_1.resample('M').mean().plot(legend=True)
plt.show()


# In[]:


week01=data['2009-07-20':'2009-07-26']
day01=data['2009-07-20':'2009-07-20']
week02=data['2009-01-05':'2009-01-11']
day02=data['2009-01-06':'2009-01-06']
#jan=data['2007-01-01':'2007-01-31']
#Feb=data['2007-02-01':'2007-02-28']


# In[]:


week01.Sub_metering_1.plot(legend=True)
week01.Sub_metering_2.plot(legend=True)
week01.Sub_metering_3.plot(legend=True)
week01.Sub_metering_4.plot(legend=True)
plt.xlabel('20July-27July 2009')
plt.ylabel('Kw/h')
plt.show()

# In[]:


day01.Sub_metering_1.plot(legend=True)
day01.Sub_metering_2.plot(legend=True)
day01.Sub_metering_3.plot(legend=True)
day01.Sub_metering_4.plot(legend=True)
plt.xlabel('22-July-2009')
plt.ylabel('Kw/h')
plt.show()


# In[]:


week02.Sub_metering_1.plot(legend=True)
week02.Sub_metering_2.plot(legend=True)
week02.Sub_metering_3.plot(legend=True)
week02.Sub_metering_4.plot(legend=True)
plt.xlabel('18May-24May 2009')
plt.ylabel('Kw/h')
plt.show()

# In[]:


day02.Sub_metering_1.plot(legend=True)
day02.Sub_metering_2.plot(legend=True)
day02.Sub_metering_3.plot(legend=True)
day02.Sub_metering_4.plot(legend=True)
plt.xlabel('06-Jan-2009')
plt.ylabel('Kw/h')
plt.show()

