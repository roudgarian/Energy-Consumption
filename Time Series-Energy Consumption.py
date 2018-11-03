
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


# In[4]:


data= pd.read_table('household_power_consumption.txt', sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, index_col='dt')


# In[5]:


data=data['2007-01-01':'2010-11-27']


# In[6]:


data["Global_active_power"] = data.Global_active_power.convert_objects(convert_numeric=True)
data["Global_reactive_power"] = data.Global_reactive_power.convert_objects(convert_numeric=True)
data["Voltage"] = data.Voltage.convert_objects(convert_numeric=True)
data["Global_intensity"] = data.Global_intensity.convert_objects(convert_numeric=True)
data["Sub_metering_1"] = data.Sub_metering_1.convert_objects(convert_numeric=True)
data["Sub_metering_2"] = data.Sub_metering_2.convert_objects(convert_numeric=True)


# In[7]:


data.Global_active_power=(data.Global_active_power*1000)/60


# In[8]:


data.Global_reactive_power=(data.Global_reactive_power*1000)/60


# In[9]:


data["Sub_metering_4"]= data.Global_active_power - data.Sub_metering_1 - data.Sub_metering_2 - data.Sub_metering_3


# In[10]:


data.head()


# In[12]:


decimals = pd.Series([2,2,2], index=['Global_active_power','Global_reactive_power', 'Sub_metering_4'])
data.round(decimals)


# In[13]:


data.info()


# In[14]:


data.isnull().sum()


# In[15]:


data=data.dropna(how='any')


# In[16]:


data.Global_active_power.resample('D').mean().plot(title='Daily Global Power', color='gray')


# In[17]:


data['Global_active_power'].resample('M').mean().plot(kind='bar',color='blue',subplots = True)


# In[18]:


data['Sub_metering_4'].resample('M').mean().plot(kind='bar',color='r',subplots = True)
#data['Sub_metering_1'].resample('M').mean().plot(kind='bar',color='red',subplots = True)


# In[50]:


data_sorted =data.resample('Y').mean()
we=data_sorted.sort_values(by='Global_active_power', ascending=False, na_position='first')
we.head()


# In[51]:


data.Global_active_power.resample('Y').mean().plot(color='b', legend=True)
data.Sub_metering_1.resample('Y').mean().plot( color='purple', legend=True)
data.Sub_metering_2.resample('Y').mean().plot(color='y', legend=True)
data.Sub_metering_3.resample('Y').mean().plot(color='r', legend=True)
data.Sub_metering_4.resample('Y').mean().plot(color='black', legend=True)
#data.Global_intensity.resample('Y').sum().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.show()


# In[52]:


data_sorted =data.resample('Q').mean()
we=data_sorted.sort_values(by='Global_active_power', ascending=False, na_position='first')
we.head()


# In[48]:


data.Global_active_power.resample('Q').mean().plot(color='b', legend=True)
data.Sub_metering_1.resample('Q').mean().plot( color='purple', legend=True)
data.Sub_metering_2.resample('Q').mean().plot(color='y', legend=True)
data.Sub_metering_3.resample('Q').mean().plot(color='r', legend=True)
data.Sub_metering_4.resample('Q').mean().plot(color='black', legend=True)
#data.Global_intensity.resample('M').mean().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.show()


# In[53]:


data_sorted =data.resample('M').mean()
we=data_sorted.sort_values(by='Global_active_power', ascending=False, na_position='first')
we.head()


# In[54]:


data.Global_active_power.resample('M').mean().plot(color='b', legend=True)
data.Sub_metering_1.resample('M').mean().plot( color='purple', legend=True)
data.Sub_metering_2.resample('M').mean().plot(color='y', legend=True)
data.Sub_metering_3.resample('M').mean().plot(color='r', legend=True)
data.Sub_metering_4.resample('M').mean().plot(color='black', legend=True)
#data.Global_intensity.resample('M').mean().plot(color='g', legend=True)
#data.Voltage.resample('M').mean().plot(color='g', legend=True)
plt.show()


# In[37]:


#we.plot(color='blue',legend=True)


# In[ ]:


#sns.heatmap(data, xticklabels=data.Global_active_power, yticklabels=data.Sub_metering_4, vmin=-1, vmax=1)
#plt.show() 

