#!/usr/bin/env python
# coding: utf-8

# In[60]:


import nasdaqdatalink
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression 

nasdaqdatalink.ApiConfig.api_key = "WrMbPV1eWZ7imYrBxX-P"

mydata = nasdaqdatalink.get("WIKI/TWTR.11")
# mydata = mydata[['Adj. CLose']]
mydata.head()



# In[45]:


mydata["Adj. Close"].plot(figsize=(15,6), color='g')
plt.legend(loc='upper left')
plt.show()


# In[50]:


forecast = 30
mydata['Prediction'] = mydata[["Adj. Close"]].shift(-forecast)
mydata

x = np.array(mydata.drop(['Prediction'], 1))
x = preprocessing.scale(x)

x_forecast = x[-forecast:]
x = x[:-forecast]

y = np.array(mydata['Prediction'])
y = y[:-forecast]


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

clf = LinearRegression()  # find the line of best fit (minimised distance)
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)  # to ensure that the data is correct
# the closer to 1, the better it is

forecast_predicted = clf.predict(x_forecast)
print(forecast_predicted)


# In[61]:


dates = pd.date_range(start="2018-03-28", end="2018-04-26")
plt.plot(dates, forecast_predicted, color='y')
mydata['Adj. Close'].plot(color='g')
plt.xlim(xmin=datetime.date(2017,4,26))


# In[ ]:




