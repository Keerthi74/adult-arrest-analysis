#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = pd.read_csv('/Users/kc/Desktop/Sem 1/AIT580/Adult_Arrests_18_and_Older_by_County___Beginning_1970.csv')

df = pd.DataFrame(data)


# In[2]:


df


# In[3]:


print("Missing values distribution: ")
print(df.isnull().mean())
print("")


# In[4]:


print(df.isnull().sum())


# In[5]:


print("Column datatypes: ")
print(df.dtypes)


# In[6]:


df.head(10)


# In[7]:


# importing package
import matplotlib.pyplot as plt
import numpy as np


# In[8]:


df2 = df.groupby(['County'])[['Felony Total','Misdemeanor Total']].sum()


# In[9]:


df2


# In[10]:


# plot lines
df2.plot(kind='line')


# In[35]:


df3 = df.groupby(['Year'])[['Felony Total','Misdemeanor Total']].sum()
df3


# In[36]:


df3.plot(kind='line')


# In[32]:


df.hist(bins=50, figsize=(20,15))

plt.show()


# In[25]:


import pandas
from sklearn import linear_model

df = pandas.read_csv("/Users/kc/Desktop/Sem 1/AIT580/Adult_Arrests_18_and_Older_by_County___Beginning_1970.csv")

X = df[['Year']]
y = df['Felony Total']


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression 

regressor = LinearRegression() 

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[33]:


plt.scatter(X_train, y_train,color='g') 

plt.plot(X_test, y_pred,color='k') 

plt.xlabel('Year') 
plt.ylabel('Felony cases') 
  
# displaying the title
plt.title("Regression")

plt.show()


# In[20]:


print (X,y)


# In[27]:


predicted22 = regressor.predict([[2022]])

print(predicted22)


# In[ ]:




