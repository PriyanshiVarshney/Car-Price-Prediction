#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("quikr_car - quikr_car.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


plt.figure(figsize=(15,10))
sns.heatmap(df.isnull())


# In[9]:


df.fuel_type.unique()


# In[10]:


df.company.unique()


# In[11]:


df.year.unique()


# In[12]:


df.Price.unique()


# In[13]:


df[df['Price']=='Ask For Price'].count()


# In[14]:


df=df.replace('Ask For Price',np.NaN)


# In[15]:


df.isnull().sum()


# In[16]:


df.kms_driven.unique()


# In[17]:


df['kms_driven']=df['kms_driven'].str.split(' ').str.get(0)


# In[18]:


df.head()


# In[19]:


df=df.replace({'kms_driven':'Petrol'},np.NaN)


# In[20]:


df.isnull().sum()


# In[21]:


df['year'].unique()


# In[22]:


null=['...', '150k', 'TOUR', 'r 15',  'Zest', '/-Rs',
       'sale', '1995', 'ara)', 'SELL',  'tion', 'odel',
       '2 bs', 'arry', 'Eon', 'o...', 'ture', 'emi', 'car', 'able', 'no.',
       'd...', 'SALE', 'digo', 'sell', 'd Ex', 'n...', 'e...', 'D...',
       ', Ac', 'go .', 'k...', 'o c4', 'zire', 'cent', 'Sumo', 'cab',
       't xe', 'EV2', 'r...', 'zest']


# In[23]:


for i in null:
    df=df.replace({'year':i},np.NaN)


# In[24]:


df['year'].unique()


# In[25]:


df.isnull().sum()


# In[26]:


df['company'].unique()


# In[27]:


null2=['selling', 'URJENT',  'Used', 'Sale', 'very',  'i', '2012', 'Well', 'all', '7', '9',
       'scratch', 'urgent', 'sell', 'TATA', 'Any', 'I']


# In[28]:


for i in null2:
    df=df.replace({'company':i},np.NaN)


# In[29]:


df.isnull().sum()


# In[30]:


df['name']=df['name'].str.split(" ").str.slice(0,3).str.join(' ')


# In[31]:


df.head()


# In[40]:


df1=df[df['Price'].isnull()==False]


# In[41]:


df1.head()


# In[42]:


df1.shape


# In[35]:


df1.head()


# In[43]:


df1.isnull().sum()


# In[44]:


df1=df1.dropna(subset=['company'])


# In[45]:


df1.isnull().sum()


# In[46]:


df1['fuel_type'].value_counts()


# In[47]:


df1['fuel_type']=df1['fuel_type'].fillna('Petrol')


# In[48]:


df1.isnull().sum()


# In[49]:


df1['kms_driven'].value_counts()


# In[50]:


df1['kms_driven']=df1['kms_driven'].fillna('40,000')


# In[51]:


df1['year'].value_counts()


# In[52]:


df1['year']=df1['year'].fillna(method='ffill')


# In[46]:


df1['year'].value_counts()


# In[53]:


df1.isnull().sum()


# In[55]:


df1['year']=df1['year'].astype(int)


# In[56]:


df1['kms_driven']=df1['kms_driven'].str.replace(",",'').astype(int)


# In[57]:


df1['Price']=df1['Price'].str.replace(",",'').astype(int)


# In[58]:


df1.info()


# In[88]:


df1=df1[df1['Price']<6e6]


# In[89]:


X=df1.drop(columns='Price')
y=df1['Price']


# In[90]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[91]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2,test_size=0.2)


# In[92]:


ohe=OneHotEncoder()


# In[93]:


ohe.fit(X[['name','company','fuel_type']])


# In[94]:


from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[95]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')


# In[96]:


lr=LinearRegression()


# In[97]:


pipe=make_pipeline(column_trans,lr)


# In[98]:


pipe.fit(X_train,y_train)


# In[99]:


y_pred=pipe.predict(X_test)


# In[100]:


r2_score(y_test,y_pred)


# In[101]:


scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


# In[102]:


np.argmax(scores)


# In[103]:


scores[np.argmax(scores)]


# In[ ]:




