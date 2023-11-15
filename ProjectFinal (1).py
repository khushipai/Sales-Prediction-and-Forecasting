#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import r2_score


# In[2]:


df_train= pd.read_csv(r'C:\Users\SyedMoimn\Desktop\FinalYearProj\SalesForecasting\Train.csv')
df_test= pd.read_csv(r'C:\Users\SyedMoimn\Desktop\FinalYearProj\SalesForecasting\Test.csv')


# In[3]:


df_train.head()


# In[4]:


df_train.shape


# In[5]:


df_train.isnull().sum()


# In[6]:


df_test.isnull().sum()


# In[7]:


df_train.info()


# In[8]:


df_train.describe()


# Item_Weight is (numerical column) filled with Mean Imputation

# In[9]:


df_train['Item_Weight'].describe()


# In[10]:


df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(),inplace=True)
df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean(),inplace=True)


# In[11]:


df_train.isnull().sum()


# In[12]:


df_train['Outlet_Size'].fillna(df_train['Outlet_Size'].mode()[0],inplace=True)
df_test['Outlet_Size'].fillna(df_test['Outlet_Size'].mode()[0],inplace=True)


# In[13]:


df_train.isnull().sum()


# In[14]:


df_test.isnull().sum()


# Drop item_identifier and outlet identifier

# In[15]:


df_train.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
df_test.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)


# In[16]:


df_train


# # Label encoding

# In[17]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[18]:


df_train['Item_Fat_Content']= le.fit_transform(df_train['Item_Fat_Content'])
df_train['Item_Type']= le.fit_transform(df_train['Item_Type'])
df_train['Outlet_Size']= le.fit_transform(df_train['Outlet_Size'])
df_train['Outlet_Location_Type']= le.fit_transform(df_train['Outlet_Location_Type'])
df_train['Outlet_Type']= le.fit_transform(df_train['Outlet_Type'])


# In[19]:


df_train


# Splitting data
# 

# In[20]:


X=df_train.drop('Item_Outlet_Sales',axis=1)
Y=df_train['Item_Outlet_Sales']


# In[21]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=101, test_size=0.2)


# Standardization

# In[22]:


X.describe()


# In[23]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()


# In[24]:


X_train_std= sc.fit_transform(X_train)
X_test_std= sc.transform(X_test)


# In[25]:


X_train_std


# In[26]:


X_test_std


# In[27]:


import joblib


# In[28]:


joblib.dump(sc,r'C:\Users\SyedMoimn\Desktop\FinalYearProj\SalesForecasting\Model\sc.sav')


# Model building

# Linear Regression-50.4%

# In[29]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()


# In[30]:


lr.fit(X_train_std,Y_train)


# In[31]:


X_test.head()


# In[32]:


Y_pred_lr=lr.predict(X_test_std)


# In[33]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print(r2_score(Y_test,Y_pred_lr))
print(mean_absolute_error(Y_test,Y_pred_lr))
print(np.sqrt(mean_squared_error(Y_test,Y_pred_lr)))


# In[34]:


joblib.dump(lr,r'C:\Users\SyedMoimn\Desktop\FinalYearProj\SalesForecasting\Model\lr.sav')


# Random Forest-54.9%

# In[35]:


from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=1000)
rf.fit(X_train_std,Y_train)


# In[36]:


Y_pred_rf= rf.predict(X_test_std)


# In[37]:


print(r2_score(Y_test,Y_pred_rf))
print(mean_absolute_error(Y_test,Y_pred_rf))
print(np.sqrt(mean_squared_error(Y_test,Y_pred_rf)))


# In[47]:


joblib.dump(rf,r'C:\Users\SyedMoimn\Desktop\FinalYearProj\SalesForecasting\Model\rf.sav')


# XGBooster-

# In[38]:


from xgboost import XGBRegressor


# In[39]:


regressor = XGBRegressor()


# In[40]:


regressor.fit(X_train, Y_train)


# In[41]:


# prediction on training data
training_data_prediction = regressor.predict(X_train)


# In[42]:


# R squared Value
r2_train = metrics.r2_score(Y_train, training_data_prediction)


# In[43]:


print('R Squared value = ', r2_train)


# In[44]:


# prediction on test data
test_data_prediction = regressor.predict(X_test)


# In[45]:


# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)


# In[46]:


print('R Squared value = ', r2_test)
mse = mean_squared_error(Y_test, test_data_prediction)
print("MSE:",mse)


# In[48]:


joblib.dump(regressor,r'C:\Users\SyedMoimn\Desktop\FinalYearProj\SalesForecasting\Model\regressor.sav')


# In[49]:


import joblib


# In[50]:


model=joblib.load(r'C:\Users\SyedMoimn\Desktop\FinalYearProj\SalesForecasting\Model\regressor.sav')


# In[ ]:




