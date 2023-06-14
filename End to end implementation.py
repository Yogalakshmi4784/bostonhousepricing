#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# # Lets load the boston house pricing dataset

# In[8]:


from sklearn.datasets import load_boston


# In[9]:


boston=load_boston()


# In[10]:


type(boston)


# In[11]:


boston.keys()


# # Lets check the DESCR of the dataset to begin with

# In[12]:


print(boston.DESCR)


# In[13]:


print(boston.data)


# In[14]:


print(boston.target)


# In[15]:


print(boston.feature_names)


# # Preparing the dataset

# In[16]:


dataset=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[17]:


dataset


# In[18]:


dataset['Price']=boston.target


# In[19]:


dataset


# In[20]:


dataset.info()


# # Summarix=zing the stats of the data

# In[21]:


dataset.describe()


# # Check the missing values

# In[22]:


dataset.isnull().sum()


# # Exploratory data analysis-Correlation is super important with any regression

# # Check for multicollinearity 
# 

# In[23]:


dataset.corr()


# In[24]:


import seaborn as sns


# In[25]:


sns.pairplot(dataset)


# In[26]:


plt.scatter(dataset['CRIM'],dataset['Price'])
plt.xlabel('Crime rate')
plt.ylabel('Price')


# In[27]:


plt.scatter(dataset['RM'],dataset['Price'])
plt.xlabel('Rooms')
plt.ylabel('Price')


# In[28]:


import seaborn as sns
sns.regplot(x='RM',y='Price',data=dataset)


# In[29]:


import seaborn as sns
sns.regplot(x='LSTAT',y='Price',data=dataset)


# In[30]:


import seaborn as sns
sns.regplot(x='CHAS',y='Price',data=dataset)


# In[31]:


import seaborn as sns
sns.regplot(x='PTRATIO',y='Price',data=dataset)


# In[32]:


dataset


# In[33]:


y=dataset['Price']


# In[34]:


x=dataset.iloc[:,:-1]
x


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[36]:


X_train


# In[37]:


y_train


# # Standard Scaling

# In[38]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[39]:


X_train=scaler.fit_transform(X_train)


# In[40]:


X_test=scaler.transform(X_test)


# In[41]:
import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))

X_train


# In[42]:


X_test


# In[43]:


from sklearn.linear_model import LinearRegression
regression=LinearRegression()


# In[44]:


regression.fit(X_train,y_train)


# In[45]:


## print the co-efficients and the intercepts

print(regression.coef_)


# In[46]:


print(regression.intercept_)


# In[47]:


## on whch parameter the model has been trained

regression.get_params()


# In[48]:


###prediction of the model

reg_pred=regression.predict(X_test)


# In[49]:


reg_pred


# # Assumptions

# In[51]:


## plot a scatter plot fpr the prediction

plt.scatter(y_test,reg_pred)


# In[52]:


##prediction w.r.t. residuals

residuals=y_test-reg_pred


# In[53]:


residuals


# In[54]:


## plot the residuals


# In[55]:


sns.displot(residuals,kind="kde")


# In[57]:


## scatterplot w.r.t residuals and prediction
#uniform distribution
plt.scatter(reg_pred,residuals)


# In[59]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# # R square and adjusted R square

# In[61]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# # New Data Prediction

# In[69]:


boston.data[0].reshape(1,-1).shape


# In[70]:


boston.data[0].reshape(1,-1)


# In[72]:


##transformation of new data
scaler.transform(boston.data[0].reshape(1,-1))


# In[73]:


regression.predict(scaler.transform(boston.data[0].reshape(1,-1)))


# # Pickling the model file for deployment

# In[74]:


import pickle


# In[75]:


pickle.dump(regression,open('regmodel.pkl','wb'))


# In[76]:


pickled_model=pickle.load(open('regmodel.pkl','rb'))


# In[77]:


pickled_model.predict(scaler.transform(boston.data[0].reshape(1,-1)))


# In[ ]:




