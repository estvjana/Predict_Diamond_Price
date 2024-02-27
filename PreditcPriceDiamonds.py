#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#get dataset names from the seaborn library
sns.get_dataset_names()


# In[2]:


# importing the Diamonds dataset
diamonds_df = sns.load_dataset('diamonds')

#printing dataset header
diamonds_df.head()


# In[3]:


diamonds_df.columns


# # Data Preprocessing

# In[4]:


#extracting features
X = diamonds_df.drop(['price'], axis =1)

#extracting labels
y = diamonds_df['price']


# In[5]:


X.head()


# In[6]:


y.head()


#  *** NUMERICAL DATA ***

# In[7]:


numerical = X.drop(['cut','color', 'clarity'], axis = 1)


# In[8]:


numerical.head()


# *** CATEGORICAL DATA ***

# In[9]:


categorical = X.filter(['cut', 'color','clarity'])


# In[10]:


categorical.head()


# # Convering Categorical Data to Numberical
# 

# # One - Hot Encoding (OneHotEncoder) 

# In[11]:


from sklearn.preprocessing import OneHotEncoder
skencoder = OneHotEncoder(handle_unknown='ignore', sparse_output = False).set_output(transform = 'pandas')


# # Filter and Transforming to One - hot form 

# In[12]:


cat_numerical = skencoder.fit_transform(categorical)
cat_numerical.head()


# In[13]:


#These are the levels of each categorical variable
skencoder.categories_


# In[14]:


X = pd.concat([numerical, cat_numerical], axis = 1)
X.head()


# # Divide Data into Training and Test Datasets

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Scale both the training and the test datasets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Regressions Models 

# # KNN Regression

# In[16]:


from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors= 5)

regressor = knn_reg.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # Linear Regression

# In[17]:


#import linear regression model from sklearn
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

#training the model
regressor = lin_reg.fit(X_train, y_train)

#making predictions on the test set
y_pred = regressor.predict(X_test)

#evaluating model performance
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # Random Forest Regression. The best model!

# In[18]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=42, n_estimators=50)


#training the model
regressor = rf_reg.fit(X_train, y_train)

#making predicitons on the test set
y_pred = regressor.predict(X_test)


#evaluating the model performance
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # Predicting a Single Value

# In[19]:


diamonds_df.head()


# In[20]:


diamonds_df.loc[100]


# In[27]:


from sklearn.preprocessing import StandardScaler 

single_input = {"carat":[0.25] , "cut":["Premium"],"color":["J"], 
              "clarity":["SI2"],  "depth":[56.7], "table":[57.0],
                "x":[4.01], "y":[4.05], "z":[2.35]}

single_input=pd.DataFrame.from_dict(single_input) 
single_input


# In[28]:


#One-Hot Encode the input

cat_input = single_input.filter(['cut', 'color', 'clarity'])
cat_input


# In[29]:


cat_input = skencoder.transform(cat_input)
cat_input


# In[32]:


num_input = single_input.drop(['cut', 'color', 'clarity'], axis= 1)
num_input


# In[33]:


X_new = pd.concat([num_input, cat_input], axis = 1)
X_new.head()


# In[34]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = X_new = RandomForestRegressor(random_state=42, n_estimators=50)
regressor = rf_reg.fit(X_train, y_train)
single_input = sc.transform (X.values[100].reshape(1, -1))
predicted_diamonds = regressor.predict(single_input)
print(predicted_diamonds)

