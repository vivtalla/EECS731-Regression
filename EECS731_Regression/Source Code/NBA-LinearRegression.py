
# coding: utf-8

# In[2]:


get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


# In[3]:


df = pd.read_csv('nba_elo.csv')


# In[4]:


df


# The data set has many NaN Values for various fields. I am going to go ahead and convert those to zero.

# In[5]:


df.fillna(0)


# After looking at the dataset here, I want to see if we can predict the score of the first team using the other feautures of the dataset. The carm-elo columns are not filled in for data until the 2016 season, and I want to use those for prediction. Because of that I'm going to go ahead and drop the data from the seasons before 2016. The game is so different during those seasons that not including them will likely give us a better prediciton too.

# In[6]:


df.drop(df.index[:63159], inplace=True)


# In[7]:


df


# In[8]:


df.fillna(0)


# Now that we have our data, we need to split the data into our training and target variables. The target variable is score1, and the training variables will be the other features in the data set.

# In[9]:


X = df[['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'carm-elo1_post', 'carm-elo2_post', 'carm-elo_prob1', 'carm-elo_prob2', 'score2']]


# In[10]:


X


# The tail of the dataset has many NaN values. Since that portion of the dataset is incomplete, I'm going to go ahead and drop them from the training and test sets.

# In[11]:


X = X[:-1217]


# In[12]:


X


# In[13]:


Y = df[['score1']]


# In[14]:


Y = Y[:-1217]


# In[15]:


Y


# In[16]:


X = np.array(X)
Y = np.array(Y)


# Now that we have feature engineered the data to fit our needs, we can now use sklearn to run linear regression on it.

# In[17]:


from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


# For regression to work, we will need to split the data into a training and a test split given we are solving a supervised learning problem. 

# In[18]:


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 42)


# In[19]:


model = LinearRegression()


# In[20]:


model.fit(X_train, y_train)


# In[21]:


model.score(X_test, y_test)


# Our model predicts with a score of about 45 percent. This is about what I expected given the features I used to test the model. Now that we have fit the model, we are going to play around with it a little.

# In[22]:


model.coef_


# Following the linear regression equation Y = X(Beta) + c + error, these are the coefficients, beta, for the model.

# In[23]:


model.intercept_


# Above we found the intercept, c given the linear regression equation. Now we can use our model to predict unknown data and then plot it.

# In[24]:


model.predict(X_test)


# In[27]:


y_pred = model.predict(X_test) 
plt.plot(y_test, y_pred, '.')
plt.title("Score1 vs Predicted Score1: $Y_i$ vs $\hat{Y}_i$")
plt.xlabel("Score1: $Y_i$")
plt.ylabel("Predicted Score1: $\hat{Y}_i$")


# Here we can visualize how our model did. This is a graph comparing the actual score1 teams had vs the predicted score1 created by our model.
