#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import load_iris


# In[ ]:


import numpy as np 


# In[17]:


iris = load_iris() 


# In[18]:


print(iris)


# In[19]:


iris.feature_names


# In[20]:


print(iris.DESCR) 


# In[21]:


from sklearn.utils import shuffle


# In[22]:


X =iris.data 


# In[23]:


Y = iris.target


# In[24]:


x,y = shuffle(X,Y,random_state = 0)


# In[26]:


print(x)


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)


# In[29]:



y_train.shape


# In[31]:


from sklearn.tree import DecisionTreeClassifier


# In[33]:


dtc = DecisionTreeClassifier()


# In[34]:


dtc.fit(x_train,y_train) 


# In[35]:


y_pred = dtc.predict(x_test)


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


acc = accuracy_score(y_test,y_pred)


# In[38]:


print(acc)


# In[ ]:




