#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import numpy and pandas for read an manipulate data and nltk for text processing
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt


# ## Read OMD dataset

# In[2]:


data = pd.read_csv('OMD.csv', header=None, sep="\t")
data.head()


# ## Extracting labels
# - In this step we split data by ',' just for extracting labels from data

# In[3]:


classes = []
feature = []
for i in range(len(data)):
    classes.append(data.iloc[i][0][0])
    feature.append(data.iloc[i][0][1:])


# In[5]:


data = pd.DataFrame(data=feature, columns=['data'])
data['class'] = classes
data.head()


# ### Count vectorizer
# - After that we acquire our class information we can vectorize data to convert text to neumerical vectors for processing data.
# - CountVectorizer module from sklearn library do this for us.
# - In this step we can delete stop words from text that have not usefull information for uor learning model
# - Our learning stategy from this dataset is bag of words
# - We define n-grams with length 1 and use all of word that exists in data. So we set max_df to 1 and max_feature to None 

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer


# Create an object from CountVectorizer and set it's parameters.

# In[7]:


vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None)


# Fit CountVectorizer on dataset and save the results in x

# In[8]:


x = vectorizer.fit_transform(data['data'].values)


# ##### create New DataFrame from extracted vectors

# In[9]:


new_df = pd.DataFrame(x.toarray(), 
            columns=vectorizer.get_feature_names())
new_df['class'] = np.array(classes)
new_df.head()


# We can see that 3809 features, or in anoher words 3809 different word extracted from dataset

# In[10]:


x = new_df.iloc[:,:-1].values
y = new_df.iloc[:,-1].values


# In[11]:


new_df['class'].value_counts()


# ## Naive bayes and C.V.
# import GaussianNB and KFold for doing practice

# In[12]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold


# In[13]:


# Create objects from modules and set their parameters
gnb = GaussianNB()
k_fold = KFold(10)


# In[14]:


# Define 'scores' list for saving results in each iteration of learning process
scores = []
for k, (train, test) in enumerate(k_fold.split(x, y)):
    gnb.fit(x[train], y[train])
    scores.append(gnb.score(x[test], y[test]) )

# print the mean of scores for models
print(np.mean(scores))


# ## Read SandersPosNeg dataset
# - All operations on this dataset for preproscessing, feaure extraction and learning process is same for before one. So we ignore details that what we do.

# In[16]:


data = pd.read_csv('SandersPosNeg.csv', header=None)
data.head()


# In[17]:


classes = []
feature = []
for i in range(len(data)):
    classes.append(data.iloc[i][0][0])
    feature.append(data.iloc[i][0][1:])


# In[18]:


data = pd.DataFrame(data=feature, columns=['data'])
data['class'] = classes
data.head()


# In[19]:


vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None)
x = vectorizer.fit_transform(data['data'])


# In[20]:


new_df = pd.DataFrame(x.toarray(), 
            columns=vectorizer.get_feature_names())
new_df['class'] = np.array(classes)
new_df.head()


# In[21]:


new_df['class'].value_counts()


# In[22]:


x = new_df.iloc[:,:-1].values
y = new_df.iloc[:,-1].values


# In[23]:


gnb = GaussianNB()
k_fold = KFold(10)


# In[24]:


scores = []
for k, (train, test) in enumerate(k_fold.split(x, y)):
    gnb.fit(x[train], y[train])
    scores.append(gnb.score(x[test], y[test]) )
print(np.mean(scores))


# In[ ]:




