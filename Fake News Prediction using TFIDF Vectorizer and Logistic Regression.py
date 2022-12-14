#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


nltk.download('stopwords')


# In[3]:


print(stopwords.words('English'))


# In[4]:


df = pd.read_csv('train.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.columns


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


df = df.fillna('')


# In[11]:


# Merge title and author


# In[12]:


df['content'] = df['author'] + ' ' + df['title']


# In[13]:


df.head()


# In[14]:


# Stemming is the process of reducing a word to its root word.
# Eg. singer, singing -> sing


# In[15]:


stemmer = PorterStemmer()


# In[16]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = [stemmer.stem(word) for word in stemmed_content.lower().split() if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)


# In[17]:


df['content'] = df['content'].apply(stemming)


# In[18]:


X = df['content'].values
Y = df['label'].values


# In[19]:


vectorizer = TfidfVectorizer()


# In[20]:


vectorizer.fit(X)


# In[21]:


X = vectorizer.transform(X)


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2, stratify = Y)


# In[23]:


model = LogisticRegression()


# In[24]:


model.fit(x_train, y_train)


# In[25]:


predict = model.predict(x_train)


# In[26]:


accuracy_score = accuracy_score(predict, y_train)


# In[27]:


print('Accuracy score is ', accuracy_score)


# In[28]:


def predictor(n):
    x_new = x_test[n]
    prediction = model.predict(x_new)

    if prediction[0] == 0:
        print('Real News')
    else:
        print('Fake News')

predictor(0)
predictor(1)
predictor(2)
predictor(3)
predictor(4)


# In[ ]:




