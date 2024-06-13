#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mindsdb_sdk

# connects to the default port (47334) on localhost 
server = mindsdb_sdk.connect()

# connects to the specified host and port
server = mindsdb_sdk.connect('http://127.0.0.1:47334')


# In[5]:


project = server.get_project()


# In[18]:


sentiment_classifier=project.list_models()


# In[20]:


sentiment_classifier = project.models.get('sentiment_classifier')


# In[22]:


sentiment_classifier.predict({"text": 'you are not welcome in this party!'})


# In[ ]:




