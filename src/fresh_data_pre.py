#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import seaborn as sns
import sys
# import matplotlib.pyplot as plt

import config


# In[2]:


df_recipes = pd.read_csv(config.PARSED_PATH)
# df_recipes = pd.read_csv('../../dataset_new/df_parsed.csv')

df_recipes = df_recipes.iloc[0:30000, :]
df_recipes.shape


# In[3]:


df = df_recipes[['id', 'name', 'minutes', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients',
       'ingredients_parsed', 'n_ingredients']]


# In[4]:


df[['calories','total fat (PDV)','sugar (PDV)','sodium (PDV)','protein (PDV)','saturated fat (PDV)','carbohydrates (PDV)']] = df.nutrition.str.split(",",expand=True) 


# In[5]:


df.head(2)


# In[6]:


df = df[['id', 'name', 'minutes', 'n_steps', 'steps', 'description',
       'ingredients', 'ingredients_parsed', 'n_ingredients', 'calories',
       'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)',
       'saturated fat (PDV)', 'carbohydrates (PDV)']]


# In[7]:


df['calories'] =  df['calories'].apply(lambda x: x.replace('[','')) 
df['carbohydrates (PDV)'] =  df['carbohydrates (PDV)'].apply(lambda x: x.replace(']','')) 

df['steps'] =  df['steps'].apply(lambda x: x.replace('[','')) 
df['steps'] =  df['steps'].apply(lambda x: x.replace(']',''))
df['ingredients'] =  df['ingredients'].apply(lambda x: x.replace('[','')) 
df['ingredients'] =  df['ingredients'].apply(lambda x: x.replace(']','')) 

df['steps'] = df['steps'].apply(lambda x: x.replace("'", ''))
df['ingredients'] = df['ingredients'].apply(lambda x: x.replace("'", ''))
df['description'] = df['description'].apply(lambda x: x.replace("'", ''))


# In[8]:


df[['calories','total fat (PDV)','sugar (PDV)','sodium (PDV)','protein (PDV)','saturated fat (PDV)','carbohydrates (PDV)']] = df[['calories','total fat (PDV)','sugar (PDV)','sodium (PDV)','protein (PDV)','saturated fat (PDV)','carbohydrates (PDV)']].astype('float')


# In[9]:


# df.dtypes


# In[10]:


df.head(3)


# In[11]:


df['steps'].head(1)


# In[12]:


df['food types'] = np.nan
df['food types'] = df['food types'].astype('str')

for i in df['ingredients'].index:
    if('eggs' not in df['ingredients'][i]):
         if('ice-cream' in df['ingredients'][i] or 'chocolate' in df['ingredients'][i] or 'cookies' in df['ingredients'][i]):
                df['food types'][i]='Veg dessert'
    elif('eggs' in df['ingredients'][i]):
        if('ice-cream' in df['ingredients'][i] or 'chocolate' in df['ingredients'][i] or 'cookies' in df['ingredients'][i]):
                df['food types'][i]='Non-Veg dessert'


# In[13]:


for i in df.index:
    if(df['food types'][i]!='Veg dessert' and df['food types'][i]!='Non-Veg dessert' and 20<df['calories'][i]<300):
        df['food types'][i]='Healthy'


# In[14]:


for i in df.index:
    if(df['food types'][i]!='Veg dessert' and df['food types'][i]!='Non-Veg dessert' and df['food types'][i]!='Healthy'):
        if('chicken' in df['ingredients'][i] or 'eggs' in df['ingredients'][i] or'ham' in df['ingredients'][i] or 'pepperoni' in df['ingredients'][i] ):
            df['food types'][i]='Non-veg'


# In[15]:


for i in df.index:
    if(df['food types'][i]!='Veg dessert' and df['food types'][i]!='Non-Veg dessert' and df['food types'][i]!='Healthy' and df['food types'][i]!='Non-veg'):
        df['food types'][i]='Veg'


# In[16]:


df['food types'].value_counts()


# In[17]:


df['food types'].isnull().sum()


# In[18]:


df.head(2)


# In[19]:


PARSED_PATH_NLP = "../input/PARSED_recipes_mini.csv"
df.to_csv(PARSED_PATH_NLP, index=False)


# In[21]:


df.columns


# In[ ]:




