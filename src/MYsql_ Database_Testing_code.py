#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from sqlalchemy import create_engine

import db_config


# In[2]:


def connect_db():
    db_uri = os.getenv('DATABASE_URI')
    try:        
        # GET THE CONNECTION OBJECT (ENGINE) FOR THE DATABASE
        engine = create_engine(db_uri, echo=True)
        return engine
    except Exception as ex:
        print("Connection could not be made due to the following error: \n", ex)


# In[3]:


import config
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sqlalchemy.types import Integer, Text, String, DateTime, Float


# In[4]:


df_recipes = pd.read_csv(config.PARSED_PATH_MINI)
# df_recipes = pd.read_csv('../../dataset_new/df_parsed.csv')

df_recipes = df_recipes.iloc[0:30000, :]
# df_recipes.dtypes


# In[5]:





# In[6]:


engine = connect_db()


# In[7]:


#This code send CSV dataset to MySql database
df_recipes.to_sql(
    os.getenv('TABLE_NAME'),
    engine,
    if_exists='replace',
    index=False,
    chunksize=500,
    dtype={
        "id": Integer,
        "name": Text,
        "minutes": Integer,
        "n_steps":  Integer,
        "steps": Text,
        "description": Text,
        "ingredients": Text,
        "n_ingredients": Integer,      
        "calories": Float,
        "total_fat_PDV": Float,
        "sugar_PDV": Float,
        "sodium_PDV": Float,
        "protein_PDV": Float,
        "saturated_fat_PDV": Float,
        "carbohydrates_PDV": Float,
        "food_types": String(50)
    }
)


# In[8]:


# This Code retrives from MySQL databse

table_df = pd.read_sql_table(
    os.getenv('TABLE_NAME'),
    con=engine
)


# In[9]:


table_df.head()


# In[8]:





# In[7]:


# DEFINE THE DATABASE CREDENTIALS
user = 'root'
password = ''
host = '127.0.0.1'
port = 3306
database = 'recipe_rec_dataset'
table_name = 'recipe_tbl_name'


# PYTHON FUNCTION TO CONNECT TO THE MYSQL DATABASE AND
# RETURN THE SQLACHEMY ENGINE OBJECT
def get_connection():
    return create_engine(
        url="mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
            user, password, host, port, database
        )
    )



def connect_db():
    try:        
        # GET THE CONNECTION OBJECT (ENGINE) FOR THE DATABASE
        engine = get_connection()
        print(
            f"Connection to the {host} for user {user} created successfully.")
    except Exception as ex:
        print("Connection could not be made due to the following error: \n", ex)
        
        
connect_db()


# In[18]:





# In[ ]:




