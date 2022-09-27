#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

import mysql.connector as connection
import database_config
import config
# df_raw = pd.read_csv(config.PARSED_PATH_MINI)
# df_recipes = pd.read_csv('../../dataset_new/df_parsed.csv')

# df_raw = df_raw.iloc[0:5000, :]

def get_dataset_from_db():
    host = database_config.HOST
    database = database_config.DATABASE
    user = database_config.USER
    passwd = database_config.PASSWD
    table_name = database_config.TABLE_NAME

    result_dataFrame = ''
#     mydb = None

    try:
        mydb = connection.connect(host=host, database=database, user=user, passwd=passwd, use_pure=True)
        query = "Select * from "+ table_name +";"
        result_dataFrame = pd.read_sql(query,mydb)
#         mydb.close() #close the connection
    except Exception as e:        
        print(str(e))
    finally:
        if mydb.is_connected():
            mydb.close()
    
    return result_dataFrame


# In[ ]:





# In[14]:


df_raw = get_dataset_from_db()
df_raw.tail()


# In[16]:


df_raw['food_types'] = df_raw['food_types'].apply(lambda x: x.strip("\r"))


# In[17]:


df_raw.head()


# In[6]:


df_raw['name'] = df_raw['name'].replace('', np.nan)
df_raw['ingredients'] = df_raw['ingredients'].replace('', np.nan)
df_raw['ingredients_parsed'] = df_raw['ingredients_parsed'].replace('', np.nan)
df_raw['description'] = df_raw['description'].replace('', np.nan)
df_raw['steps'] = df_raw['steps'].replace('', np.nan)
new_df = df_raw.dropna()


# In[7]:


new_df.shape


# In[8]:


df_recipes = new_df


# In[9]:


df_recipes.drop(columns=['id', 'name', 'description', 'steps', 'ingredients', 'ingredients_parsed'], inplace=True)


# In[10]:


import numpy as np

def convert_datatype_to_float(data, col):
    data[col] = data[col].astype('float32')
    return data

def imput_nan_value(data, col):
    median_value = np.median(data[col].dropna())
    data[col] = data[col].fillna(median_value)
    return data   


# In[11]:


# df_recipes['minutes'] = df_recipes['minutes'].astype('float32')
# df_recipes['n_steps'] = df_recipes['n_steps'].astype('float32')
# df_recipes['n_ingredients'] = df_recipes['n_ingredients'].astype('float32')


# In[12]:


cols = df_recipes.columns[:10]
for col in cols:
    df_recipes = imput_nan_value(df_recipes, col)
# cols


# In[13]:


df_recipes.head(20)


# In[10]:


# df_recipes_final.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[11]:


df_recipes_final = df_recipes 
X = df_recipes_final.drop('food types', axis=1)
y = df_recipes_final['food types']


# In[12]:


col_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# , stratify=y


# In[13]:


# scalar = StandardScaler()
# X_train = pd.DataFrame(scalar.fit_transform(X_train))
# X_test = pd.DataFrame(scalar.fit_transform(X_test))


# In[14]:


y_train.value_counts()


# In[15]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# In[17]:


y_train_res.value_counts()


# In[18]:


from sklearn.svm import SVC

model_svc = SVC(kernel='rbf', C=1.0)
# model_svc = SVC(kernel='linear', C=2, probability=True,random_state=0)
model_svc.fit(X_train_res, y_train_res)
y_pred = model_svc.predict(X_test)


# In[19]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[20]:


print(accuracy_score(y_test, y_pred))


# In[21]:


print(classification_report(y_test, y_pred))


# In[22]:


# input = [[70, 8, 7, 395.4, 31.0, 20.0, 29.0, 51.0, 33.0, 8.0]]
input = [[55, 11, 7, 51.5, 0.0, 13.0, 0.0, 2.0, 0.0, 4.0]]
# input = [[15, 4, 9, 380.7, 53.0, 7.0, 24.0, 6.0, 24.0, 6.0]]

gg = model_svc.predict(input)


# In[23]:


gg[0]


# In[24]:


df_raw.loc[df_raw['food types'] == gg[0]]


# In[46]:


import pickle 
with open(config.CLASSIFIER_MODEL_PATH, "wb") as f:
    pickle.dump(model_svc, f)


# In[25]:


# df_new_new = df_raw[df_raw['food types'] == dd[0]]

# print(df_new_new)


# In[26]:


import pickle 
from sklearn.ensemble import AdaBoostClassifier

model_ada = AdaBoostClassifier(learning_rate = 0.01, n_estimators= 20)
model_ada.fit(X_train, y_train)
ada_y_pred = model_ada.predict(X_test)

# with open(config.CLASSIFIER_MODEL_PATH, "wb") as f:
#     pickle.dump(model_ada, f)


# In[25]:


# input = [[30, 9, 6, 173.4, 18.0, 0.0, 17.0, 22.0, 35.0, 1.0]]
# input = [[0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
# input = [[55, 11, 7, 51.5, 0.0, 13.0, 0.0, 2.0, 0.0, 4.0]]
input = [[15, 4, 9, 380.7, 53.0, 7.0, 24.0, 6.0, 24.0, 6.0]]


dd = model_ada.predict(input)
dd


# In[28]:


print(accuracy_score(y_test, ada_y_pred))


# In[36]:


from sklearn.model_selection import GridSearchCV

ada_clf = AdaBoostClassifier()
params = {
    'n_estimators': np.arange(10, 300, 10),
    'learning_rate': [0.01, 0.05, 0.1, 1]
}
classes = y_train.unique()
#execute GridSearch
grid_clf = GridSearchCV(estimator=ada_clf, scoring='f1_weighted', param_grid=params, cv=5 )
grid_clf.fit(X_train, y_train)
print("The best parameters are: ", grid_clf.best_params)


# In[34]:





# In[35]:





# In[ ]:




