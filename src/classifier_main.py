#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd

# from ingredient_parser import ingredient_parser
import pickle
import config 
import unidecode, ast

# Top-N recomendations order by score
def get_recommendations(N, res_class):
    # load in recipe dataset 
    df_recipes = pd.read_csv(config.PARSED_PATH_MINI)
    # order the scores with and filter to get the highest N scores
#     new_scores = [x for x in scores if x > 0.5 ]
    #print(new_scores)
    df_data = df_recipes.loc[df_recipes['food types'] == res_class[0]].sample(n = 10)

    #top = sorted(range(len(df_data)), key=lambda i: df_data[i],  reverse=True)
    #print(len(top))
    # create dataframe to load in recommendations 
    recommendation = pd.DataFrame(columns= ['id', 'recipe_name', 'cook_time', 'n_steps', 'steps', 'n_ingredients', 'calories', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)', 'food types'])
    count = 0
    #for i in df_data:
        #recommendation.at[count, "id"] = df_recipes["id"][i]
        #recommendation.at[count, "recipe_name"] = title_parser(df_recipes["name"][i])
        #recommendation.at[count, "cook_time"] = df_recipes["minutes"][i]        
        #recommendation.at[count, "n_steps"] = df_recipes["n_steps"][i]
        #recommendation.at[count, "steps"] = df_recipes["steps"][i]
        #recommendation.at[count, "description"] = df_recipes["description"][i]        
        #recommendation.at[count, "ingredients"] = ingredient_parser_final(
            #df_recipes["ingredients"][i]
        #)
        #recommendation.at[count, "n_ingredients"] = df_recipes["n_ingredients"][i]
        #recommendation.at[count, "calories"] = df_recipes["calories"][i]
        #recommendation.at[count, "total fat (PDV)"] = df_recipes["total fat (PDV)"][i]
        #recommendation.at[count, "sugar (PDV)"] = df_recipes["sugar (PDV)"][i]
        #recommendation.at[count, "sodium (PDV)"] = df_recipes["sodium (PDV)"][i]
        #recommendation.at[count, "protein (PDV)"] = df_recipes["protein (PDV)"][i]
        #recommendation.at[count, "saturated fat (PDV)"] = df_recipes["saturated fat (PDV)"][i]
        #recommendation.at[count, "carbohydrates (PDV)"] = df_recipes["carbohydrates (PDV)"][i]
        #recommendation.at[count, "food types"] = df_recipes["food types"][i]
#         recommendation.at[count, "score"] = f"{new_scores[i]}"
        #count += 1
    return df_data

# neaten the ingredients being outputted 
def ingredient_parser_final(ingredient):
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)
    
    ingredients = ','.join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def title_parser(title):
    title = unidecode.unidecode(title)
    return title 

def RecSys(inputs, N=5):
    """
    The reccomendation system takes in a list of inputs and returns a list of top N
    recipes based of classifer model function. 
    :param inputs: a list of numeric features
    :param N: the number of reccomendations returned 
    :return: top 5 reccomendations for cooking recipes
    """

    # load in classifier model and encodings 
    with open(config.CLASSIFIER_MODEL_PATH, 'rb') as f:
        classifier_model = pickle.load(f)

    recipe_class = classifier_model.predict(inputs)
    
    recommendations = get_recommendations(N, recipe_class)
    
    return recommendations

if __name__ == "__main__":
    # test ingredients
    inputs = [[5, 6, 5, 8.2, 0.0, 10.0, 2.0, 0.0, 0.0, 0.0]]
    recs = RecSys(inputs)
    print(recs)
    #print(scores)



# In[ ]:





# In[ ]:




