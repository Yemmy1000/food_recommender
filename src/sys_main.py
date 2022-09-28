import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from ingredient_parser import ingredient_parser
import pickle
import config 
import unidecode, ast

# Top-N recomendations order by score
def get_recommendations(N, scores):
    # load in recipe dataset 
    df_recipes = pd.read_csv(config.PARSED_PATH_MINI)
    # order the scores with and filter to get the highest N scores
    new_scores = [x for x in scores if x > 0.5 ]
    #print(new_scores)
    top = sorted(range(len(new_scores)), key=lambda i: new_scores[i],  reverse=True)[:5]
    print(len(top))
    # create dataframe to load in recommendations 
    recommendation = pd.DataFrame(columns= ['id', 'recipe_name', 'cook_time', 'n_steps', 'steps', 'n_ingredients', 'calories', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)', 'food types'])
    count = 0
    for i in top:
        recommendation.at[count, "id"] = df_recipes["id"][i]
        recommendation.at[count, "recipe_name"] = title_parser(df_recipes["name"][i])
        recommendation.at[count, "cook_time"] = df_recipes["minutes"][i]        
        recommendation.at[count, "n_steps"] = df_recipes["n_steps"][i]
        recommendation.at[count, "steps"] = df_recipes["steps"][i]
        recommendation.at[count, "description"] = df_recipes["description"][i]        
        recommendation.at[count, "ingredients"] = ingredient_parser_final(
            df_recipes["ingredients"][i]
        )
        recommendation.at[count, "n_ingredients"] = df_recipes["n_ingredients"][i]
        recommendation.at[count, "calories"] = df_recipes["calories"][i]
        recommendation.at[count, "total fat (PDV)"] = df_recipes["total fat (PDV)"][i]
        recommendation.at[count, "sugar (PDV)"] = df_recipes["sugar (PDV)"][i]
        recommendation.at[count, "sodium (PDV)"] = df_recipes["sodium (PDV)"][i]
        recommendation.at[count, "protein (PDV)"] = df_recipes["protein (PDV)"][i]
        recommendation.at[count, "saturated fat (PDV)"] = df_recipes["saturated fat (PDV)"][i]
        recommendation.at[count, "carbohydrates (PDV)"] = df_recipes["carbohydrates (PDV)"][i]
        recommendation.at[count, "food types"] = df_recipes["food types"][i]
        recommendation.at[count, "score"] = f"{new_scores[i]}"
        count += 1
    return recommendation

# neaten the ingredients being outputted 
def ingredient_parser_final(ingredient):
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        #ingredients = ast.literal_eval(ingredient)
        ingredients = list(ingredient)
    
    ingredients = ','.join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def title_parser(title):
    title = unidecode.unidecode(title)
    return title 

def RecSys(ingredients, N=5):
    """
    The reccomendation system takes in a list of ingredients and returns a list of top 5 
    recipes based of of cosine similarity. 
    :param ingredients: a list of ingredients
    :param N: the number of reccomendations returned 
    :return: top 5 reccomendations for cooking recipes
    """

    # load in tdidf model and encodings 
    with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
        tfidf_encodings = pickle.load(f)

    with open(config.TFIDF_MODEL_PATH, "rb") as f:
        tfidf = pickle.load(f)

    # parse the ingredients using my ingredient_parser 
    try: 
        ingredients_parsed = ingredient_parser(ingredients)
    except:
        ingredients_parsed = ingredient_parser([ingredients])
    
    # use our pretrained tfidf model to encode our input ingredients
    ingredients_tfidf = tfidf.transform([ingredients_parsed])

    # calculate cosine similarity between actual recipe ingreds and test ingreds
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)

    #print(scores)
    # Filter top N recommendations 
    recommendations = get_recommendations(N, scores)
    return recommendations

if __name__ == "__main__":
    # test ingredients
    test_ingredients = "ginger, garlic"
    recs = RecSys(test_ingredients)
    print(recs)
    #print(scores)

