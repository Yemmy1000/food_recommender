# recipe_recommender
<<<<<<< HEAD

The main function this program provides is to recommend recipes information i.e For a recipe we wanto desplay the following
['id', 'recipe_name', 'cook_time', 'n_steps', 'steps', 'n_ingredients', 'calories', 
'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 
'carbohydrates (PDV)', 'food types']. Note: Some of these can be removed 

The entry point is 'src/sys_main.py' 

You can test the program by running the sys_main.py
Locate the main as shown below

if __name__ == "__main__":
    # test ingredients
    test_ingredients = "ginger, garlic" #list of ingredients, as many as you can
    recs = RecSys(test_ingredients)
    print(recs)
    #print(scores)

Also Note that the input is a text input
=======
recipe recommender
>>>>>>> 4ae6a4b (Initial commit)

* The Recipe parser is already included
* The Classifier.py has been updated to include 
    - cut_n (number to show )
    - random_n (TRUE/FALSE - to determine random fetch or not) 
