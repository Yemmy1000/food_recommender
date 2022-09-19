# DEFINE THE DATABASE CREDENTIALS

from os import environ

user = 'root'
password = ''
host = '127.0.0.1'
port = 3306
database = 'recipe_rec_dataset'
environ['TABLE_NAME'] = 'recipe_tbl_name'
environ['DATABASE_URI'] = "mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(user, password, host, port, database)