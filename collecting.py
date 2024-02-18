    # collecte_donnees.py

# Importations des bibliothèques nécessaires
from pymongo import MongoClient
import pickle

# Connexion à la base de données MongoDB
def connect_to_mongodb(host='localhost', port=27017, db_name='film_recommendation'):
    client = MongoClient(host, port)
    db = client[db_name]
    return db

# Fonction pour récupérer les données des utilisateurs et des films
def get_user_movie_data(db):
    users_collection = db['users']
    movies_collection = db['movies']
    
    users_data = {}
    movies_data = {}
    
    for user in users_collection.find():
        user_id = user['_id']
        preferences = {entry['movie_id']: (entry['rating'], entry['timestamp']) for entry in user['preferences']}
        users_data[user_id] = preferences
    
    for movie in movies_collection.find():
        movie_id = movie['_id']
        ratings = {entry['user_id']: (entry['rating'], entry['timestamp']) for entry in movie['ratings']}
        movies_data[movie_id] = ratings
        
    return users_data, movies_data

# Fonction pour sauvegarder les données dans un fichier
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# Script Principal
def main():
    # Connexion à la base de données
    db = connect_to_mongodb()
    
    # Récupération des données
    users_data, movies_data = get_user_movie_data(db)
    
    # Sauvegarde des données
    save_data(users_data, "./data/users_data.pkl")
    save_data(movies_data, "./data/movies_data.pkl")

if __name__ == "__main__":
    main()
