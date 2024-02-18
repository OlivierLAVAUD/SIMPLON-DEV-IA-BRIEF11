# preprocessing.py

# Importations des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
import pickle

# Fonction pour charger les données à partir du fichier
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Fonction pour traiter les données
def process_data(users_data, movies_data, test_size=0.2):
    processed_users_data = {}
    processed_movies_data = {}
    
    for user_id, preferences in users_data.items():
        train_preferences, test_preferences = train_test_split(list(preferences.items()), test_size=test_size, random_state=42)
        processed_users_data[user_id] = {
            'train_preferences': dict(train_preferences),
            'test_preferences': dict(test_preferences)
        }
    
    for movie_id, ratings in movies_data.items():
        processed_movies_data[movie_id] = ratings
        
    return processed_users_data, processed_movies_data

# Fonction pour sauvegarder les données dans un fichier
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# Script Principal
def main():
    # Chargement des données
    users_data, movies_data = load_data("./data/users_data.pkl"), load_data("./data/movies_data.pkl")
    
    # Traitement des données
    processed_users_data, processed_movies_data = process_data(users_data, movies_data)
    
    # Sauvegarde des données
    save_data(processed_users_data, "./data/processed_users_data.pkl")
    save_data(processed_movies_data, "./data/processed_movies_data.pkl")

if __name__ == "__main__":
    main()
