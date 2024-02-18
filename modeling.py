# modeling.py

# Importations des bibliothèques nécessaires
from sklearn.decomposition import NMF
import pickle

# Fonction pour charger les données à partir d'un fichier
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Fonction pour effectuer la modélisation avec NMF
def perform_nmf(users_data, movies_data, n_components=10):
    model = NMF(n_components=n_components, init='random', random_state=42)
    matrix = construct_rating_matrix(users_data, movies_data)  # Fonction pour construire la matrice de notation
    W = model.fit_transform(matrix)
    H = model.components_
    return W, H

# Fonction pour construire la matrice de notation
def construct_rating_matrix(users_data, movies_data):
    num_users = len(users_data)
    num_movies = len(movies_data)
    matrix = [[0] * num_movies for _ in range(num_users)]
    for user_id, preferences in users_data.items():
        for movie_id, rating in preferences.items():
            matrix[user_id - 1][movie_id - 1] = rating  # Indices basés sur zéro
    return matrix

# Fonction pour sauvegarder le modèle dans un fichier
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Script Principal
def main():
    # Chargement des données
    users_data, movies_data = load_data("./data/processed_users_data.pkl"), load_data("./data/processed_movies_data.pkl")
    
    # Modélisation avec NMF
    W, H = perform_nmf(users_data, movies_data)
    
    # Sauvegarde du modèle
    save_model((W, H), "nmf_model.pkl")

if __name__ == "__main__":
    main()
