# backtesting.py

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import NMF

def train_model(X_train):
    """Entraîne le modèle de recommandation."""
    model = NMF(n_components=10, init='random', random_state=42)
    model.fit(X_train)
    return model

def evaluate_model(model, X_test):
    """Évalue les performances du modèle."""
    # Évaluation directe sur l'ensemble de test
    evaluation_result = model.score(X_test)
    
    # Validation croisée sur l'ensemble d'entraînement
    cv_scores = cross_val_score(model, X_train, cv=5)  # 5-fold cross-validation
    average_cv_score = cv_scores.mean()
    
    return evaluation_result, average_cv_score

# main.py

def main():
    # Connexion à la base de données
    db = connect_to_database()

    # Chargement des données
    users_data = load_users_data(db)
    objects_data = load_objects_data(db)

    # Prétraitement des données
    preprocessed_data = preprocess_data(users_data, objects_data)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = train_model(X_train)

    # Évaluation du modèle
    evaluation_result, average_cv_score = evaluate_model(model, X_test)

    # Utilisation du modèle pour la recommandation
    user_id = '123'
    recommended_objects = recommend_objects(model, user_id)

    # Affichage des résultats
    print("Résultat de l'évaluation sur l'ensemble de test:", evaluation_result)
    print("Score moyen de validation croisée sur l'ensemble d'entraînement:", average_cv_score)
    print("Objets recommandés pour l'utilisateur", user_id, ":", recommended_objects)

if __name__ == "__main__":
    main()
