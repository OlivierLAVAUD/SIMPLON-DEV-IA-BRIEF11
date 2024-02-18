
# Système de Recommandation de Films

Ce projet consiste en un système de recommandation de films basé sur une approche de factorisation de matrices non négatives (NMF). Le modèle recommande des films aux utilisateurs en se basant sur leurs préférences passées.

## Structure du Projet

Le projet est organisé comme suit :

-   `data/`: Ce répertoire contient les données utilisées pour l'entraînement du modèle.
-   `src/`: Ce répertoire contient le code source du projet.
    -   `collecting.py`: Script Python pour la collecte de donnees
    -   `preprocessing.py`: Script Python pour le prétraitement des données.
    -   `modeling.py`: Script Python pour l'entraînement du modèle de recommandation.
    -   `evaluation_metrics.py`: Script Python pour évaluer les performances du modèle avec différentes métriques.

-   `README.md`: Ce fichier. Il fournit des informations sur le projet et comment l'utiliser.

## Installation

1.  Clonez ce dépôt sur votre machine locale :


`git clone https://github.com/votre_utilisateur/système-recommandation-films.git` 

2.  Accédez au répertoire du projet :

`cd système-recommandation-films` 

3.  Installez les dépendances Python requises :

bashCopy code

`pip install -r requirements.txt` 

## Utilisation

0. Collectez les données:
`python src/collecting.py` 

1.  Pré-traitez les données :
`python src/preprocessing.py` 

2.  Entraînez le modèle de recommandation :
`python src/modeling.py` 

3.  Évaluez les performances du modèle avec différentes métriques :
	`python src/evaluation.py` 

4. Backtesting
	`python src/backtesting.py` 

### Exemples de Metriques

### Classification binaire :

-   **Exactitude (Accuracy)** : Nombre de predictions correctesNombre total de predictionsNombre total de predictionsNombre de predictions correctes​
-   **Précision (Precision)** : Nombre de predictions positives correctesNombre total de predictions positivesNombre total de preˊdictions positivesNombre de predictions positives correctes​
-   **Rappel (Recall)** : Nombre de predictions positives correctesNombre total d’instances reellement positivesNombre total d’instances reellement positivesNombre de predictions positives correctes​
-   **F1-score** : La moyenne harmonique de la précision et du rappel, donnant une mesure unique de la performance en termes de précision et de rappel.
-   **Courbe ROC et Aire sous la courbe (ROC AUC)** : Une mesure de la performance du modèle en termes de taux de vrais positifs par rapport au taux de faux positifs à différents seuils de classification.

### Classification multiclasse :

-   **Exactitude (Accuracy)** : Nombre de preˊdictions correctesNombre total de preˊdictionsNombre total de preˊdictionsNombre de preˊdictions correctes​
-   **Précision (Precision)** : La précision moyenne sur toutes les classes.
-   **Rappel (Recall)** : Le rappel moyen sur toutes les classes.
-   **F1-score** : La moyenne harmonique de la précision et du rappel sur toutes les classes.
-   **Matrice de confusion** : Une matrice montrant le nombre de prédictions correctes et incorrectes pour chaque classe.

### Régression :

-   **Erreur quadratique moyenne (Mean Squared Error, MSE)** : La moyenne des carrés des écarts entre les valeurs prédites et les valeurs réelles.
-   **Erreur absolue moyenne (Mean Absolute Error, MAE)** : La moyenne des valeurs absolues des écarts entre les valeurs prédites et les valeurs réelles.
-   **Coefficient de détermination (R-squared)** : Une mesure de la proportion de la variance dans la variable dépendante qui est prédictible à partir de la variable indépendante.

### Séries temporelles :

-   **Erreur quadratique moyenne (Mean Squared Error, MSE)** : La moyenne des carrés des écarts entre les valeurs prédites et les valeurs réelles sur une séquence de temps.
-   **Erreur absolue moyenne (Mean Absolute Error, MAE)** : La moyenne des valeurs absolues des écarts entre les valeurs prédites et les valeurs réelles sur une séquence de temps.
-   **Erreur absolue moyenne pondérée (Weighted Mean Absolute Error, WMAE)** : MAE pondéré par l'importance de chaque période de temps.
-   **Erreur absolue moyenne pourcentage (Mean Absolute Percentage Error, MAPE)** : La moyenne des pourcentages absolus des écarts entre les valeurs prédites et les valeurs réelles sur une séquence de temps.

### Backtesting
Pur effectuer différentes formes de backtesting, nous pouvons utiliser des techniques telles que la validation croisée, la validation croisée temporelle, et la validation croisée avec séparation des groupes. Voici comment nous pouvons intégrer ces techniques dans notre structure de code existante :

1.  **Validation croisée (k-fold cross-validation) :** Cette technique divise les données en k sous-ensembles et entraîne le modèle k fois, en utilisant chaque sous-ensemble comme ensemble de test une fois et les autres sous-ensembles comme ensembles d'entraînement.
    
2.  **Validation croisée temporelle :** Pour les données séquentielles comme les séries temporelles, cette technique divise les données en blocs de temps séquentiels, en utilisant les données les plus anciennes comme ensemble d'entraînement et les données les plus récentes comme ensemble de test.
    
3.  **Validation croisée avec séparation des groupes :** Si les données contiennent des groupes ou des clusters, cette technique divise les données en k sous-ensembles de telle sorte que les groupes ne se chevauchent pas entre les ensembles d'entraînement et de test.
    
### Validation croisée (k-fold cross-validation) :

    from sklearn.model_selection import KFold, cross_val_score
    
        def evaluate_model_with_cross_validation(model, X_train):
            k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, cv=k_fold)
            average_cv_score = cv_scores.mean()
            return average_cv_score

### Validation croisée temporelle :

    from sklearn.model_selection import TimeSeriesSplit
    
    def evaluate_model_with_time_series_cross_validation(model, X_train):
        """Évalue les performances du modèle avec validation croisée temporelle."""
        time_series_cv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, cv=time_series_cv)
        average_cv_score = cv_scores.mean()
        return average_cv_score

### Validation croisée avec séparation des groupes :

    from sklearn.model_selection import GroupKFold
    
    def evaluate_model_with_group_cross_validation(model, X_train, groups):
        """Évalue les performances du modèle avec validation croisée avec séparation des groupes."""
        group_k_fold = GroupKFold(n_splits=5)
        cv_scores = cross_val_score(model, X_train, groups=groups, cv=group_k_fold)
        average_cv_score = cv_scores.mean()
        return average_cv_score

