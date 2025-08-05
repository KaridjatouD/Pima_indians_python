# Prédiction du diabète – Dataset Pima Indians Diabetes

## Contexte

Ce projet de machine learning a pour objectif de prédire la présence de diabète chez des patientes à partir d’un ensemble de caractéristiques médicales. Le dataset utilisé provient de l’UCI Machine Learning Repository
et se concentre sur une population de femmes d’origine amérindienne Pima âgées de 21 ans ou plus.

Ce projet a été réalisé dans le cadre de mon portfolio, à destination de recruteurs et de clients potentiels, afin de démontrer mes compétences en data science, data visualisation et modélisation supervisée.

## Objectif
Prédire si une patiente est atteinte de diabète (Outcome = 1) ou non (Outcome = 0) à partir de variables cliniques comme : le nombre de grossesses, la concentration de glucose ou la pression artérielle


### Description du dataset:

Pregnancies: Nombre de grossesses, Glucose: Concentration de glucose, BloodPressure: Pression artérielle diastolique, SkinThickness: Épaisseur du pli cutané (en mm), Insulin: Taux d’insuline sérique, BMI: Indice de masse corporelle (poids/taille²), DiabetesPedigreeFunction: Hérédité du diabète
Age: Âge en années
Outcome	1 = Diabétique, 0 = Non diabétique

### Pipeline de traitement:

1. Préparation des données
* Nettoyage des valeurs aberrantes : certaines colonnes avaient des zéros impossibles ( pression artérielle = 0)..
* Remplacement des valeurs manquantes : les zéros anormaux ont été remplacés par la médiane de chaque variable concernée.

2. Analyse exploratoire
* Analyse des statistiques descriptives
* Visualisation des distributions (histogrammes, boxplots)
* Étude de la corrélation entre variables (matrice de corrélation)
* Vérification de la distribution des classes : 65% non-diabétiques vs 35% diabétiques

3. Prétraitement
* Standardisation des variables avec StandardScaler pour les modèles sensibles à l’échelle (logistic regression, KNN)
* Split train/test : 80% pour l'entraînement, 20% pour le test

4. Modélisation

Régression Logistique
* Accuracy : 71%
* Précision classe 1 (diabétiques) : 0.60
* Rappel classe 1 : 0.50
* F1-score : 0.55

K-Nearest Neighbors (KNN)
* Test de plusieurs k entre 1 et 10
* Meilleur score obtenu avec k = 8
* Accuracy : 76.6%
* Précision classe 1 : 0.71
* Rappel classe 1 : 0.56
* F1-score : 0.62
* Courbe ROC : AUC = 0.79

### Visualisations:
* Matrices de confusion
* Courbes ROC
* AUC (aire sous la courbe)

### Résultats:
* Le modèle KNN (k=8) a donné les meilleurs résultats globaux avec un bon équilibre entre précision et rappel.
* L’aire sous la courbe ROC (0.79) montre une bonne capacité de discrimination entre classes.
* Possibilité d’améliorer les performances via :
    * des techniques de rééquilibrage des classes (SMOTE)
    * l’optimisation d’hyperparamètres
    * l’essai d’autres modèles (Random Forest, XGBoost...)



### À propos
Ce projet fait partie de mon portfolio de data scientist, destiné aux missions freelance et aux postes en entreprise. Je peux adapter ce type d’analyse aux besoins d’une marque ou d’un client spécifique, notamment dans le secteur de la santé, du e-commerce ou de la mode.
