#Importation des librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix


##Chargement et aperçu des données
data=pd.read_csv("/Users/karidjatoudiaby/Documents/PYTHON/Machine learning/diabetes.csv")
print(data.head)
print(data.info)


##Analyse exploratoire des données
print('Analyse exploratoire des données')
print(data.describe()) #stat de base
print((data == 0).sum) #varifier valeur manquantes ou aberantes

#Apercu de la variable d'interet, distribution des classes
print(data['Outcome'].value_counts())
print(data['Outcome'].value_counts(normalize=True)*100)

#Outcome 0 500 n'ont pas le diabetes ( 65.104167%) / 1 contre 268 ont le diabetes ( 34.895833%), l'ecart de classe n'est pas choquant 


#Histogrammes pour les variables importantes
cols = ['Glucose', 'BMI', 'Age', 'Insulin', 'Pregnancies']

plt.figure(figsize=(15,10))

for i, col in enumerate(cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(data[col], kde=True, color='skyblue')
    plt.title(f'Distribution de {col}')

plt.tight_layout()
plt.show()

#Boxplots pour les valeurs extremes 

plt.figure(figsize=(15,10))

for i, col in enumerate(cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=data[col], color='lightcoral')
    plt.title(f'Boxplot de {col}')

plt.tight_layout()
plt.show()

#correlation entre les variables
matrice_corr=data.corr()
print(matrice_corr)

##Preparation des données

#netoyage des donnes, remplacer les valeurs 0 qui n'ont pas de sens, comme pour la colonne Glucose, Blood pressure, SkinTickness, Insulin et BMI, pour toutes ces variables il est impossible d'avoir une valeur a 0, donc on va remplacer ces 0 par NA
var_0=['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] 
data[var_0]=data[var_0].replace(0, np.nan)
print(data[var_0])
print(data.isna().sum())  

#dans la colonne insuline il y a plus de 370 donnees manquantes, supprimer toutes ces lignes reduirai trop notre dataset, il est plus judicieux de remplacer ces valeurs, on peut utiliser la mediane, la moyenne ou utiliser les plus proches voisins 
#les plus proches voisins ou knn, consiste a regarder les k lignes dont les autres variables sont les plus similaires a notre Nan et remplacer la valeur manquantes en faisant la moyenne de ses voisins pour la variable concernes.
#certaines variables sont correllee mais dans l ensemble il ny a pas une correlation asses forte entre les 5 variables comprennant des donnes manquantes pour remplacer par un knn, nous allons donc faire un remplacement de ces valeurs manquantes par la mediane ou la moyenne

#remplacer les Nan par la mediane
data.fillna(data.median(numeric_only=True),inplace=True)
print(data.isna().sum()) #plus de valeurs manquantes
print(data.head)
print(data.describe())


#separer le jeu de donnees entre variable d interet (y) et les autres
#  X et y

X=data.drop('Outcome', axis=1)
y=data['Outcome']

#split en train/test dees varaibles explicatives et de la variable a expliquer (ratio 80/20)
#on garde la mm proportion de classes de y (meme nmb de 0 et 1)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

#on va normaliser les donnees pour mettre toute les variables a la mm "echelle" (DiabetesPedigreeFunction on en des 0.3, 2.3 et insulin on a des 125, 94...)
std=StandardScaler()

X_train_scaled=std.fit_transform(X_train)
X_test_scaled=std.transform(X_test)

print(std)
print(train_test_split)

##Modelisation 

    ##Regression logistique

#Nous sommes dans le cas dune classification binaire, (variable dinteret qualitatif 0 ou 1), nous allons donc utiliser la regression logistique pour commencer (outil du machine learningpour predire une valeur qualitatif de donnes d apres les obs reelles du jeu de donnees)

##creation et entrainement du modele 
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

##prediction
y_pred_logreg=log_reg.predict(X_test_scaled)

##Evaluation 

print("Regression logistique")
print(confusion_matrix(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
print("Accuracy:", accuracy_score(y_test,y_pred_logreg))

#on a une accuracy de de 71%
#f1 score de cas positifs est de 55% et negatifs 78%, le modele predit mal les cas de diabetes, (pourtant l enjeu le plus important)
#le modele predit mieux la classe majoritaire qui est non diabete (il l a connait mieux car plus de donnees), mais dans ce cas/contexte on voudrait qu'elle predit mieux la classe des personnes qui ont le diabete.


    ##K plus proche voisins

##creation du modele avec 5 voisin plus proche 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

##prediction 
y_pred_knn = knn.predict(X_test_scaled)

##Evaluation

print("K-Nearest Neighbors, k=5")
print(confusion_matrix(y_test,y_pred_knn))
print(classification_report(y_test,y_pred_knn))
print("Accuracy:", accuracy_score(y_test,y_pred_knn))

#on a un accuracy de 75%, f1 score de cas positifs est de 63%, et pour les cas negatifs 81%
#on a de meilleurs resultats avec KNN, mais allons plus loins en modifiant le nombre de voisin 

#chercher le k avec le meilleur accuracy
for k in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred=knn.predict(X_test_scaled)
    print(f"k={k}, Accuracy = {accuracy_score(y_test, y_pred):.3f}")

#avec k=8, Accuracy = 0.766, on obtient une accuracy max en retenant les 8 voisins les plus proches 


    ##K plus proche voisins 
    ##creation du modele avec 8 voisin plus proche 

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_scaled, y_train)

##prediction 
y_pred_knn = knn.predict(X_test_scaled)

##Evaluation

print("K-Nearest Neighbors, k=8")
print(confusion_matrix(y_test,y_pred_knn))
print(classification_report(y_test,y_pred_knn))
print("Accuracy:", accuracy_score(y_test,y_pred_knn))

#On a 88 vrais negatifs et 30 vrais positifs, avec une accuracy a 77%, et un f1 score de cas positifs a 62% et pour les cas megatifs 83% 

#l accuracy est 6 point de pourcentage supperieur comparer a la regresion logistique...

##Visualison nos resultats!


#probabilite predites pour la classe1 
y_prob = knn.predict_proba(X_test_scaled)[:,1]

##courbe ROC

fpr, tpr, thresholds= roc_curve(y_test, y_prob)
auc_score=roc_auc_score(y_test, y_prob)

##Plot

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', color='darkorange')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('False positif rate')
plt.ylabel('True positif rate')
plt.title('Courbe ROC - KNN, k=8')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#Air sous la courbe est de 79%, probabilite de classer un posotif plus haut qu'un negatif
#rappel AUC supperieur a 75 est bon, inferieur a 65 est faible

##Matrice de confusion

cm=confusion_matrix(y_test, knn.predict(X_test_scaled))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predit')
plt.ylabel('Reel')
plt.title('Matrice de confusion - KNN, k=8')
plt.tight_layout()
plt.show()

#Allons plus loin?

#Prediction reelle sur un nouveau patient

def predict_nv_patient(knn, std, patient_data):

    patient_array = np.array(patient_data).reshape(1, -1)
    patient_scaled = std.transform(patient_array)

    prediction = knn.predict(patient_scaled)[0]
    proba = knn.predict_proba(patient_scaled)[0][1]

    print("Résultat de la prédiction :")
    print(f" - Classe prédite : {'Diabétique' if prediction == 1 else 'Non diabétique'}")
    print(f" - Probabilité d’être diabétique : {round(proba * 100, 2)}%")

    return prediction, proba

nv_patient=[[6,148,72,35,168,33.6,0.627,50]]
predict_nv_patient(knn, std, nv_patient)

