# Diabetes Prediction – Pima Indians Diabetes Dataset

## Context

This machine learning project aims to predict the presence of diabetes in patients using a set of medical features. The dataset comes from the UCI Machine Learning Repository and focuses on Pima Indian women aged 21 years and older.
This project was carried out as part of my portfolio, for recruiters and potential clients, in order to demonstrate my skills in data science, data visualization and supervised modeling.

## Objective

Predict whether a patient is diabetic (Outcome = 1) or not (Outcome = 0) from clinical variables such as number of pregnancies, glucose concentration or blood pressure.

## Dataset description:

Pregnancies: Number of pregnancies, Glucose: Plasma glucose concentration, BloodPressure: Diastolic blood pressure, SkinThickness: Triceps skinfold thickness (mm), Insulin: Serum insulin level, BMI: Body Mass Index (weight/height²), DiabetesPedigreeFunction: Diabetes pedigree function, Age: Age in years Outcome: 1 = Diabetic, 0 = Non-diabetic

### Processing pipeline:

1. Data preparation
    * Cleaning of abnormal values: some columns had impossible zeros (e.g., blood pressure = 0)
    * Replacement of missing values: abnormal zeros were replaced by the median value of each variable
      
2. Exploratory analysis
    * Descriptive statistics
    * Visualization of distributions (histograms, boxplots)
    * Study of variable correlations (correlation matrix)
    * Checking class distribution: 65% non-diabetic vs 35% diabetic
      
3. Preprocessing
    * Standardization of variables with StandardScaler for scale-sensitive models (logistic regression, KNN)
    * Train/Test split: 80% training, 20% test
      
4. Modeling
   
Logistic Regression
* Accuracy: 71%
* Precision class 1 (diabetic): 0.60
* Recall class 1: 0.50
* F1-score: 0.55
  
K-Nearest Neighbors (KNN)
* Tested k values from 1 to 10
* Best score obtained with k = 8
* Accuracy: 76.6%
* Precision class 1: 0.71
* Recall class 1: 0.56
* F1-score: 0.62
* ROC curve: AUC = 0.79
  
### Visualizations:
* [Confusion matrices](images/Matrice_de_confusion.png)
* [ROC curves](images/Courbe_ROC.png)
* [AUC](images/Courbe_ROC.png)
* [Boxplots](images/Boxplots.png)
* [Histograms](images/Histogrammes.png)
  
### Results:
* The KNN model (k=8) produced the best global results with a good balance between precision and recall.
* AUC of 0.79 shows a good discriminative capacity between classes.
* Potential improvements:
    * Class balancing techniques (SMOTE)
    * Hyperparameter tuning
    * Testing other models (Random Forest, XGBoost...)
         
### About
This project is part of my data science portfolio, aimed at freelance missions and corporate positions. I can adapt this kind of analysis to the specific needs of a brand or client, particularly in the health, e-commerce or fashion sectors.

### Author

Feel free to explore, give feedback, or fork the project.

Karidjatou Diaby – Data Analyst
