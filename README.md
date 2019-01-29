# naivebayes-scratch
Classifies diabetes cases using Naive Bayes algorithm from scratch, relying on less python libraries as possible. For more sofisticated operations I suggest you to look at [scikit-learn](https://scikit-learn.org/stable/) functions.

Usage:

    python run.py

## Repository content:
* **run.py** file:

It is the main file which get all data from dataset file and train/test a classifier, reporting its accuracy at the end of its execution.


* **diabetes.csv** file:

Pima Indians Diabetes Database consist of several medical predictor (independent) variables and one target (dependent) variable, Outcome. Independent variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

1. *Pregnancies*: Number of times pregnant.
2. *Glucose*: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
3. *BloodPressure*: Diastolic blood pressure (mm Hg).
4. *SkinThickness*: Triceps skin fold thickness (mm).
5. *Insulin*: 2-Hour serum insulin (mu U/ml).
6. *BMI*: Body mass index (weight in kg/(height in m)^2).
7. *DiabetesPedigreeFunction*: Diabetes pedigree function.
8. *Age*: Age (years).
9. *Outcome*: Class variable (0 or 1) 268 of 768 are 1, the others are 0.
