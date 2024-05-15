# Import libraries
import os
import pickle 
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupKFold

from process_images import csv_filepath
np.random.seed(42)



#File path, please update before running
#metadata is not neccesary as all relevant info from there has been baked into the Attributes_final file
Attributescsv_path= csv_filepath


# Read the CSV file into a DataFrame
df = pd.read_csv(Attributescsv_path, delimiter=',')
# Separate features (X) and target variable (y)
X = df.drop(columns=['lesion_name', 'is_cancer_bool','patient_id'])  # Features - a new copy of df without name and cancer status
y = df['is_cancer_bool']  # Target variable


# Define the parameter grid for grid search
param_grid = {
    'max_depth': [10,15,20,25,30,40,50,60,70,80],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required to be at a leaf node
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Define the class weights
class_weights = {
    0: 1,  # Class 0
    1: 10  # Class 1 (adjust this weight accordingly)
}

# Initialize the Decision Tree classifier with class weights
decision_tree = DecisionTreeClassifier(class_weight=class_weights,random_state=44)

# Initialize GridSearchCV
# Prepare cross-validation
group_kfold = GroupKFold(n_splits=5)
#Prepare cross-validation - images from the same patient must always stay together
groups_train = df.loc[X_train.index, 'patient_id']
# Initialize GridSearchCV with the GroupKFold
grid_search = GridSearchCV(decision_tree, param_grid, cv=group_kfold, n_jobs=-1, scoring='accuracy')
# Perform grid search cross-validation 
grid_search.fit(X_train, y_train, groups=groups_train)

# Get the best model
best_decision_tree = grid_search.best_estimator_

# Make predictions on the test set
y_pred_best = best_decision_tree.predict(X_test)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Decision Tree Accuracy:", accuracy_best)

# Compute confusion matrix for the best model
cm_best = confusion_matrix(y_test, y_pred_best)
print("Best Model Confusion Matrix:")
print(cm_best)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate the model with recall
recall_best = recall_score(y_test, y_pred_best)
print("Best Model Recall:", recall_best)


classifier = DecisionTreeClassifier(class_weight=class_weights,
                                    max_depth=best_params['max_depth'],
                                    min_samples_leaf=best_params['min_samples_leaf'],
                                    min_samples_split=best_params['min_samples_split'])
#It will be tested on external data, so we can try to maximize the use of our available data by training on 
#ALL of x and y
classifier = classifier.fit(X,y)

#This is the classifier you need to save using pickle, add this to your zip file submission
filename = 'groupNJ_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Get the absolute path of the saved file
print("Trained model has been exported and saved at:")
print(os.path.abspath(filename))