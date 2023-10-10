
#XGBoosted
 import xgboost as xgb

# Calculate the ratio of negative to positive instances
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

clf_xgb = xgb.XGBClassifier(scale_pos_weight=ratio)
clf_xgb.fit(X_train, y_train)
y_pred = clf_xgb.predict(X_test)

print(classification_report(y_test, y_pred))

 from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'scale_pos_weight': [ratio]
}

grid_search = GridSearchCV(clf_xgb, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
----------------------

# Create a RandomForest classifier
clf_rf = RandomForestClassifier()

# Define the hyperparameters and their possible values
param_grid = {
    'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at every split
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'bootstrap': [True, False],  # Method of selecting samples for training each tree
    'class_weight': ['balanced', None]  # Weights associated with classes
}

# Initialize GridSearchCV with the classifier and the hyperparameters
grid_search = GridSearchCV(clf_rf, param_grid, scoring='f1', cv=5, verbose=2, n_jobs=-1)  # `n_jobs=-1` will use all processors to speed up the process

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Retrieve the best classifier
best_rf = grid_search.best_estimator_


 ----------

 from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Create a Decision Tree classifier
clf_tree = DecisionTreeClassifier()

# Define the hyperparameters and their possible values
param_grid = {
    'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
    'splitter': ['best', 'random'],  # Strategy used to choose the split at each node
    'max_depth': [None, 10, 20, 30, 40],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'max_features': [None, 'auto', 'sqrt', 'log2'],  # Number of features to consider at every split
    'class_weight': ['balanced', None]  # Weights associated with classes in case of imbalanced datasets
}

# Initialize GridSearchCV with the classifier and the hyperparameters
grid_search = GridSearchCV(clf_tree, param_grid, scoring='f1', cv=5, verbose=2, n_jobs=-1)  # `n_jobs=-1` will use all processors to speed up the process

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Retrieve the best classifier
best_tree = grid_search.best_estimator_


#Confusion matrix is the best option for multiclass
---------
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset as an example
iris = load_iris()
X = iris.data
y = iris.target

# Binarize the true labels (one-hot encoding)
n_classes = len(np.unique(y))
y_bin = label_binarize(y, classes=np.unique(y))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.5, random_state=42)

# Create a multi-class classifier (OneVsRestClassifier) with a binary classifier (SVC)
classifier = OneVsRestClassifier(SVC(probability=True, random_state=42))

# Fit the classifier to the training data
classifier.fit(X_train, y_train)

# Predict probabilities for each class on the test set
y_score = classifier.predict_proba(X_test)

# Initialize variables to store the false positive rates and true positive rates for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (class {0}) (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multi-Class Classification')
plt.legend(loc="lower right")
plt.show()

