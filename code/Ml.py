# Importing necessary modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Initialize the Random Forest Classifier ---
clf_rf = RandomForestClassifier()

# --- Step 2: Define the Hyperparameters ---
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', None]
}

# --- Step 3: Initialize Grid Search ---
grid_search = GridSearchCV(
    estimator=clf_rf,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    verbose=2,
    n_jobs=-1
)

# --- Step 4: Fit the Model to the Training Data ---
grid_search.fit(X_train, y_train)

# --- Step 5: Retrieve the Best Estimator ---
best_rf = grid_search.best_estimator_

# --- Step 6: Plotting the results ---
parameters = list(param_grid.keys())

results = grid_search.cv_results_

for param in parameters:
    unique_vals = list(set(results[f'param_{param}']))
    mean_scores = []

    for val in unique_vals:
        mask = (np.array(results[f'param_{param}'].data) == val)
        mean_scores.append(np.mean(np.array(results['mean_test_score'])[mask]))

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(unique_vals)), mean_scores, marker='o')
    plt.xticks(range(len(unique_vals)), unique_vals)
    plt.title(f'F1 Score Trend based on {param}')
    plt.xlabel(param)
    plt.ylabel('Mean F1 Score')
    plt.grid(True)
    plt.show()



----------
#for XGBoost if the model is too slow use random search
 # Importing necessary modules
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Initialize the XGBoost Classifier ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# --- Step 2: Define the Hyperparameters ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.001, 0.01, 0.1],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0]
}

# --- Step 3: Initialize Grid Search ---
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    verbose=2,
    n_jobs=-1  # Use all processors
)

# --- Step 4: Fit the Model to the Training Data ---
grid_search.fit(X_train, y_train)

# --- Step 5: Retrieve the Best Estimator ---
best_xgb = grid_search.best_estimator_

# --- Step 6: Plotting the results ---
parameters = list(param_grid.keys())

results = grid_search.cv_results_

for param in parameters:
    unique_vals = list(set(results[f'param_{param}']))
    mean_scores = []

    for val in unique_vals:
        mask = (np.array(results[f'param_{param}'].data) == val)
        mean_scores.append(np.mean(np.array(results['mean_test_score'])[mask]))

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(unique_vals)), mean_scores, marker='o')
    plt.xticks(range(len(unique_vals)), unique_vals)
    plt.title(f'F1 Score Trend based on {param}')
    plt.xlabel(param)
    plt.ylabel('Mean F1 Score')
    plt.grid(True)
    plt.show()


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

