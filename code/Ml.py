
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
