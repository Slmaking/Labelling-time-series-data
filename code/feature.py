import shap

# Using DeepExplainer to compute SHAP values for the given model
explainer = shap.DeepExplainer(model, x_train)
shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values, x_train.values, plot_type="bar", class_names= class_names, feature_names = x_train.columns)
shap.summary_plot(shap_values[1], x_train.values, feature_names = x_train.columns)

֫# If we pass a numpy array instead of a data frame then we
# need pass the feature names in separately
shap.dependence_plot(0, shap_values[0], x_train.values, feature_names=x_train.columns)

--------------

import lime
from lime import lime_tabular
import numpy as np

# Assuming you have already defined and trained your model, and have your x_train, x_test data ready

# Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    x_train.astype(np.float32),
    training_labels=y_train, # if you want to give training labels for better discretization
    feature_names=list(range(x_train.shape[1])), # or replace with actual feature names
    class_names=['Class 0', 'Class 1', 'Class 2'], # replace with your actual class names
    mode='classification'
)

# Pick an instance from the test set
i = 10 # for instance, you can choose any
instance = x_test[i]

# Get the explanation for this instance
exp = explainer.explain_instance(instance, model.predict_proba, num_features=x_train.shape[1])

# Visualize the explanation
exp.show_in_notebook()
