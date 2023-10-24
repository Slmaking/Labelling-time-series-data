import shap

# Using DeepExplainer to compute SHAP values for the given model
explainer = shap.DeepExplainer(model, x_train)
shap_values = explainer.shap_values(x_test)

# Visualize the first prediction's explanation
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test[0])


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
