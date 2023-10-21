import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Build the model for classification
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

# Compile the model for classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming y_train contains labels 0, 1, and 2. Convert to one-hot encoding
y_train_encoded = to_categorical(y_train, 3)
y_test_encoded = to_categorical(y_test, 3)

# Calculate class weights for the imbalanced dataset
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)
class_weights = dict(enumerate(class_weights))

# Train the model on the training data
history = model.fit(x_train, y_train_encoded, epochs=70, verbose=0, class_weight=class_weights)

# Predict the class labels for the test data
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Print classification metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualization code remains similar to before, but ensure accuracy is displayed instead of MAE
plt.figure(figsize=(20,10))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Loss and Accuracy Across Epochs')
plt.ylabel('Loss / Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

----------
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

# Define a function to create the model, required for KerasClassifier
def create_model(learning_rate=0.01, neurons_layer1=32, neurons_layer2=64):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons_layer1, activation='relu'),
        tf.keras.layers.Dense(neurons_layer2, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the KerasClassifier wrapper for the create_model function
model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=16, verbose=0)

# Define the hyperparameters search space
param_dist = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'neurons_layer1': [16, 32, 64, 128],
    'neurons_layer2': [32, 64, 128]
}

# Random search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2)
random_search_result = random_search.fit(x_train, y_train_encoded)

# Print the best hyperparameters
print("Best Score: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))

# Visualization of the parameters over the epochs
results = random_search_result.cv_results_

# Plotting Learning Rate over Epochs
plt.figure(figsize=(15,6))
for mean, stdev, param in zip(results['mean_test_score'], results['std_test_score'], results['params']):
    plt.plot(range(1,31), [mean]*30, label='Learning Rate: ' + str(param['learning_rate']))
    plt.fill_between(range(1,31), mean-stdev, mean+stdev, alpha=0.1)

plt.title('Learning Rate over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Accuracy')
plt.legend(loc='best')
plt.show()


----------
#need to be tested

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt

# Define the function to create the Keras model
def create_model(learning_rate=0.01):
    model = keras.Sequential([
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Dense(units=3, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model

# Create the KerasRegressor wrapper
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the hyperparameter search space
param_grid = {
    'learning_rate': np.arange(0.001, 0.1, 0.001),
    'epochs': np.arange(50, 200, 10),
    'batch_size': np.arange(16, 64, 8)
}

# Use RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    verbose=2
)
random_search.fit(x_train, y_train)

# Print best hyperparameters
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean test score: ", random_search.best_score_)

# Use the best hyperparameters to train the model over different numbers of epochs
best_params = random_search.best_params_
epochs_list = [10, 30, 50, 70, 100]
histories = []

for e in epochs_list:
    best_model = create_model(learning_rate=best_params['learning_rate'])
    history = best_model.fit(x_train, y_train, epochs=e, batch_size=best_params['batch_size'], validation_data=(x_test, y_test))
    histories.append(history)

# Plot the training and test loss and accuracy
plt.figure(figsize=(20,10))
for i, history in enumerate(histories):
    plt.subplot(2, 3, i+1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Epochs = ' + str(epochs_list[i]))
    plt.ylabel('Loss / MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Test loss', 'Train MAE', 'Test MAE'], loc='upper right')
plt.tight_layout()
plt.show()
