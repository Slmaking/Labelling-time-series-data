from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_categorical = to_categorical(y_train, num_classes=3)
y_test_categorical = to_categorical(y_test, num_classes=3)

# Calculate class weights for imbalanced classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define hyperparameters grid to search
lstm_units = [50, 100, 150]
batch_sizes = [32, 64, 128]
epochs = [50, 100]
dropout_rates = [0.0, 0.2, 0.5]  # Assuming you want to add a Dropout layer

best_accuracy = 0
best_params = {}

for units in lstm_units:
    for batch_size in batch_sizes:
        for epoch in epochs:
            for dropout_rate in dropout_rates:
                print(f"Training model with {units} LSTM units, batch size {batch_size}, {epoch} epochs, and {dropout_rate} dropout rate.")

                # Define model architecture
                model = Sequential()
                model.add(LSTM(units=units, input_shape=(5,1), dropout=dropout_rate))
                model.add(Dense(units=3, activation='softmax'))

                # Compile the model
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                # Fit the model
                history = model.fit(X_train, y_train_categorical, epochs=epoch, batch_size=batch_size,
                                    class_weight=class_weight_dict, validation_split=0.2, verbose=0)

                # Evaluate the model on the test set
                _, accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)

                # Update best params if current model is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'units': units,
                        'batch_size': batch_size,
                        'epochs': epoch,
                        'dropout_rate': dropout_rate
                    }

print(f"Best accuracy: {best_accuracy}")
print(f"Best parameters: {best_params}")




import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Sample hyperparameters for demonstration
lstm_units_options = [50, 100]
dropout_rate_options = [0.2, 0.5]
batch_size = 64  # Fixed for simplicity
epochs = 50  # Fixed number of epochs

# Assuming X_train, y_train have already been defined and preprocessed

# Convert labels to categorical (one-hot encoding)
y_train_categorical = to_categorical(y_train, num_classes=3)

# Calculate class weights for imbalanced classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Training models with different hyperparameters
for lstm_units in lstm_units_options:
    for dropout_rate in dropout_rate_options:
        print(f"Training model with {lstm_units} LSTM units and {dropout_rate} dropout rate.")

        # Define the model architecture
        model = Sequential()
        model.add(LSTM(units=lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=dropout_rate))
        model.add(Dense(units=3, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit the model
        history = model.fit(
            X_train, y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            validation_split=0.2,
            verbose=1
        )

        # Plotting the accuracy
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Accuracy over 50 epochs\nLSTM Units: {lstm_units}, Dropout: {dropout_rate}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plotting the loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss over 50 epochs\nLSTM Units: {lstm_units}, Dropout: {dropout_rate}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()


