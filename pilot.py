import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Define Helper Functions ---

def create_model(input_shape, num_classes=5, l2_penalty=0.001):
    """
    Creates and compiles a fresh instance of the CNN model.
    """
    tf.keras.backend.clear_session() # Clear previous models from memory.

    # model = models.Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(16, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_penalty)),
    #     layers.BatchNormalization(),
    #     layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_penalty)),
    #     layers.BatchNormalization(),
    #     layers.MaxPooling2D((2, 1)),
    #     layers.Dropout(0.4),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_penalty)),
    #     layers.Dropout(0.4),
    #     layers.Dense(num_classes, activation='softmax')
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # return model
    def EEGNet_Corrected(num_classes=5, F1=16, D=2, F2=32, dropout_rate=0.5, l2_penalty=0.001):
        """
        A corrected and regularized EEGNet for smaller datasets.
        Fixes the kernel_regularizer argument for depthwise and separable layers.
        """
        inputs = layers.Input(shape=(30, 5, 1))

        # Define the regularizer
        reg = regularizers.l2(l2_penalty)

        # Block 1: Temporal and Spatial Filtering
        # ------------------------------------------
        # Conv2D uses 'kernel_regularizer'
        x = layers.Conv2D(F1, (1, 5), padding='same', use_bias=False, kernel_regularizer=reg)(inputs)
        x = layers.BatchNormalization()(x)

        # DepthwiseConv2D uses 'depthwise_regularizer' -- THIS IS THE FIX
        x = layers.DepthwiseConv2D((30, 1), use_bias=False, depth_multiplier=D, padding='valid',
                                   depthwise_regularizer=reg)(x)  # <-- FIX HERE
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AveragePooling2D((1, 2))(x)
        x = layers.Dropout(dropout_rate)(x)

        # Block 2: Separable Convolution
        # ------------------------------------------
        # SeparableConv2D has two regularizers: 'depthwise_regularizer' and 'pointwise_regularizer'
        x = layers.SeparableConv2D(F2, (1, 3), padding='same', use_bias=False,
                                   depthwise_regularizer=reg,
                                   pointwise_regularizer=reg)(x)  # <-- FIX HERE
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)

        x = layers.Flatten()(x)

        # Classifier Block
        # ------------------------------------------
        # Dense layer uses 'kernel_regularizer'
        outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg)(x)

        model = models.Model(inputs, outputs)
        return model

    # --- How to create and compile it ---
    model = EEGNet_Corrected(num_classes=5, dropout_rate=0.6, l2_penalty=0.01)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def stratified_split(X, y, train_ratio=0.7):
    """
    Performs a stratified split of the data to maintain class proportions.
    """
    num_classes = y.shape[1]
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list   = [], []

    for class_idx in range(num_classes):
        class_mask = np.where(y[:, class_idx] == 1)[0]
        np.random.shuffle(class_mask) # Shuffle indices for randomness

        n_train = int(len(class_mask) * train_ratio)
        train_idx = class_mask[:n_train]
        test_idx  = class_mask[n_train:]

        X_train_list.append(X[train_idx])
        y_train_list.append(y[train_idx])
        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])

    # Concatenate and shuffle the final datasets
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test  = np.concatenate(X_test_list, axis=0)
    y_test  = np.concatenate(y_test_list, axis=0)

    train_perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[train_perm], y_train[train_perm]

    test_perm  = np.random.permutation(len(X_test))
    X_test, y_test   = X_test[test_perm], y_test[test_perm]

    return X_train, y_train, X_test, y_test

# --- 2. Main Training Loop ---

def main():
    """
    Main function to run the training and evaluation loop.
    """
    # --- Configuration ---
    # IMPORTANT: Update this path to where your .pkl files are stored.
    DATA_PATH = "/Users/anuraagsrivatsa/Documents/Capstone/EAV/EAV/processed_data_psd/"
    OUTPUT_FILE = "subject_accuracies.txt"
    NUM_SUBJECTS = 42
    EPOCHS = 300
    BATCH_SIZE = 32

    all_accuracies = []

    print("Starting training process for all subjects...")

    # Loop through each subject file from 1 to 42
    for subject_id in range(1, NUM_SUBJECTS + 1):
        file_path = os.path.join(DATA_PATH, f"subject{subject_id}_processed_psd.pkl")

        # Check if the data file exists
        if not os.path.exists(file_path):
            print(f"--> [Warning] Subject {subject_id}: Data file not found. Skipping.")
            continue

        try:
            print(f"\n--- Processing Subject {subject_id}/{NUM_SUBJECTS} ---")

            # 1. Load Data
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            X = data["features"]
            y = data["labels"]

            # Add a channel dimension for the Conv2D layers (samples, 30, 5) -> (samples, 30, 5, 1)
            X = np.expand_dims(X, axis=-1)

            # 2. Split Data
            X_train, y_train, X_test, y_test = stratified_split(X, y, train_ratio=0.7)
            print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            # 3. Create a new model instance for this subject
            input_shape = X_train.shape[1:]
            model = create_model(input_shape)

            # 4. Train the model
            print(f"Training model for Subject {subject_id}...")
            early_stopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=150, restore_best_weights=True)

            model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test),
                shuffle=True,
                verbose=0,  # Use verbose=0 or 1 to reduce console output
                callbacks=[early_stopper]
            )

            # 5. Evaluate and store the test accuracy
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            all_accuracies.append((subject_id, accuracy)) # Store as a tuple
            print(f"✅ Subject {subject_id}: Test Accuracy = {accuracy:.4f}")

        except Exception as e:
            print(f"--> [Error] An error occurred while processing subject {subject_id}: {e}")

    # --- 3. Save Results to File ---
    print("\n--- Training complete. Saving all accuracies. ---")
    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write("Subject Test Accuracies\n")
            f.write("------------------------\n")
            for subject_id, acc in all_accuracies:
                f.write(f"Subject {subject_id}: {acc:.4f}\n")

            # Calculate and write the average accuracy
            if all_accuracies:
                avg_accuracy = np.mean([acc for _, acc in all_accuracies])
                f.write("------------------------\n")
                f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")

        print(f"Results successfully saved to '{OUTPUT_FILE}'")
    except Exception as e:
        print(f"--> [Error] Could not write results to file: {e}")

# --- Run the main function ---
if __name__ == "__main__":
    main()