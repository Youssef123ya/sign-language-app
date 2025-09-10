#!/usr/bin/env python3
"""
Sign Language Recognition Model Training Script

This script loads the sign language dataset, builds a neural network model,
trains it, and evaluates its performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


def load_dataset(csv_path):
    """Load the sign language dataset from CSV file."""
    print(f"Loading dataset from {csv_path}...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    
    return df


def prepare_data(df):
    """Separate features and labels, encode labels, and split data."""
    print("\nPreparing data...")
    
    # Separate features (pixel statistics) and labels
    feature_columns = ['pixel_min', 'pixel_mean', 'pixel_max']
    X = df[feature_columns].values
    y = df['Label'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")
    
    # Encode labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Encoded labels shape: {y_encoded.shape}")
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nData split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, label_encoder


def create_model(input_dim, num_classes):
    """Define a simple neural network model using TensorFlow/Keras."""
    print(f"\nCreating neural network model...")
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print("Model architecture:")
    model.summary()
    
    return model


def compile_and_train_model(model, X_train, y_train, X_test, y_test):
    """Compile and train the model."""
    print("\nCompiling model...")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training model...")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model's performance on the test set."""
    print("\nEvaluating model on test data...")
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Make predictions for detailed analysis
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Show some example predictions
    print(f"\nSample predictions:")
    for i in range(min(5, len(X_test))):
        actual_label = label_encoder.inverse_transform([y_test[i]])[0]
        predicted_label = label_encoder.inverse_transform([predicted_classes[i]])[0]
        confidence = np.max(predictions[i])
        print(f"Sample {i+1}: Actual={actual_label}, Predicted={predicted_label}, Confidence={confidence:.3f}")
    
    return test_accuracy


def main():
    """Main function to orchestrate the training process."""
    print("Sign Language Recognition Model Training")
    print("=" * 50)
    
    try:
        # 1. Load the dataset
        csv_path = "sign_language_formatted_dataset.csv"
        df = load_dataset(csv_path)
        
        # 2. Prepare the data
        X_train, X_test, y_train, y_test, label_encoder = prepare_data(df)
        
        # 3. Create the model
        input_dim = X_train.shape[1]
        num_classes = len(label_encoder.classes_)
        model = create_model(input_dim, num_classes)
        
        # 4. Compile and train the model
        model, history = compile_and_train_model(model, X_train, y_train, X_test, y_test)
        
        # 5. Evaluate the model
        test_accuracy = evaluate_model(model, X_test, y_test, label_encoder)
        
        print(f"\n" + "=" * 50)
        print(f"Final Model Accuracy: {test_accuracy*100:.2f}%")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()