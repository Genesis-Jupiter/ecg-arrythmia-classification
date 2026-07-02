import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D,
                                     BatchNormalization, Bidirectional,
                                     LSTM, Dense, Dropout, GlobalAveragePooling1D, 
                                     Reshape, Multiply, Add, Activation)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Hardcoded Parameters
BEAT_LEN = 375
EPOCHS = 10
BATCH_SIZE = 250
LEARNING_RATE = 0.001

def se_block(x, filters):
    se = GlobalAveragePooling1D()(x)
    se = Dense(filters // 8, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, filters))(se)
    return Multiply()([x, se])

def residual_block(x, filters):
    shortcut = x
    x = Conv1D(filters, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = se_block(x, filters)
    
    shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    x = Add()([x, shortcut])
    
    # Updated: Using Keras layer instead of raw TF math operation
    x = Activation('relu')(x)
    
    return x

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 96)
    x = residual_block(x, 128)
    x = residual_block(x, 160)

    # BiLSTM layers
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def train_model(data_path, model_dir):
    try:
        logger.info(f"Loading processed data from {data_path}...")
        data = np.load(data_path)
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        
        num_classes = len(np.unique(y_train))
        logger.info(f"Data loaded. Found {num_classes} classes. X_train shape: {X_train.shape}")
        
        # Compute class weights to handle dataset imbalance
        logger.info("Computing class weights...")
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(enumerate(weights))

        logger.info("Building the 1D-CNN + BiLSTM model architecture...")
        model = build_model((BEAT_LEN, 1), num_classes)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True, 
            verbose=1
        )

        logger.info("Starting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Save the model - Updated to .h5 format
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, "ecg_model.h5")
        model.save(model_save_path)
        logger.info(f"Model training completed! Saved to {model_save_path}")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Pipeline Step")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=os.path.join("data", "interim", "processed_ecg.npz"),
        help="Path to the processed data"
    )
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="models",
        help="Directory to save the trained model"
    )
    args = parser.parse_args()
    
    train_model(args.data_path, args.model_dir)