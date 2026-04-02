"""
Deep Autoencoder for Feature Extraction and Dimensionality Reduction
- Learns compressed representations of property features
- Used for both feature extraction (SNN input) and fraud detection (reconstruction error)
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import joblib
from config import AUTOENCODER_CONFIG, MODEL_DIR


class DeepAutoencoder:
    def __init__(self, input_dim, encoding_dim=None):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim or AUTOENCODER_CONFIG["encoding_dim"]
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.history = None
        self._build_model()

    def _build_model(self):
        """Build deep autoencoder architecture"""
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,), name="encoder_input")

        # Encoding layers with decreasing dimensions
        dim1 = max(self.input_dim * 2, 64)
        dim2 = max(self.input_dim, 32)
        dim3 = max(self.input_dim // 2, 16)

        x = layers.Dense(dim1, activation='relu', name="enc_dense_1")(input_layer)
        x = layers.BatchNormalization(name="enc_bn_1")(x)
        x = layers.Dropout(0.2, name="enc_dropout_1")(x)

        x = layers.Dense(dim2, activation='relu', name="enc_dense_2")(x)
        x = layers.BatchNormalization(name="enc_bn_2")(x)
        x = layers.Dropout(0.2, name="enc_dropout_2")(x)

        x = layers.Dense(dim3, activation='relu', name="enc_dense_3")(x)
        x = layers.BatchNormalization(name="enc_bn_3")(x)

        # Bottleneck (encoded representation)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name="bottleneck")(x)

        # Decoder (mirror of encoder)
        x = layers.Dense(dim3, activation='relu', name="dec_dense_1")(encoded)
        x = layers.BatchNormalization(name="dec_bn_1")(x)
        x = layers.Dropout(0.2, name="dec_dropout_1")(x)

        x = layers.Dense(dim2, activation='relu', name="dec_dense_2")(x)
        x = layers.BatchNormalization(name="dec_bn_2")(x)
        x = layers.Dropout(0.2, name="dec_dropout_2")(x)

        x = layers.Dense(dim1, activation='relu', name="dec_dense_3")(x)
        x = layers.BatchNormalization(name="dec_bn_3")(x)

        decoded = layers.Dense(self.input_dim, activation='sigmoid', name="decoder_output")(x)

        # Build models
        self.autoencoder = Model(input_layer, decoded, name="autoencoder")
        self.encoder = Model(input_layer, encoded, name="encoder")

        # Compile
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=AUTOENCODER_CONFIG["learning_rate"]),
            loss='mse',
            metrics=['mae']
        )

        print(f"[Autoencoder] Built model: {self.input_dim} -> {self.encoding_dim} -> {self.input_dim}")

    def train(self, X_train, X_val=None, epochs=None, batch_size=None):
        """Train the autoencoder"""
        epochs = epochs or AUTOENCODER_CONFIG["epochs"]
        batch_size = batch_size or AUTOENCODER_CONFIG["batch_size"]

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        if X_val is not None:
            self.history = self.autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, X_val),
                callbacks=callbacks,
                verbose=0
            )
        else:
            self.history = self.autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=AUTOENCODER_CONFIG["validation_split"],
                callbacks=callbacks,
                verbose=0
            )

        final_loss = self.history.history['loss'][-1]
        print(f"[Autoencoder] Training complete. Final loss: {final_loss:.6f}")
        return self.history

    def encode(self, X):
        """Extract encoded features"""
        return self.encoder.predict(X, verbose=0)

    def reconstruct(self, X):
        """Reconstruct input"""
        return self.autoencoder.predict(X, verbose=0)

    def reconstruction_error(self, X):
        """Calculate reconstruction error for each sample"""
        reconstructed = self.reconstruct(X)
        errors = np.mean(np.square(X - reconstructed), axis=1)
        return errors

    def get_training_history(self):
        """Return training history as dict"""
        if self.history is None:
            return {}
        return {
            "loss": [float(v) for v in self.history.history['loss']],
            "val_loss": [float(v) for v in self.history.history.get('val_loss', [])],
            "mae": [float(v) for v in self.history.history.get('mae', [])],
            "val_mae": [float(v) for v in self.history.history.get('val_mae', [])],
        }

    def save_model(self):
        """Save the autoencoder model"""
        self.autoencoder.save(os.path.join(MODEL_DIR, "autoencoder.keras"))
        self.encoder.save(os.path.join(MODEL_DIR, "encoder.keras"))
        print("[Autoencoder] Models saved")

    def load_model(self):
        """Load the autoencoder model"""
        self.autoencoder = keras.models.load_model(os.path.join(MODEL_DIR, "autoencoder.keras"))
        self.encoder = keras.models.load_model(os.path.join(MODEL_DIR, "encoder.keras"))
        print("[Autoencoder] Models loaded")

    def summary(self):
        """Print model summary"""
        return {
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "total_params": self.autoencoder.count_params(),
            "architecture": [
                {"layer": l.name, "output_shape": str(l.output_shape), "params": l.count_params()}
                for l in self.autoencoder.layers
            ]
        }
