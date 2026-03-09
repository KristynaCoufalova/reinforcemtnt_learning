"""
TabularGAN - GAN architecture for synthetic tabular data detection.
"""
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TabularDataGAN:
    """
    GAN architecture for detecting synthetic tabular data.
    Includes both the GAN model and a standalone detector.
    """
    
    def __init__(self, input_shape, latent_dim=128):
        """
        Initialize the TabularDataGAN model
        
        Args:
            input_shape: Dimension of input data (feature count)
            latent_dim: Dimension of the latent space for the generator
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Build the discriminator and generator
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Build the combined GAN model
        self.gan = self._build_gan()
        
        # Build the standalone detector for inference
        self.detector = self._build_detector()
        
    def _build_generator(self):
        """Build the generator model that creates synthetic tabular data"""
        noise = layers.Input(shape=(self.latent_dim,))
        
        # Hidden layers
        x = layers.Dense(256, activation='relu')(noise)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer matching the input shape of the discriminator
        output = layers.Dense(self.input_shape, activation='tanh')(x)
        
        # Create and compile the generator model
        generator = Model(noise, output, name="generator")
        return generator
    
    def _build_discriminator(self):
        """Build the discriminator model that detects synthetic data"""
        data_input = layers.Input(shape=(self.input_shape,))
        
        # Feature extraction layers
        x = layers.Dense(256, activation='leaky_relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(data_input)
        x = layers.Dropout(0.3)(x)
        
        # Pay special attention to patterns that might appear in synthetic data
        x = layers.Dense(256, activation='leaky_relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.3)(x)
        
        # Additional layer for capturing complex relationships
        x = layers.Dense(128, activation='leaky_relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.3)(x)
        
        # Output: 0 for synthetic, 1 for real
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Create and compile the discriminator model
        discriminator = Model(data_input, output, name="discriminator")
        discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        return discriminator
    
    def _build_gan(self):
        """Build the combined GAN model"""
        # For the combined model, we only train the generator
        self.discriminator.trainable = False
        
        # The generator takes noise as input and generates synthetic data
        noise = layers.Input(shape=(self.latent_dim,))
        synthetic_data = self.generator(noise)
        
        # The discriminator determines if the synthetic data is real or fake
        validity = self.discriminator(synthetic_data)
        
        # The combined model (stacked generator and discriminator)
        combined = Model(noise, validity, name="gan")
        combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        )
        return combined
    
    def _build_detector(self):
        """Build a standalone detector model for inference"""
        # Clone the discriminator architecture but add some additional layers
        data_input = layers.Input(shape=(self.input_shape,))
        
        # Re-use discriminator architecture with added complexity
        x = layers.Dense(256, activation='leaky_relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(data_input)
        x = layers.Dropout(0.3)(x)
        
        # Statistical feature extraction layer
        stats_layer = layers.Dense(64, activation='relu', name="stats_layer")(x)
        
        # Pattern recognition layer
        pattern_layer = layers.Dense(64, activation='relu', name="pattern_layer")(x)
        
        # Combine statistical and pattern features
        combined = layers.Concatenate()([stats_layer, pattern_layer])
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        
        # Output with confidence score
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Create and compile the detector model
        detector = Model(data_input, output, name="detector")
        detector.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0001),
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return detector
    
    def train_gan(self, real_data, epochs=1000, batch_size=32, eval_interval=100, verbose=1):
        """
        Train the GAN using adversarial training
        
        Args:
            real_data: DataFrame or array of real tabular data
            epochs: Number of training epochs
            batch_size: Batch size for training
            eval_interval: Interval to evaluate and print metrics
            verbose: Verbosity level (0, 1, or 2)
        """
        # Get real data as numpy array
        if not isinstance(real_data, np.ndarray):
            real_data = np.array(real_data)
            
        # Define labels for real and synthetic data
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Train the GAN
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of real data
            idx = np.random.randint(0, real_data.shape[0], batch_size)
            real_batch = real_data[idx]
            
            # Generate a batch of synthetic data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            synthetic_batch = self.generator.predict(noise)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(synthetic_batch, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # The generator wants the discriminator to label the generated samples as real
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)
            
            # Print progress
            if verbose > 0 and epoch % eval_interval == 0:
                print(f"Epoch {epoch}/{epochs} | D loss: {d_loss[0]:.4f}, acc: {d_loss[1]:.4f} | G loss: {g_loss:.4f}")
                
        return self
    
    def train_detector(self, real_data, synthetic_data, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the standalone detector model using labeled data
        
        Args:
            real_data: DataFrame or array of real tabular data
            synthetic_data: DataFrame or array of synthetic tabular data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level (0, 1, or 2)
        """
        # Convert inputs to numpy arrays if needed
        if not isinstance(real_data, np.ndarray):
            real_data = np.array(real_data)
        if not isinstance(synthetic_data, np.ndarray):
            synthetic_data = np.array(synthetic_data)
            
        # Combine real and synthetic data with labels
        X_data = np.vstack([real_data, synthetic_data])
        y_data = np.vstack([np.ones((real_data.shape[0], 1)), np.zeros((synthetic_data.shape[0], 1))])
        
        # Shuffle the data
        indices = np.arange(X_data.shape[0])
        np.random.shuffle(indices)
        X_data = X_data[indices]
        y_data = y_data[indices]
        
        # Split into train and validation sets
        val_samples = int(X_data.shape[0] * validation_split)
        X_train, X_val = X_data[val_samples:], X_data[:val_samples]
        y_train, y_val = y_data[val_samples:], y_data[:val_samples]
        
        # Define early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the detector
        history = self.detector.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        # Evaluate the model
        y_pred = self.detector.predict(X_val)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred_classes, average='binary'
        )
        
        print(f"Detector Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return history
    
    def detect_synthetic(self, data, threshold=0.5):
        """
        Detect if data is synthetic using the trained detector
        
        Args:
            data: DataFrame or array of tabular data to check
            threshold: Confidence threshold for classification (0.0 to 1.0)
            
        Returns:
            Dictionary with prediction results:
            - predictions: Raw prediction scores (0.0 to 1.0)
            - is_synthetic: Boolean mask indicating synthetic data (True/False)
            - confidence: Confidence scores (higher = more confident)
        """
        # Convert input to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        # Get raw predictions
        predictions = self.detector.predict(data)
        
        # Calculate results
        results = {
            'predictions': predictions,
            'is_synthetic': predictions < threshold,  # < threshold means synthetic
            'confidence': np.abs(predictions - 0.5) + 0.5  # Transform to confidence score
        }
        
        return results
    
    def save(self, directory):
        """
        Save the model to disk
        
        Args:
            directory: Directory where to save the model
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        self.generator.save(os.path.join(directory, "generator_model"))
        self.discriminator.save(os.path.join(directory, "discriminator_model"))
        self.detector.save(os.path.join(directory, "detector_model"))
        
        # Save model configuration
        config = {
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim
        }
        
        with open(os.path.join(directory, "model_config.pkl"), 'wb') as f:
            pickle.dump(config, f)
            
    @classmethod
    def load(cls, directory):
        """
        Load a saved model from disk
        
        Args:
            directory: Directory where the model is saved
            
        Returns:
            Loaded TabularDataGAN model
        """
        # Load configuration
        with open(os.path.join(directory, "model_config.pkl"), 'rb') as f:
            config = pickle.load(f)
            
        # Create a new instance
        instance = cls(**config)
        
        # Load models
        instance.generator = load_model(os.path.join(directory, "generator_model"))
        instance.discriminator = load_model(os.path.join(directory, "discriminator_model"))
        instance.detector = load_model(os.path.join(directory, "detector_model"))
        
        # Rebuild GAN
        instance.gan = instance._build_gan()
        
        return instance