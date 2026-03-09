import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

class TabularWGAN_GP:
    def __init__(self, input_shape, latent_dim=128, gradient_penalty_weight=10.0):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.gradient_penalty_weight = gradient_penalty_weight
        
        self.generator = self._build_generator()
        self.critic = self._build_critic()
        self.detector = self._build_detector()

        # Optimizers
        self.critic_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.generator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    def _build_generator(self):
        noise = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(noise)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        output = layers.Dense(self.input_shape, activation='tanh')(x)
        generator = Model(noise, output, name="generator")
        return generator

    def _build_critic(self):
        data_input = layers.Input(shape=(self.input_shape,))
        x = layers.Dense(256, activation='relu')(data_input)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        output = layers.Dense(1)(x)
        critic = Model(data_input, output, name="critic")
        return critic

    def _build_detector(self):
        data_input = layers.Input(shape=(self.input_shape,))
        x = layers.Dense(256, activation='relu')(data_input)
        x = layers.Dropout(0.3)(x)
        stats_layer = layers.Dense(64, activation='relu', name="stats_layer")(x)
        pattern_layer = layers.Dense(64, activation='relu', name="pattern_layer")(x)
        combined = layers.Concatenate()([stats_layer, pattern_layer])
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        detector = Model(data_input, output, name="detector")
        detector.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0001),
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return detector

    def _gradient_penalty(self, real_samples, fake_samples):
        # Interpolate between real and fake samples
        alpha = tf.random.uniform((real_samples.shape[0], 1), 0.0, 1.0)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        gp = tf.reduce_mean((grads_norm - 1.0) ** 2)
        return gp

    def train_wgan_gp(self, real_data, epochs=1000, batch_size=32, eval_interval=100, verbose=1):
        if not isinstance(real_data, np.ndarray):
            real_data = np.array(real_data)
        
        for epoch in range(epochs):
            for _ in range(5):  # Train critic 5x
                idx = np.random.randint(0, real_data.shape[0], batch_size)
                real_batch = real_data[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_batch = self.generator.predict(noise)

                with tf.GradientTape() as tape:
                    real_validity = self.critic(real_batch)
                    fake_validity = self.critic(fake_batch)
                    gp = self._gradient_penalty(real_batch, fake_batch)
                    critic_loss = tf.reduce_mean(fake_validity) - tf.reduce_mean(real_validity) + self.gradient_penalty_weight * gp

                grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
            
            # Train generator 1x
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_batch = self.generator(noise)
                fake_validity = self.critic(fake_batch)
                generator_loss = -tf.reduce_mean(fake_validity)
            grads = tape.gradient(generator_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
            
            # Progress
            if verbose > 0 and epoch % eval_interval == 0:
                print(f"Epoch {epoch}/{epochs} | Critic loss: {critic_loss:.4f} | Generator loss: {generator_loss:.4f}")

    def train_detector(self, real_data, synthetic_data, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        if not isinstance(real_data, np.ndarray):
            real_data = np.array(real_data)
        if not isinstance(synthetic_data, np.ndarray):
            synthetic_data = np.array(synthetic_data)
        
        X_data = np.vstack([real_data, synthetic_data])
        y_data = np.vstack([np.ones((real_data.shape[0], 1)), np.zeros((synthetic_data.shape[0], 1))])
        
        indices = np.arange(X_data.shape[0])
        np.random.shuffle(indices)
        X_data = X_data[indices]
        y_data = y_data[indices]
        
        val_samples = int(X_data.shape[0] * validation_split)
        X_train, X_val = X_data[val_samples:], X_data[:val_samples]
        y_train, y_val = y_data[val_samples:], y_data[:val_samples]
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.detector.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        y_pred = self.detector.predict(X_val)
        y_pred_classes = (y_pred > 0.5).astype(int)
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
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        predictions = self.detector.predict(data)
        results = {
            'predictions': predictions,
            'is_synthetic': predictions < threshold,
            'confidence': np.abs(predictions - 0.5) + 0.5
        }
        return results
