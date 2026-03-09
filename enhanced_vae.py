import argparse
import os
import random
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VAEConfig:
    """Configuration class for VAE parameters"""
    # Architecture
    input_size: int = 5
    latent_dim: int = 64  # Reduced from 128 for simpler data
    hidden_dims: List[int] = None  # Will be set in __post_init__
    dropout: float = 0.2  # Reduced dropout
    activation: str = 'leaky_relu'
    
    # Regularization
    kl_weight: float = 0.1  # Reduced from 0.5
    l2_reg: float = 1e-5
    spectral_norm: bool = True
    
    # Training
    train_split: float = 0.8
    val_split: float = 0.1  # Added validation split
    epochs: int = 200
    batch_size: int = 64  # Increased batch size
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_type: str = 'cosine'  # 'plateau', 'cosine', or None
    early_stopping_patience: int = 20
    gradient_clip: float = 1.0
    
    # Evaluation
    eval_freq: int = 5
    n_samples_generation: int = 1000
    
    # Paths
    save_dir: str = "./models"
    model_name: str = "enhanced_vae_model.pth"
    data_path: str = "/path/to/data_banknote_authentication.txt"
    
    # Anomaly detection
    anomaly_threshold: float = None  # Will be computed automatically
    threshold_percentile: float = 95.0  # Percentile for threshold
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]
        
        # Create save directory
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return F.leaky_relu(x + self.net(x), 0.2)


class EnhancedEncoder(nn.Module):
    """Enhanced encoder with residual connections and better architecture"""
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Build encoder layers
        layers = []
        in_dim = config.input_size
        
        for hidden_dim in config.hidden_dims:
            layer = nn.Linear(in_dim, hidden_dim)
            if config.spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            layers.extend([
                layer,
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout)
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Add residual block for the last hidden layer
        self.residual = ResidualBlock(config.hidden_dims[-1], config.dropout)
        
        # Output layers for mean and log variance
        self.mu_layer = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        self.logvar_layer = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log variance"""
        h = self.encoder(x)
        h = self.residual(h)
        
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar


class EnhancedDecoder(nn.Module):
    """Enhanced decoder with residual connections"""
    
    def __init__(self, config: VAEConfig):
        
        super().__init__()
        self.config = config
        
        # Build decoder layers (reverse of encoder)
        layers = []
        hidden_dims = list(reversed(config.hidden_dims))
        in_dim = config.latent_dim
        
        for hidden_dim in hidden_dims:
            layer = nn.Linear(in_dim, hidden_dim)
            if config.spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            layers.extend([
                layer,
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout)
            ])
            in_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Add residual block
        self.residual = ResidualBlock(hidden_dims[-1], config.dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], config.input_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder"""
        h = self.decoder(z)
        h = self.residual(h)
        return self.output_layer(h)


class EnhancedVAE(nn.Module):
    """Enhanced VAE with better loss function and anomaly detection capabilities"""
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = EnhancedEncoder(config)
        self.decoder = EnhancedDecoder(config)
        
        # For tracking training progress
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []
        }
        
        # For anomaly detection
        self.reconstruction_threshold = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def compute_loss(self, x: torch.Tensor, recon_x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute VAE loss with improved formulation"""
        batch_size = x.size(0)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss with annealing
        total_loss = recon_loss + self.config.kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error for anomaly detection"""
        self.eval()
        with torch.no_grad():
            recon_x, _, _ = self.forward(x)
            error = F.mse_loss(recon_x, x, reduction='none').sum(dim=1)
        return error
    
    def fit_threshold(self, normal_data_loader: DataLoader, percentile: float = 95.0):
        """Fit anomaly detection threshold using normal data"""
        errors = []
        self.eval()
        
        with torch.no_grad():
            for batch in normal_data_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                error = self.get_reconstruction_error(x)
                errors.append(error.cpu())
        
        all_errors = torch.cat(errors, dim=0).numpy()
        self.reconstruction_threshold = np.percentile(all_errors, percentile)
        logger.info(f"Anomaly threshold set to: {self.reconstruction_threshold:.4f}")
    
    def predict_anomaly(self, x: torch.Tensor, return_scores: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict anomalies based on reconstruction error"""
        if self.reconstruction_threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold() first.")
        
        errors = self.get_reconstruction_error(x)
        predictions = (errors > self.reconstruction_threshold).int()
        
        if return_scores:
            return predictions, errors
        return predictions
    
    def generate_samples(self, n_samples: int) -> torch.Tensor:
        """Generate new samples from the learned distribution"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.latent_dim)
            samples = self.decoder(z)
        return samples
    
    def save_model(self, path: Optional[str] = None):
        """Save complete model state"""
        if path is None:
            path = os.path.join(self.config.save_dir, self.config.model_name)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'reconstruction_threshold': self.reconstruction_threshold,
            'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str):
        """Load complete model state"""
        checkpoint = torch.load(path, map_location='cpu')
        
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_history = checkpoint.get('training_history', {})
        model.reconstruction_threshold = checkpoint.get('reconstruction_threshold')
        
        # Restore scaler if available
        if checkpoint.get('scaler_mean') is not None:
            model.scaler.mean_ = checkpoint['scaler_mean']
            model.scaler.scale_ = checkpoint['scaler_scale']
            model.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return model


class VAETrainer:
    """Trainer class for VAE with advanced training features"""
    
    def __init__(self, model: EnhancedVAE, config: VAEConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        
        if config.scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5
            )
        elif config.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': [], 'recon': [], 'kl': []}
        
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            
            self.optimizer.zero_grad()
            
            recon_x, mu, logvar = self.model(x)
            losses = self.model.compute_loss(x, recon_x, mu, logvar)
            
            losses['total_loss'].backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Store losses
            epoch_losses['total'].append(losses['total_loss'].item())
            epoch_losses['recon'].append(losses['recon_loss'].item())
            epoch_losses['kl'].append(losses['kl_loss'].item())
        
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_losses = {'total': [], 'recon': [], 'kl': []}
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                
                recon_x, mu, logvar = self.model(x)
                losses = self.model.compute_loss(x, recon_x, mu, logvar)
                
                val_losses['total'].append(losses['total_loss'].item())
                val_losses['recon'].append(losses['recon_loss'].item())
                val_losses['kl'].append(losses['kl_loss'].item())
        
        return {k: np.mean(v) for k, v in val_losses.items()}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> EnhancedVAE:
        """Full training loop with early stopping"""
        logger.info(f"Starting training on {self.device}")
        
        for epoch in range(self.config.epochs):
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            if epoch % self.config.eval_freq == 0:
                val_losses = self.validate(val_loader)
                
                # Store history
                self.model.training_history['train_loss'].append(train_losses['total'])
                self.model.training_history['val_loss'].append(val_losses['total'])
                self.model.training_history['recon_loss'].append(train_losses['recon'])
                self.model.training_history['kl_loss'].append(train_losses['kl'])
                
                # Logging
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {train_losses['total']:.4f} | "
                    f"Val Loss: {val_losses['total']:.4f} | "
                    f"Recon: {train_losses['recon']:.4f} | "
                    f"KL: {train_losses['kl']:.4f}"
                )
                
                # Early stopping and model saving
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.patience_counter = 0
                    self.model.save_model()
                    logger.info("New best model saved!")
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Scheduler step
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_losses['total'])
                    else:
                        self.scheduler.step()
        
        return self.model


class VAEAnalyzer:
    """Analysis and visualization tools for VAE"""
    
    def __init__(self, model: EnhancedVAE):
        self.model = model
    
    def plot_training_history(self):
        """Plot training history"""
        history = self.model.training_history
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Validation')
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Reconstruction loss
        axes[1].plot(history['recon_loss'], label='Reconstruction')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        
        # KL loss
        axes[2].plot(history['kl_loss'], label='KL Divergence')
        axes[2].set_title('KL Divergence')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_reconstructions(self, data_loader: DataLoader, n_samples: int = 100):
        """Plot original vs reconstructed data"""
        self.model.eval()
        
        # Get sample data
        for batch in data_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if len(x) >= n_samples:
                x = x[:n_samples]
                break
        
        with torch.no_grad():
            recon_x, _, _ = self.model(x)
        
        x_np = x.cpu().numpy()
        recon_np = recon_x.cpu().numpy()
        
        n_features = x_np.shape[1]
        fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 8))
        if n_features <= 2:
            axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i] if n_features > 2 else axes[i]
            
            ax.hist(x_np[:, i], bins=30, alpha=0.7, label='Original', density=True)
            ax.hist(recon_np[:, i], bins=30, alpha=0.7, label='Reconstructed', density=True)
            
            ax.set_title(f'Feature {i+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_anomaly_detection(self, normal_loader: DataLoader, 
                                  anomaly_loader: DataLoader) -> Dict[str, float]:
        """Evaluate anomaly detection performance"""
        # Get reconstruction errors for normal data
        normal_errors = []
        with torch.no_grad():
            for batch in normal_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                errors = self.model.get_reconstruction_error(x)
                normal_errors.append(errors.cpu())
        normal_errors = torch.cat(normal_errors, dim=0).numpy()
        
        # Get reconstruction errors for anomalous data
        anomaly_errors = []
        with torch.no_grad():
            for batch in anomaly_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                errors = self.model.get_reconstruction_error(x)
                anomaly_errors.append(errors.cpu())
        anomaly_errors = torch.cat(anomaly_errors, dim=0).numpy()
        
        # Create labels
        y_true = np.concatenate([
            np.zeros(len(normal_errors)),  # Normal = 0
            np.ones(len(anomaly_errors))   # Anomaly = 1
        ])
        y_scores = np.concatenate([normal_errors, anomaly_errors])
        
        # Calculate metrics
        auc_score = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Plot distributions
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', density=True)
        if self.model.reconstruction_threshold:
            plt.axvline(self.model.reconstruction_threshold, color='red', 
                       linestyle='--', label='Threshold')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AUC: {pr_auc:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'roc_auc': auc_score,
            'pr_auc': pr_auc,
            'normal_error_mean': np.mean(normal_errors),
            'normal_error_std': np.std(normal_errors),
            'anomaly_error_mean': np.mean(anomaly_errors),
            'anomaly_error_std': np.std(anomaly_errors)
        }


def load_and_preprocess_data(config: VAEConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and preprocess the banknote authentication dataset"""
    # Load data
    df = pd.read_csv(config.data_path, header=None)
    X_normal = df.iloc[:, :].values
    # Separate features and labels
    #X = df.iloc[:, :-1].values
    #y = df.iloc[:, -1].values
    
    # Use only authentic banknotes for training (label = 0)
    #normal_indices = np.where(y == 0)[0]
    #anomaly_indices = np.where(y == 1)[0]
    
    #X_normal = X[normal_indices]
    #X_anomaly = X[anomaly_indices]
    
    # Normalize features
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    #X_anomaly_scaled = scaler.transform(X_anomaly)
    
    # Create train/validation split from normal data
    normal_tensor = torch.tensor(X_normal_scaled, dtype=torch.float32)
    normal_dataset = TensorDataset(normal_tensor)
    
    train_size = int(config.train_split * len(normal_dataset))
    val_size = int(config.val_split * len(normal_dataset))
    test_size = len(normal_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_normal_dataset = random_split(
        normal_dataset, [train_size, val_size, test_size]
    )
    
    # Create anomaly test dataset
    #anomaly_tensor = torch.tensor(X_anomaly_scaled, dtype=torch.float32)
    #test_anomaly_dataset = TensorDataset(anomaly_tensor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    # Combine normal and anomaly test data
    #test_dataset = torch.utils.data.ConcatDataset([test_normal_dataset, test_anomaly_dataset])
    test_dataset = torch.utils.data.ConcatDataset([test_normal_dataset])
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def create_config_for_experiment(experiment_name: str = "default") -> VAEConfig:
    """Create different configurations for various experiments"""
    
    configs = {
        "lightweight": VAEConfig(
            input_size=5,
            latent_dim=16,
            hidden_dims=[32, 16],
            epochs=50,
            batch_size=32,
            lr=1e-3,
            kl_weight=0.05,
            model_name="lightweight_vae.pth"
        ),
        
        "deep": VAEConfig(
            input_size=5,
            latent_dim=64,
            hidden_dims=[128, 64, 32, 16],
            epochs=200,
            batch_size=64,
            lr=5e-4,
            kl_weight=0.1,
            dropout=0.3,
            model_name="deep_vae.pth"
        ),
        
        "robust": VAEConfig(
            input_size=5,
            latent_dim=32,
            hidden_dims=[64, 32],
            epochs=150,
            batch_size=64,
            lr=1e-3,
            kl_weight=0.01,  # Lower KL weight for better reconstruction
            spectral_norm=True,
            gradient_clip=0.5,
            model_name="robust_vae.pth"
        ),
        
        "default": VAEConfig(
            input_size=5,
            latent_dim=32,
            hidden_dims=[64, 32],
            epochs=100,
            batch_size=64,
            lr=1e-3,
            data_path="data_banknote_authentication.txt"
        )
    }
    
    return configs.get(experiment_name, configs["default"])


def run_experiment(config: VAEConfig, experiment_name: str = "experiment"):
    """Run a complete VAE experiment"""
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Configuration: {config}")
    
    try:
        # Load data
        train_loader, val_loader, test_loader = load_and_preprocess_data(config)
        logger.info("Data loaded successfully")
        
        # Create and train model
        model = EnhancedVAE(config)
        trainer = VAETrainer(model, config)
        trained_model = trainer.train(train_loader, val_loader)
        
        # Fit threshold
        trained_model.fit_threshold(val_loader, percentile=config.threshold_percentile)
        
        # Analysis
        analyzer = VAEAnalyzer(trained_model)
        analyzer.plot_training_history()
        analyzer.plot_reconstructions(val_loader)
        
        # Generate samples
        generated_samples = trained_model.generate_samples(100)
        logger.info(f"Generated {len(generated_samples)} samples")
        
        # Save final model
        final_path = os.path.join(config.save_dir, f"final_{experiment_name}_{config.model_name}")
        trained_model.save_model(final_path)
        
        logger.info(f"Experiment {experiment_name} completed successfully!")
        return trained_model, analyzer
        
    except Exception as e:
        logger.error(f"Experiment {experiment_name} failed: {e}")
        raise


def main():
    """Main training and evaluation pipeline"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Parse arguments for different experiments
    parser = argparse.ArgumentParser(description="Enhanced VAE for Anomaly Detection")
    parser.add_argument("--experiment", type=str, default="default", 
                       choices=["lightweight", "deep", "robust", "default"],
                       help="Experiment configuration to run")
    parser.add_argument("--data_path", type=str, 
                       default="data_banknote_authentication.txt",
                       help="Path to data file")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Get configuration
    config = create_config_for_experiment(args.experiment)
    config.data_path = args.data_path
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Run experiment
    model, analyzer = run_experiment(config, args.experiment)
    
    # Load data
    try:
        train_loader, val_loader, test_loader = load_and_preprocess_data(config)
        logger.info(f"Data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Create model and trainer
    model = EnhancedVAE(config)
    trainer = VAETrainer(model, config)
    
    # Train model
    trained_model = trainer.train(train_loader, val_loader)
    
    # Fit anomaly detection threshold
    trained_model.fit_threshold(val_loader, percentile=config.threshold_percentile)
    
    # Analysis
    analyzer = VAEAnalyzer(trained_model)
    
    # Plot training history
    analyzer.plot_training_history()
    
    # Plot reconstructions
    analyzer.plot_reconstructions(val_loader)
    
    # If you have separate normal and anomaly test sets, evaluate:
    # metrics = analyzer.evaluate_anomaly_detection(normal_test_loader, anomaly_test_loader)
    # logger.info(f"Anomaly detection metrics: {metrics}")
    
    logger.info("Training and evaluation completed!")


if __name__ == "__main__":
    main()