"""
Quantum Model Training
Specialized trainer for Variational Quantum Circuits (VQC)
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, Callable
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import json


class QuantumTrainer:
    """
    Trainer for Quantum Neural Networks
    Handles quantum-specific training considerations
    """
    
    def __init__(
        self,
        forward_fn: Callable,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable = None,
        checkpoint_dir: str = "experiments/checkpoints/quantum",
        seed: int = 42
    ):
        """
        Args:
            forward_fn: Forward function (params, features) -> prediction
            optimizer: Optax optimizer
            loss_fn: Loss function (default: MSE)
            checkpoint_dir: Directory for checkpoints
            seed: Random seed
        """
        self.forward_fn = forward_fn
        self.optimizer = optimizer
        self.loss_fn = loss_fn or self._mse_loss
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.rng = jax.random.PRNGKey(seed)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'epoch_time': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        print(f"✅ Quantum Trainer initialized")
        print(f"   Checkpoint dir: {checkpoint_dir}")
    
    @staticmethod
    def _mse_loss(pred, target):
        """Mean Squared Error"""
        return jnp.mean((pred - target) ** 2)
    
    @staticmethod
    def _mae_metric(pred, target):
        """Mean Absolute Error"""
        return jnp.mean(jnp.abs(pred - target))
    
    def _compute_loss(self, params, features, target):
        """Compute loss for a single sample"""
        prediction = self.forward_fn(params, features)
        return self.loss_fn(prediction, target)
    
    def _train_step(self, params, opt_state, features, target):
        """
        Single training step
        
        Returns:
            Updated params, opt_state, loss value
        """
        loss, grads = jax.value_and_grad(self._compute_loss)(params, features, target)
        
        # Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    def train_epoch(
        self,
        params,
        opt_state,
        train_features: jnp.ndarray,
        train_targets: jnp.ndarray,
        batch_size: int = 1
    ) -> Tuple:
        """
        Train for one epoch
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            train_features: [n_samples, n_qubits] quantum-encoded features
            train_targets: [n_samples] target values
            batch_size: Batch size (1 for quantum, usually)
            
        Returns:
            Updated params, opt_state, metrics
        """
        n_samples = len(train_features)
        indices = np.random.permutation(n_samples)
        
        epoch_losses = []
        all_predictions = []
        all_targets = []
        
        # Train on each sample
        for idx in indices:
            features = train_features[idx]
            target = train_targets[idx]
            
            params, opt_state, loss = self._train_step(params, opt_state, features, target)
            
            epoch_losses.append(float(loss))
            
            # Get prediction for metrics
            pred = self.forward_fn(params, features)
            all_predictions.append(float(pred))
            all_targets.append(float(target))
        
        # Calculate epoch metrics
        all_predictions = jnp.array(all_predictions)
        all_targets = jnp.array(all_targets)
        
        metrics = {
            'loss': np.mean(epoch_losses),
            'mae': float(self._mae_metric(all_predictions, all_targets))
        }
        
        return params, opt_state, metrics
    
    def evaluate(
        self,
        params,
        eval_features: jnp.ndarray,
        eval_targets: jnp.ndarray
    ) -> Dict:
        """
        Evaluate on validation/test set
        
        Returns:
            Dictionary of metrics
        """
        predictions = []
        losses = []
        
        for i in range(len(eval_features)):
            features = eval_features[i]
            target = eval_targets[i]
            
            pred = self.forward_fn(params, features)
            loss = self.loss_fn(pred, target)
            
            predictions.append(float(pred))
            losses.append(float(loss))
        
        predictions = jnp.array(predictions)
        targets = jnp.array(eval_targets)
        
        metrics = {
            'loss': np.mean(losses),
            'mae': float(self._mae_metric(predictions, targets)),
            'rmse': float(jnp.sqrt(jnp.mean((predictions - targets) ** 2))),
            'r2': float(1 - jnp.sum((targets - predictions) ** 2) / 
                       (jnp.sum((targets - jnp.mean(targets)) ** 2) + 1e-8))
        }
        
        return metrics
    
    def save_checkpoint(self, params, epoch, metrics, name="checkpoint"):
        path = self.checkpoint_dir / f"{name}.npz"
        flat, _ = jax.tree_util.tree_flatten(params)
        np.savez(
            path,
            **{f"param_{i}": np.array(p) for i, p in enumerate(flat)},
            epoch=epoch,
            metrics=json.dumps(metrics),
        )
    
    def fit(
        self,
        params,
        train_features: jnp.ndarray,
        train_targets: jnp.ndarray,
        val_features: jnp.ndarray,
        val_targets: jnp.ndarray,
        num_epochs: int = 50,
        patience: int = 10,
        verbose: bool = True
    ):
        """
        Full training loop
        
        Args:
            params: Initial parameters
            train_features: Training quantum features
            train_targets: Training targets
            val_features: Validation quantum features
            val_targets: Validation targets
            num_epochs: Maximum epochs
            patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Best params, optimizer state, history
        """
        opt_state = self.optimizer.init(params)
        
        best_params = params
        epochs_without_improvement = 0
        
        if verbose:
            print(f"\n🏋️  Training Quantum Model")
            print(f"   Train samples: {len(train_features)}")
            print(f"   Val samples: {len(val_features)}")
            print(f"   Max epochs: {num_epochs}")
            print(f"   Patience: {patience}\n")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            params, opt_state, train_metrics = self.train_epoch(
                params, opt_state, train_features, train_targets
            )
            
            val_metrics = self.evaluate(params, val_features, val_targets)
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['epoch_time'].append(epoch_time)
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                best_params = params
                epochs_without_improvement = 0
                self.save_checkpoint(params, epoch, val_metrics, name='best')
            else:
                epochs_without_improvement += 1
            
            if verbose and (epoch % 5 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.4f} | "
                    f"Time: {epoch_time:.1f}s")
            
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                    print(f"   Best val loss: {self.best_val_loss:.4f} "
                        f"at epoch {self.best_epoch+1}")
                break
        
        self.save_checkpoint(params, num_epochs, val_metrics, name='final')
        
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        if verbose:
            print(f"\n✅ Training complete!")
            print(f"   Best epoch: {self.best_epoch + 1}")
            print(f"   Best val MAE: {self.history['val_mae'][self.best_epoch]:.4f}")
        
        return best_params, opt_state, self.history