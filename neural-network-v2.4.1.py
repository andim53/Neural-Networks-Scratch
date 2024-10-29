#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Neural Network Implementation for Auto-MPG Dataset
Features:
- Early stopping
- Learning rate scheduling
- Improved initialization
- Better organization with classes
- Type hints and documentation
- Configuration management
- Improved visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional
from pathlib import Path

@dataclass
class NetworkConfig:
    """Configuration for neural network hyperparameters."""
    layer_dims: List[int]
    activation_funcs: List[str]
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_iterations: int = 1000
    batch_size: int = 32
    early_stopping_patience: int = 50
    min_delta: float = 1e-4

class ActivationFunctions:
    """Collection of activation functions and their derivatives."""
    
    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(Z, 0)
    
    @staticmethod
    def relu_deriv(Z: np.ndarray) -> np.ndarray:
        return Z > 0
    
    @staticmethod
    def identity(Z: np.ndarray) -> np.ndarray:
        return Z
    
    @staticmethod
    def identity_deriv(Z: np.ndarray) -> np.ndarray:
        return np.ones_like(Z)

class NeuralNetwork:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.params: Dict = {}
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
        self._initialize_parameters()
        
    def _initialize_parameters(self) -> None:
        """Initialize network parameters using He initialization."""
        for i in range(1, len(self.config.layer_dims)):
            # He initialization with proper scaling
            scale = np.sqrt(2.0 / self.config.layer_dims[i - 1])
            self.params[f'W{i}'] = np.random.randn(self.config.layer_dims[i], 
                                                  self.config.layer_dims[i - 1]).astype(np.float32) * scale
            self.params[f'b{i}'] = np.zeros((self.config.layer_dims[i], 1), dtype=np.float32)
            self.params[f'activation{i}'] = self.config.activation_funcs[i - 1]
        
        # Initialize Adam optimizer parameters
        for i in range(1, len(self.config.layer_dims)):
            for param in ['v_dW', 'v_db', 's_dW', 's_db']:
                if param not in self.params:
                    self.params[param] = {}
                self.params[param][f'd{"W" if "W" in param else "b"}{i}'] = \
                    np.zeros_like(self.params[f'{"W" if "W" in param else "b"}{i}'], dtype=np.float32)

    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Perform forward propagation through the network."""
        A = X
        Z_vals = []
        A_vals = [A]
        
        for i in range(1, len(self.config.layer_dims)):
            Z = self.params[f'W{i}'].dot(A) + self.params[f'b{i}']
            Z_vals.append(Z)
            
            A = (ActivationFunctions.relu(Z) if self.params[f'activation{i}'] == 'ReLu' 
                 else ActivationFunctions.identity(Z))
            A_vals.append(A)
            
        return Z_vals, A_vals

    def backward_propagation(self, Z_vals: List[np.ndarray], A_vals: List[np.ndarray], 
                           X: np.ndarray, Y: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Perform backward propagation to compute gradients."""
        m = Y.shape[1]
        dW = {}
        db = {}
        
        # Output layer gradient
        dZ = np.sign(A_vals[-1] - Y)  # MAE gradient
        
        for layer in reversed(range(1, len(self.config.layer_dims))):
            dW[layer] = (1 / m) * dZ.dot(A_vals[layer - 1].T)
            db[layer] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            
            if layer > 1:
                activation_deriv = (ActivationFunctions.relu_deriv if self.params[f'activation{layer-1}'] == 'ReLu' 
                                  else ActivationFunctions.identity_deriv)
                dZ = self.params[f'W{layer}'].T.dot(dZ) * activation_deriv(Z_vals[layer - 2])
                
        return dW, db

    def update_parameters(self, dW: Dict[int, np.ndarray], db: Dict[int, np.ndarray], t: int) -> None:
        """Update parameters using Adam optimizer."""
        lr = self._get_learning_rate(t)
        
        for layer in range(1, len(self.config.layer_dims)):
            # Update momentum and RMSprop terms
            for param_name, grad, param_type in [('W', dW[layer], 'dW'), ('b', db[layer], 'db')]:
                # Momentum
                self.params[f'v_{param_type}'][f'{param_type}{layer}'] = \
                    (self.config.beta1 * self.params[f'v_{param_type}'][f'{param_type}{layer}'] + 
                     (1 - self.config.beta1) * grad).astype(np.float32)
                
                # RMSprop
                self.params[f's_{param_type}'][f'{param_type}{layer}'] = \
                    (self.config.beta2 * self.params[f's_{param_type}'][f'{param_type}{layer}'] + 
                     (1 - self.config.beta2) * (grad ** 2)).astype(np.float32)
                
                # Bias correction
                v_corrected = self.params[f'v_{param_type}'][f'{param_type}{layer}'] / (1 - self.config.beta1 ** t)
                s_corrected = self.params[f's_{param_type}'][f'{param_type}{layer}'] / (1 - self.config.beta2 ** t)
                
                # Update parameters
                self.params[f'{param_name}{layer}'] -= \
                    (lr * v_corrected / (np.sqrt(s_corrected) + self.config.epsilon)).astype(np.float32)

    def _get_learning_rate(self, t: int) -> float:
        """Implement learning rate scheduling."""
        return self.config.learning_rate * (1.0 / (1.0 + 0.01 * t))

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
             X_val: np.ndarray, Y_val: np.ndarray) -> None:
        """Train the neural network with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for i in range(1, self.config.max_iterations + 1):
            # Forward and backward passes
            Z_vals, A_vals = self.forward_propagation(X_train)
            dW, db = self.backward_propagation(Z_vals, A_vals, X_train, Y_train)
            
            # Update parameters
            self.update_parameters(dW, db, i)
            
            # Calculate metrics
            train_loss = self._compute_loss(A_vals[-1], Y_train)
            train_mse = self._compute_mse(A_vals[-1], Y_train)
            
            # Validation metrics
            _, val_A_vals = self.forward_propagation(X_val)
            val_loss = self._compute_loss(val_A_vals[-1], Y_val)
            val_mse = self._compute_mse(val_A_vals[-1], Y_val)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mse'].append(train_mse)
            self.history['val_mse'].append(val_mse)
            
            # Early stopping check
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered at iteration {i}")
                break
            
            if i % 10 == 0:
                print(f"Iteration {i}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained network."""
        _, A_vals = self.forward_propagation(X)
        return A_vals[-1]

    @staticmethod
    def _compute_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute MAE loss."""
        return float(np.mean(np.abs(predictions - targets)))

    @staticmethod
    def _compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute MSE loss."""
        return float(np.mean((predictions - targets) ** 2))

    def plot_training_history(self) -> None:
        """Plot training metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Loss Over Iterations')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss (MAE)')
        ax1.grid(True)
        ax1.legend()
        
        # MSE plot
        ax2.plot(self.history['train_mse'], label='Train MSE', color='blue')
        ax2.plot(self.history['val_mse'], label='Validation MSE', color='red')
        ax2.set_title('Mean Squared Error Over Iterations')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('MSE')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def load_and_preprocess_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the auto-mpg dataset."""
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                   'Acceleration', 'Model Year', 'Origin']
    
    # Load data
    dataset = pd.read_csv(filepath, names=column_names, na_values='?', 
                         comment='\t', sep=' ', skipinitialspace=True)
    
    # Clean and preprocess
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
    
    # Split data
    data = dataset.values
    train_size = int(0.8 * len(data))
    
    # Prepare training and validation sets
    train_data = data[train_size:].T
    val_data = data[:train_size].T
    
    Y_train = train_data[0:1]
    X_train = train_data[1:]
    Y_val = val_data[0:1]
    X_val = val_data[1:]
    
    # Ensure arrays are float64
    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    Y_train = Y_train.astype(np.float64)
    Y_val = Y_val.astype(np.float64)
    
    # Normalize features with handling for zero standard deviation
    means = np.mean(X_train, axis=1, keepdims=True)
    stds = np.std(X_train, axis=1, keepdims=True)
    # Replace zero standard deviations with 1 to avoid division by zero
    stds[stds == 0] = 1.0
    
    X_train = (X_train - means) / stds
    X_val = (X_val - means) / stds
    
    return X_train, Y_train, X_val, Y_val

def main():
    # Load and preprocess data
    X_train, Y_train, X_val, Y_val = load_and_preprocess_data("dataset/auto-mpg.data")
    
    # Configure network
    config = NetworkConfig(
        layer_dims=[X_train.shape[0], 128, 128, 1],
        activation_funcs=['ReLu', 'ReLu', 'identity'],
        learning_rate=0.001,
        max_iterations=1000,
        early_stopping_patience=50
    )
    
    # Create and train network
    model = NeuralNetwork(config)
    model.train(X_train, Y_train, X_val, Y_val)
    
    # Evaluate model
    train_predictions = model.predict(X_train)
    train_mae = model._compute_loss(train_predictions, Y_train)
    print(f"Final Train MAE: {train_mae:.4f}")
    
    val_predictions = model.predict(X_val)
    val_mae = model._compute_loss(val_predictions, Y_val)
    print(f"Final Validation MAE: {val_mae:.4f}")
    
    # Plot training history
    model.plot_training_history()

if __name__ == "__main__":
    main()