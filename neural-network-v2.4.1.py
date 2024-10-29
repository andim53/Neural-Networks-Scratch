#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:33:50 2024

@author: andisyamsul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Initialize parameters with Adam-specific parameters
def init_params(layers: List[int], activation_funcs: List[str]) -> Dict[str, np.ndarray]:
    params = {}
    for i in range(1, len(layers)):
        W = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1]).astype(np.float32)  # He initialization
        b = np.zeros((layers[i], 1), dtype=np.float32)
        params[f'W{i}'] = W
        params[f'b{i}'] = b
        params[f'activation{i}'] = activation_funcs[i - 1]
    
    # Initialize Adam parameters
    params['v_dW'] = {f'dW{i}': np.zeros_like(params[f'W{i}'], dtype=np.float32) for i in range(1, len(layers))}
    params['v_db'] = {f'db{i}': np.zeros_like(params[f'b{i}'], dtype=np.float32) for i in range(1, len(layers))}
    params['s_dW'] = {f'dW{i}': np.zeros_like(params[f'W{i}'], dtype=np.float32) for i in range(1, len(layers))}
    params['s_db'] = {f'db{i}': np.zeros_like(params[f'b{i}'], dtype=np.float32) for i in range(1, len(layers))}
    
    return params

# Activation functions and their derivatives
def ReLU(Z: np.ndarray) -> np.ndarray:
    return np.maximum(Z, 0)

def ReLU_deriv(Z: np.ndarray) -> np.ndarray:
    return Z > 0

def identity(Z: np.ndarray) -> np.ndarray:
    return Z

activation_functions = {
    'ReLu': (ReLU, ReLU_deriv),
    'identity': (identity, None)
}

# Forward propagation
def forward_prop(params: Dict[str, np.ndarray], X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    A = X
    Z_vals, A_vals = [], [A]
    
    for i in range(1, len(params) // 2 + 1):
        W, b = params[f'W{i}'], params[f'b{i}']
        Z = W.dot(A) + b
        Z_vals.append(Z)

        activation_func = params[f'activation{i}']
        A = activation_functions[activation_func][0](Z)
        A_vals.append(A)

    return Z_vals, A_vals

# Loss function: Mean Absolute Error
def compute_loss(A_last: np.ndarray, Y: np.ndarray, loss_type: str = 'MAE') -> float:
    if loss_type == 'MAE':
        return np.mean(np.abs(A_last - Y))
    elif loss_type == 'MSE':
        return np.mean((A_last - Y) ** 2)

# Backward propagation for MAE
def backward_prop(Z_vals: List[np.ndarray], A_vals: List[np.ndarray], params: Dict[str, np.ndarray], X: np.ndarray, Y: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    m = Y.size
    dW, db = {}, {}
    
    # Compute dZ for the output layer
    dZ = np.sign(A_vals[-1] - Y)  # Gradient for MAE
    
    for layer in reversed(range(1, len(params) // 2 + 1)):
        W, Z, A_prev = params[f'W{layer}'], Z_vals[layer - 1], A_vals[layer - 1]
        dW[layer] = (1 / m) * dZ.dot(A_prev.T)
        db[layer] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            dZ = W.T.dot(dZ) * activation_functions[params[f'activation{layer - 1}']][1](Z)

    return dW, db

# Adam optimizer update function
def update_params(params: Dict[str, np.ndarray], dW: Dict[int, np.ndarray], db: Dict[int, np.ndarray], alpha: float, beta1: float, beta2: float, epsilon: float, t: int) -> Dict[str, np.ndarray]:
    for layer in range(1, len(params) // 2 + 1):
        # Update biased first moment estimate
        params['v_dW'][f'dW{layer}'] = beta1 * params['v_dW'][f'dW{layer}'] + (1 - beta1) * dW[layer]
        params['v_db'][f'db{layer}'] = beta1 * params['v_db'][f'db{layer}'] + (1 - beta1) * db[layer]

        # Update biased second raw moment estimate
        params['s_dW'][f'dW{layer}'] = beta2 * params['s_dW'][f'dW{layer}'] + (1 - beta2) * (dW[layer] ** 2)
        params['s_db'][f'db{layer}'] = beta2 * params['s_db'][f'db{layer}'] + (1 - beta2) * (db[layer] ** 2)

        # Compute bias-corrected first and second moment estimates
        v_dW_corr = params['v_dW'][f'dW{layer}'] / (1 - beta1 ** t)
        v_db_corr = params['v_db'][f'db{layer}'] / (1 - beta1 ** t)
        s_dW_corr = params['s_dW'][f'dW{layer}'] / (1 - beta2 ** t)
        s_db_corr = params['s_db'][f'db{layer}'] / (1 - beta2 ** t)

        # Update parameters
        params[f'W{layer}'] -= (alpha * v_dW_corr / (np.sqrt(s_dW_corr) + epsilon))
        params[f'b{layer}'] -= (alpha * v_db_corr / (np.sqrt(s_db_corr) + epsilon))

    return params

# Training loop using Adam optimizer
def gradient_descent(X: np.ndarray, Y: np.ndarray, activation_funcs: List[str], layer_dims: List[int], alpha: float, iterations: int, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> Tuple[Dict[str, np.ndarray], List[float], List[float]]:
    params = init_params(layer_dims, activation_funcs)
    losses, mses = [], []

    for i in range(1, iterations + 1):
        Z_vals, A_vals = forward_prop(params, X)
        
        loss = compute_loss(A_vals[-1], Y, 'MAE')
        mse = compute_loss(A_vals[-1], Y, 'MSE')
        losses.append(loss)
        mses.append(mse)

        dW, db = backward_prop(Z_vals, A_vals, params, X, Y)
        params = update_params(params, dW, db, alpha, beta1, beta2, epsilon, i)

        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}, MSE = {mse:.4f}")

    return params, losses, mses

# Predictions and evaluation
def make_predictions(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    _, A_vals = forward_prop(params, X)
    return A_vals[-1]

def evaluate_model(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    return np.mean(np.abs(Y_pred - Y_true))

# Load and preprocess dataset
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv("dataset/auto-mpg.data", names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

# Clean dataset
dataset = raw_dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

# Ensure all numeric columns are of float type for calculations
numeric_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
dataset[numeric_cols] = dataset[numeric_cols].astype(float)

# Prepare features and target
X = dataset.drop('MPG', axis=1).values.T
Y = dataset['MPG'].values.reshape(1, -1)

# Normalize data
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# Neural network parameters
layers = [X.shape[0], 10, 5, 1]  # Input layer, hidden layers, output layer
activation_funcs = ['ReLu', 'ReLu', 'identity']
alpha = 0.001  # Learning rate
iterations = 500

# Train neural network
params, losses, mses = gradient_descent(X, Y, activation_funcs, layers, alpha, iterations)

# Make predictions and evaluate
Y_pred = make_predictions(X, params)
mae = evaluate_model(Y_pred.flatten(), Y.flatten())
print(f"Mean Absolute Error: {mae:.4f}")

# Plotting loss and MSE over iterations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss (MAE)')

plt.subplot(1, 2, 2)
plt.plot(mses)
plt.title('Mean Squared Error over Iterations')
plt.xlabel('Iterations')
plt.ylabel('MSE')

plt.tight_layout()
plt.show()
