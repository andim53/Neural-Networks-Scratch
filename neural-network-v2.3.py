#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:33:50 2024

@author: andisyamsul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Initialize parameters with Adam-specific parameters
# def init_params(layers, activation_func):
#     params = {}
#     for i in range(1, len(layers)):
#         # Initialize weights and biases
#         W = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])  # He initialization
#         b = np.zeros((layers[i], 1))
#         params[f'W{i}'] = W
#         params[f'b{i}'] = b
#         params[f'activation{i}'] = activation_func[i - 1]
    
#     # Initialize Adam parameters
#     params['v_dW'] = {f'dW{i}': np.zeros_like(params[f'W{i}']) for i in range(1, len(layers))}
#     params['v_db'] = {f'db{i}': np.zeros_like(params[f'b{i}']) for i in range(1, len(layers))}
#     params['s_dW'] = {f'dW{i}': np.zeros_like(params[f'W{i}']) for i in range(1, len(layers))}
#     params['s_db'] = {f'db{i}': np.zeros_like(params[f'b{i}']) for i in range(1, len(layers))}
    
#     return params

# Initialize parameters with Adam-specific parameters
def init_params(layers, activation_func):
    params = {}
    for i in range(1, len(layers)):
        # Initialize weights and biases with float dtype
        W = (np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])).astype(np.float32)  # He initialization
        b = np.zeros((layers[i], 1), dtype=np.float32)
        params[f'W{i}'] = W
        params[f'b{i}'] = b
        params[f'activation{i}'] = activation_func[i - 1]
    
    # Initialize Adam parameters with float dtype
    params['v_dW'] = {f'dW{i}': np.zeros_like(params[f'W{i}'], dtype=np.float32) for i in range(1, len(layers))}
    params['v_db'] = {f'db{i}': np.zeros_like(params[f'b{i}'], dtype=np.float32) for i in range(1, len(layers))}
    params['s_dW'] = {f'dW{i}': np.zeros_like(params[f'W{i}'], dtype=np.float32) for i in range(1, len(layers))}
    params['s_db'] = {f'db{i}': np.zeros_like(params[f'b{i}'], dtype=np.float32) for i in range(1, len(layers))}
    
    return params


# Activation functions and their derivatives
def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def identity(Z):
    return Z

# Forward propagation
def forward_prop(params, X):
    A = X
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))
    
    Z_vals = []
    A_vals = [A]
    
    for i in range(1, num_layers + 1):
        W = params[f'W{i}']
        b = params[f'b{i}']
        
        Z = W.dot(A) + b
        Z_vals.append(Z)
        
        activation_func = params[f'activation{i}']
        if activation_func == 'ReLu':
            A = ReLU(Z)
        elif activation_func == 'identity':
            A = identity(Z)
        
        A_vals.append(A)
    return Z_vals, A_vals

# # Mean Absolute Error (MAE) Loss function
# def compute_loss(A_last, Y):
#     m = Y.size
#     loss = np.sum(np.abs(A_last - Y)) / m
#     return loss

# Loss function: Mean Absolute Error
def compute_loss(A_last, Y):
    loss = np.mean(np.abs(A_last - Y))  # MAE
    return loss

def compute_mse(A_last, Y):
    mse = np.mean((A_last - Y) ** 2)
    return mse

# Backward propagation for MAE
def backward_prop(Z_vals, A_vals, params, X, Y):
    m = Y.size
    dW = {}
    db = {}
    
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))
    
    # Compute dZ for the output layer
    A_last = A_vals[-1]
    dZ = np.sign(A_last - Y)  # Gradient for MAE
    
    for layer in reversed(range(1, num_layers + 1)):
        W = params[f'W{layer}']
        Z = Z_vals[layer - 1]
        A_prev = A_vals[layer - 1]
        
        dW[layer] = (1 / m) * dZ.dot(A_prev.T)
        db[layer] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        if layer > 1:
            dZ = W.T.dot(dZ) * ReLU_deriv(Z)
            
    return dW, db

# Adam optimizer update function
def update_params(params, dW, db, alpha, beta1, beta2, epsilon, t):
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))
    
    for layer in range(1, num_layers + 1):
        # Update biased first moment estimate
        params['v_dW'][f'dW{layer}'] = (beta1 * params['v_dW'][f'dW{layer}'] + (1 - beta1) * dW[layer]).astype(np.float32)
        params['v_db'][f'db{layer}'] = (beta1 * params['v_db'][f'db{layer}'] + (1 - beta1) * db[layer]).astype(np.float32)
        
        # Update biased second raw moment estimate
        params['s_dW'][f'dW{layer}'] = (beta2 * params['s_dW'][f'dW{layer}'] + (1 - beta2) * (dW[layer] ** 2)).astype(np.float32)
        params['s_db'][f'db{layer}'] = (beta2 * params['s_db'][f'db{layer}'] + (1 - beta2) * (db[layer] ** 2)).astype(np.float32)
        
        # Compute bias-corrected first and second moment estimates
        v_dW_corr = (params['v_dW'][f'dW{layer}'] / (1 - beta1 ** t)).astype(np.float32)
        v_db_corr = (params['v_db'][f'db{layer}'] / (1 - beta1 ** t)).astype(np.float32)
        s_dW_corr = (params['s_dW'][f'dW{layer}'] / (1 - beta2 ** t)).astype(np.float32)
        s_db_corr = (params['s_db'][f'db{layer}'] / (1 - beta2 ** t)).astype(np.float32)
        
        # Update parameters
        params[f'W{layer}'] -= (alpha * v_dW_corr / (np.sqrt(s_dW_corr) + epsilon)).astype(np.float32)
        params[f'b{layer}'] -= (alpha * v_db_corr / (np.sqrt(s_db_corr) + epsilon)).astype(np.float32)
    
    return params


# Training loop using Adam optimizer
# def gradient_descent(X, Y, activation_func, layer_dims, alpha, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
#     params = init_params(layer_dims, activation_func)
#     for i in range(1, iterations + 1):
#         Z_vals, A_vals = forward_prop(params, X)
#         loss = compute_loss(A_vals[-1], Y)
        
#         dW, db = backward_prop(Z_vals, A_vals, params, X, Y)
#         params = update_params(params, dW, db, alpha, beta1, beta2, epsilon, i)
        
#         if i % 10 == 0:
#             print(f"Iteration {i}: Loss = {loss}")
            
#     return params

def gradient_descent(X, Y, activation_func, layer_dims, alpha, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
    params = init_params(layer_dims, activation_func)
    
    losses = []
    mses = []
    
    for i in range(1, iterations + 1):
        Z_vals, A_vals = forward_prop(params, X)

        loss = compute_loss(A_vals[-1], Y)
        mse = compute_mse(A_vals[-1], Y)
        
        losses.append(loss)
        mses.append(mse)
        
        dW, db = backward_prop(Z_vals, A_vals, params, X, Y)
        params = update_params(params, dW, db, alpha, beta1, beta2, epsilon, i)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {loss}")
            
    return params, losses, mses, r2_scores

# Predictions and evaluation
def make_predictions(X, params):
    _, A_vals = forward_prop(params, X)
    return A_vals[-1]

def evaluate_model(Y_pred, Y_true):
    mae = np.mean(np.abs(Y_pred - Y_true))
    return mae

###############################################################################

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv("dataset/auto-mpg.data", names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

data = np.array(dataset)
m, n = dataset.shape

data_train = data[100:m].T
Y_train = data_train[0].reshape(1, -1)
X_train = data_train[1:n]

data_dev = data[0:100].T
Y_dev = data_dev[0].reshape(1, -1)
X_dev = data_dev[1:n]

# Normalize features
X_train = X_train.astype(np.float64)
X_dev = X_dev.astype(np.float64)
X_train = (X_train - X_train.mean(axis=1, keepdims=True)) / X_train.std(axis=1, keepdims=True)
X_dev = (X_dev - X_dev.mean(axis=1, keepdims=True)) / X_dev.std(axis=1, keepdims=True)

# Define architecture
layer_dims = [n - 1, 100, 100, 1]
activation_func = ['ReLu', 'ReLu', 'identity']

# Train the model
params, losses, mses, r2_scores = gradient_descent(X_train, Y_train, activation_func, layer_dims, alpha=0.001, iterations=100)

# # Predictions
# train_predictions = make_predictions(X_train, params)
# train_mae = evaluate_model(train_predictions, Y_train)
# print("Train MAE:", train_mae)

# dev_predictions = make_predictions(X_dev, params)
# dev_mae = evaluate_model(dev_predictions, Y_dev)
# print("Development Set MAE:", dev_mae)

# Make predictions
Y_pred_train = make_predictions(X_train, params)
Y_pred_dev = make_predictions(X_dev, params)

# Plotting metrics
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss', color='blue')
plt.title('Loss Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid()

# MSE plot
plt.subplot(1, 2, 2)
plt.plot(mses, label='MSE', color='green')
plt.title('Mean Squared Error Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.grid()

plt.tight_layout()
plt.legend()
plt.show()
