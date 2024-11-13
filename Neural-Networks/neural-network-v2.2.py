#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:31:41 2024

@author: andisyamsul

Dynamics Model - Regression
"""

# Regression ANN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define parameters
def init_params(layers, activation_func):
    params = {}
    for i in range(1, len(layers)):
        # Initialize weights and biases
        W = np.random.rand(layers[i], layers[i - 1]) - 0.5
        b = np.random.rand(layers[i], 1) - 0.5 
        params[f'W{i}'] = W
        params[f'b{i}'] = b
        params[f'activation{i}'] = activation_func[i - 1]
    return params

# Activation functions and their derivatives
def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def identity(Z):
    return Z

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

# Loss function and backward propagation for Mean Squared Error
def compute_loss(A_last, Y):
    m = Y.size
    loss = np.sum((A_last - Y)**2) / (2 * m)
    return loss

def backward_prop(Z_vals, A_vals, params, X, Y):
    m = Y.size
    dW = {}
    db = {}
    
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))
    
    # Compute dZ for the output layer
    A_last = A_vals[-1]
    dZ = (A_last - Y)  # Regression loss gradient
    
    for layer in reversed(range(1, num_layers + 1)):
        W = params[f'W{layer}']
        Z = Z_vals[layer - 1]
        A_prev = A_vals[layer - 1]
        
        dW[layer] = (1 / m) * dZ.dot(A_prev.T).astype(np.float64) 
        db[layer] = (1 / m) * np.sum(dZ, axis=1, keepdims=True).astype(np.float64) 
        
        if layer > 1:
            dZ = W.T.dot(dZ) * ReLU_deriv(Z).astype(np.float64) 
            
    return dW, db, dZ

def update_params(params, dW, db, alpha):
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))
    for layer in range(1, num_layers + 1):
        params[f'W{layer}'] -= alpha * dW[layer]
        params[f'b{layer}'] -= alpha * db[layer]
    return params

# Training loop
def gradient_descent(X, Y, activation_func, layer_dims, alpha, iterations):
    params = init_params(layer_dims, activation_func)
    
    for i in range(iterations):
        Z_vals, A_vals = forward_prop(params, X)
        loss = compute_loss(A_vals[-1], Y)
        
        dW, db = backward_prop(Z_vals, A_vals, params, X, Y)
        params = update_params(params, dW, db, alpha)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {loss}")
            
    return params

# Predictions and evaluation
def make_predictions(X, params):
    _, A_vals = forward_prop(params, X)
    return A_vals[-1]

def evaluate_model(Y_pred, Y_true):
    mse = np.mean((Y_pred - Y_true) ** 2)
    return mse


###############################################################################

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv("dataset/auto-mpg.data", names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

# Drop rows with missing values
dataset = dataset.dropna()

# Convert 'Origin' to one-hot encoding
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
data = np.array(dataset)
m, n = dataset.shape

data_train = data[100:m].T
Y_train = data_train[0].reshape(1, -1)  # Target for regression
X_train = data_train[1:n] # Normalize features

data_dev = data[0:100].T
Y_dev = data_dev[0].reshape(1, -1)
X_dev = data_dev[1:n]

# Normalization
X_train = X_train.astype(np.float64)
X_dev = X_dev.astype(np.float64)

# Replace NaNs with column means or another appropriate value
X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
X_dev = np.nan_to_num(X_dev, nan=np.nanmean(X_dev))

X_train = (X_train - X_train.mean(axis=1, keepdims=True)) / X_train.std(axis=1, keepdims=True)
X_dev = (X_dev - X_dev.mean(axis=1, keepdims=True)) / X_dev.std(axis=1, keepdims=True)

# Define architecture
layer_dims = [n-1, 100, 100, 1]  # Output layer has 1 neuron for regression
activation_func = ['ReLu', 'ReLu', 'identity']

# Train the model
params = gradient_descent(X_train, 
                          Y_train, 
                          activation_func, 
                          layer_dims, 
                          alpha=0.10, 
                          iterations=100)


# Split the data into training and test sets
# train_dataset = dataset.sample(frac=0.8, random_state=0)
# test_dataset = dataset.drop(train_dataset.index)

# # Split features from labels
# train_features = train_dataset.copy()
# test_features = test_dataset.copy()

# train_labels = train_features.pop('MPG')
# test_labels = test_features.pop('MPG')

# # Transpose, for calculating neural network
# train_features = np.array(train_features.T)
# test_features = np.array(test_features.T)
# train_labels = np.array(train_labels)
# test_labels = np.array(test_labels)


###############################################################################

# # Load dataset and preprocess for regression
# data = pd.read_csv("./dataset/train.csv")
# data = np.array(data)
# m, n = data.shape

# data_train = data[1000:m].T
# Y_train = data_train[0].reshape(1, -1)  # Target for regression
# X_train = data_train[1:n] / 255.  # Normalize features

# data_dev = data[0:1000].T
# Y_dev = data_dev[0].reshape(1, -1)
# X_dev = data_dev[1:n] / 255.

# # Define architecture
# layer_dims = [n-1, 100, 100, 1]  # Output layer has 1 neuron for regression
# activation_func = ['ReLu', 'ReLu', 'identity']

# # Train the model
# params = gradient_descent(X_train, Y_train, activation_func, layer_dims, alpha=0.10, iterations=100)

# # Evaluate on development set
# Y_dev_pred = make_predictions(X_dev, params)
# dev_mse = evaluate_model(Y_dev_pred, Y_dev)
# print("Development Set MSE:", dev_mse)
