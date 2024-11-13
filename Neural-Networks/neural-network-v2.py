#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:26:54 2024

@author: andisyamsul

regression model

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

########################### Initiation Params #################################

# Def params
def init_params(layers, activation_func):
    params = {}
    for i in range(1, len(layers)):

        # Initialize weights
        # if i == 1:  # For the input layer (W1)
        #     # W = np.random.rand(layers[i], layers[i - 1] - 1) - 0.5  # Subtract 1 from the input dimension
        #     W = np.random.rand(layers[i], layers[i - 1]) - 0.5  # Subtract 1 from the input dimension
        # else:  # For all other layers
        #     W = np.random.rand(layers[i], layers[i - 1]) - 0.5
        
        # W = np.random.rand(layers[i], layers[i - 1]) - 0.5
        # params[f'W{i}'] = W

        # # Initialize biases
        # b = np.random.rand(layers[i], 1) - 0.5 
        # params[f'b{i}'] = b
        
        # # Initialize activation func
        # params[f'activation{i}'] = activation_func[i - 1]
        
        # Initialize weights as float64
        W = np.random.rand(layers[i], layers[i - 1]).astype(np.float64) - 0.5
        params[f'W{i}'] = W
        
        # Initialize biases as float64
        b = np.random.rand(layers[i], 1).astype(np.float64) - 0.5 
        params[f'b{i}'] = b
        
        # Initialize activation function
        params[f'activation{i}'] = activation_func[i - 1]
        
    return params

# Activation function
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Derivative
def ReLU_deriv(Z):
    return Z > 0

################### Forward propagation (prediction function) #################
def forward_prop(params, X):
    
    # Initialize input
    A = X
    
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))  # Count weights to determine the number of layers
    
    # Initialize Z and A
    Z_vals = []
    A_vals = [A]
    
    for i in range(1, num_layers + 1):
        W = params[f'W{i}']
        b = params[f'b{i}']
        
        # Calculate Z
        Z = W.dot(A) + b
        Z_vals.append(Z)
        
        activation_func = params[f'activation{i}']
        
        # Apply activation funct
        if activation_func == 'ReLu':
            A = ReLU(Z)
        elif activation_func == 'softmax':
            A = softmax(Z)
        else:
            A = Z
        
        A_vals.append(A)
        
    return Z_vals, A_vals

def forward_prop(params, X):
    
    A = X
    Z_vals, A_vals = [], [A]
    
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))  # Count weights to determine the number of layers
    
    for i in range(1, num_layers + 1):
        W = params[f'W{i}']
        b = params[f'b{i}']
        Z = W.dot(A) + b
        Z_vals.append(Z)
        
        activation_func = params[f'activation{i}']
        
        # Apply activation funct
        if activation_func == 'ReLu':
            A = ReLU(Z)
        else:
            A = Z # Linear activation for regression
        
        A_vals.append(A)
        
    return Z_vals, A_vals

# Backward propagation
# def backward_prop(Z_vals, A_vals, params, X, Y):
#     m = Y.size  # Number of samples
#     Y = np.array(Y)
#     # one_hot_Y = one_hot(Y)  # Convert labels to one-hot encoding for Classification problem

#     # Initialize dictionaries to store gradients for weights and biases
#     dW = {}
#     db = {}
    
#     # Number of layers in the network
#     num_layers = sum(1 for key in params.keys() if key.startswith('W'))

#     # Compute gradient for the output layer
#     A_last = A_vals[-1]
#     # dZ = A_last - one_hot_Y  # Gradient of the loss w.r.t. the output layer pre-activation

#     dZ = A_last - Y.reshape(-1, 1)
    
#     # Backpropagation for the last layer
#     dW[num_layers] = (1 / m) * dZ.dot(A_vals[-2].T)  # Gradient for weights of the last layer
#     db[num_layers] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # Gradient for biases of the last layer

#     # Backpropagate through each hidden layer
#     for layer in reversed(range(1, num_layers)):
#         W_next = params[f'W{layer + 1}']
#         Z = Z_vals[layer - 1]  # Pre-activation value for the current layer
#         A_prev = A_vals[layer - 1]  # Activation from the previous layer
        
#         # Compute dZ for the current layer
#         dZ = W_next.T.dot(dZ) * ReLU_deriv(Z)  # Element-wise multiplication with ReLU derivative
        
#         # Calculate gradients for the current layer's weights and biases
#         dW[layer] = (1 / m) * dZ.dot(A_prev.T)
#         db[layer] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

#     return dW, db

# def backward_prop(Z_vals, A_vals, params, X, Y):
#     m = Y.size  # Number of samples

#     # Initialize dictionaries to store gradients for weights and biases
#     dW = {}
#     db = {}
    
#     # Number of layers in the network
#     num_layers = sum(1 for key in params.keys() if key.startswith('W'))

#     # Compute gradient for the output layer
#     A_last = A_vals[-1]  # Predictions from the last layer
#     dZ = A_last - np.array(Y).reshape(1, -1)  # Gradient of the loss w.r.t. the output layer pre-activation
    
#     # Backpropagation for the last layer
#     dW[num_layers] = (1 / m) * dZ.dot(A_vals[-2].T)  # Gradient for weights of the last layer
#     db[num_layers] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # Gradient for biases of the last layer

#     # Backpropagate through each hidden layer
#     for layer in reversed(range(1, num_layers)):
#         W_next = params[f'W{layer + 1}']
#         Z = Z_vals[layer - 1]  # Pre-activation value for the current layer
#         A_prev = A_vals[layer - 1]  # Activation from the previous layer
        
#         # Compute dZ for the current layer
#         dZ = W_next.T.dot(dZ) * ReLU_deriv(Z)  # Element-wise multiplication with ReLU derivative
        
#         # Calculate gradients for the current layer's weights and biases
#         dW[layer] = (1 / m) * dZ.dot(A_prev.T)
#         db[layer] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

#     return dW, db

def backward_prop(Z_vals, A_vals, params, X, Y):
    m = Y.size  # Number of samples

    # Initialize dictionaries to store gradients for weights and biases
    dW = {}
    db = {}
    
    # Number of layers in the network
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))

    # Compute gradient for the output layer
    A_last = A_vals[-1]  # Predictions from the last layer
    dZ = A_last - np.array(Y).reshape(1, -1)  # Gradient of the loss w.r.t. the output layer pre-activation
    
    # Backpropagation for the last layer
    dW[num_layers] = ((1 / m) * dZ.dot(A_vals[-2].T)).astype(np.float64)  # Gradient for weights of the last layer
    db[num_layers] = ((1 / m) * np.sum(dZ, axis=1, keepdims=True)).astype(np.float64)  # Gradient for biases of the last layer

    # Backpropagate through each hidden layer
    for layer in reversed(range(1, num_layers)):
        W_next = params[f'W{layer + 1}']
        Z = Z_vals[layer - 1]  # Pre-activation value for the current layer
        A_prev = A_vals[layer - 1]  # Activation from the previous layer
        
        # Compute dZ for the current layer
        dZ = W_next.T.dot(dZ) * ReLU_deriv(Z)  # Element-wise multiplication with ReLU derivative
        
        # Calculate gradients for the current layer's weights and biases
        dW[layer] = ((1 / m) * dZ.dot(A_prev.T)).astype(np.float64)
        db[layer] = ((1 / m) * np.sum(dZ, axis=1, keepdims=True)).astype(np.float64)

    return dW, db




def update_params(params, dW, db, alpha):
    # Number of layers in the network
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))
    
    # Update each layer's parameters
    for layer in range(1, num_layers + 1):
        params[f'W{layer}'] -= alpha * dW[layer]  # Update weights
        params[f'b{layer}'] -= alpha * db[layer]  # Update biases
    
    return params

########################### Gradient Function ###########################

def get_predictions(A_last):
    return np.argmax(A_last, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, activation_func, layer_dims, alpha, iterations):
    # Initialize parameters dynamically based on layer dimensions
    params = init_params(layer_dims, activation_func)
    
    for i in range(iterations):
        # Forward propagation
        Z_vals, A_vals = forward_prop(params, X)
        
        # Backward propagation
        dW, db = backward_prop(Z_vals, A_vals, params, X, Y)
        
        # Update parameters
        params = update_params(params, dW, db, alpha)
        
        if i % 10 == 0:
            predictions = get_predictions(A_vals[-1])
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy = {accuracy}")
            
    return params

########################### Additinoal Function ###########################

def make_predictions(X, params):
    _, A_vals = forward_prop(params, X)
    predictions = get_predictions(A_vals[-1])
    return predictions

def test_prediction(index, X, Y, params):
    current_image = X[:, index, None]
    prediction = make_predictions(current_image, params)
    label = Y[index]
    print("Prediction:", prediction)
    print("Label:", label)

########################### Run Model ###########################

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

# Split the data into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
train_dataset.describe().transpose()

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Transpose, for calculating neural network
train_features = train_features.T
test_features = test_features.T

# Normalization
# train_dataset.describe().transpose()[['mean', 'std']]

input_layer = train_features.shape[0] # Number of faeture after transpose

layer_dims = [input_layer, 100, 100, 1] 
activation_func = ['ReLu','ReLu','no activation']
params = gradient_descent(train_features, train_labels, activation_func, layer_dims, alpha=0.10, iterations=10)


###################################################################

# params = init_params(layer_dims, activation_func)
# Z_vals, A_vals = forward_prop(params, train_features)

# # Backward propagation
# dW, db = backward_prop(Z_vals, A_vals, params, train_features, train_labels)

# num_layers = sum(1 for key in params.keys() if key.startswith('W'))
# layer = range(1, num_layers + 1)[0]

# print(f"Layer {layer}: params['W{layer}'] dtype = {params[f'W{layer}'].dtype}, dW[{layer}] dtype = {dW[layer].dtype}")
# print(f"Layer {layer}: params['b{layer}'] dtype = {params[f'b{layer}'].dtype}, db[{layer}] dtype = {db[layer].dtype}")

# alpha = 0.10
# params[f'W{layer}'] -= alpha * dW[layer] 

###################################################################

# layer_dims = [n, 100, 100, 10] # First layer has 784 neuron, second layer has 10 neuron, third layer has 10 neuron. First layer is the input layer, last layer is the output layer.
# # gives n-1, so that it could match the size of the input feature

# activation_func = ['ReLu','ReLu','softmax']

# # Train the network using gradient descent
# params = gradient_descent(X_train, Y_train, activation_func, layer_dims, alpha=0.10, iterations=500)

# # Test individual predictions with the trained parameters
# for i in range(4):  # Testing the first four samples
#     test_prediction(i, X_train, Y_train, params)

# # Evaluate the model on the development set
# dev_predictions = make_predictions(X_dev, params)
# dev_accuracy = get_accuracy(dev_predictions, Y_dev)
# print("Development Set Accuracy:", dev_accuracy)







