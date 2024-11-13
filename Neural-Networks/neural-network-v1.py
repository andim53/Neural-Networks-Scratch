# Dynamics ANN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Created on Tue Oct 29 12:26:54 2024

@author: andisyamsul

Dynamics model - Need cleaning

"""

# Def params
def init_params(layers, activation_func):
    params = {}
    for i in range(1, len(layers)):
        # Initialize weights
        # W = np.random.rand(layers[i], layers [i - 1]) - 0.5  # Constrain the random value between -0.5 to 0.5
        # params[f'W{i}'] = W

        # Initialize weights
        if i == 1:  # For the input layer (W1)
            W = np.random.rand(layers[i], layers[i - 1] - 1) - 0.5  # Subtract 1 from the input dimension
        else:  # For all other layers
            W = np.random.rand(layers[i], layers[i - 1]) - 0.5
        params[f'W{i}'] = W

        # Initialize biases
        b = np.random.rand(layers[i], 1) - 0.5 
        params[f'b{i}'] = b
        
        # Initialize activation func
        params[f'activation{i}'] = activation_func[i - 1]
        
    return params


# One Hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    
    return one_hot_Y

# Activation function
def ReLU(Z):
    return np.maximum(Z, 0)
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Derivative
def ReLU_deriv(Z):
    return Z > 0

########################### Forward propagation (prediction function) ###########################
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

# Backward propagation
# def backward_prop(Z_vals, A_vals, params, X, Y):
    
#     m = Y.size # Number of data
#     one_hot_Y = one_hot(Y)
    
#     # Initialize dict to hold grad for weight and bias
#     dW = {}
#     db = {}
    
#     # Compute grad for the last layer
#     A_last = A_vals[-1]
#     dZ_last = A_last - one_hot_Y # Derivative of the output layer
    
#     num_layers = sum(1 for key in params.keys() if key.startswith('W'))
    
#     # Backpropagation for the last layer
#     W_last = params[f'W{num_layers}']
#     dW[num_layers] = (1 / m) * dZ_last.dot(A_vals[-2].T)  # Gradient for weights of the last layer
#     db[num_layers] = (1 / m) * np.sum(dZ_last, axis=1, keepdims=True)  # Gradient for biases of the last layer

def backward_prop(Z_vals, A_vals, params, X, Y):
    m = Y.size  # Number of samples
    one_hot_Y = one_hot(Y)  # Convert labels to one-hot encoding for Classification problem

    # Initialize dictionaries to store gradients for weights and biases
    dW = {}
    db = {}
    
    # Number of layers in the network
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))

    # Compute gradient for the output layer
    A_last = A_vals[-1]
    dZ = A_last - one_hot_Y  # Gradient of the loss w.r.t. the output layer pre-activation

    # Backpropagation for the last layer
    dW[num_layers] = (1 / m) * dZ.dot(A_vals[-2].T)  # Gradient for weights of the last layer
    db[num_layers] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # Gradient for biases of the last layer

    # Backpropagate through each hidden layer
    for layer in reversed(range(1, num_layers)):
        W_next = params[f'W{layer + 1}']
        Z = Z_vals[layer - 1]  # Pre-activation value for the current layer
        A_prev = A_vals[layer - 1]  # Activation from the previous layer
        
        # Compute dZ for the current layer
        dZ = W_next.T.dot(dZ) * ReLU_deriv(Z)  # Element-wise multiplication with ReLU derivative
        
        # Calculate gradients for the current layer's weights and biases
        dW[layer] = (1 / m) * dZ.dot(A_prev.T)
        db[layer] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

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
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



data = pd.read_csv("./dataset/train.csv")
data = np.array(data)
m, n = data.shape # row and column

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] # Feature, number of data

# Normalization for image input
X_train = X_train / 255.
_,m_train = X_train.shape

data_dev = data[0:1000].T # The first 1000 data, transposed
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

# layer_dims = [n, 100, 100, 10] # First layer has 784 neuron, second layer has 10 neuron, third layer has 10 neuron. First layer is the input layer, last layer is the output layer.
# # gives n-1, so that it could match the size of the input feature

# activation_func = ['ReLu','ReLu','softmax']
# params = init_params(layer_dims, activation_func)

# num_layers = sum(1 for key in params.keys() if key.startswith('W'))  # Count weights to determine the number of layers


# print(params['W1'])

layer_dims = [n, 100, 100, 10] # First layer has 784 neuron, second layer has 10 neuron, third layer has 10 neuron. First layer is the input layer, last layer is the output layer.
# gives n-1, so that it could match the size of the input feature

activation_func = ['ReLu','ReLu','softmax']

# Train the network using gradient descent
params = gradient_descent(X_train, Y_train, activation_func, layer_dims, alpha=0.10, iterations=500)

# Test individual predictions with the trained parameters
for i in range(4):  # Testing the first four samples
    test_prediction(i, X_train, Y_train, params)

# Evaluate the model on the development set
dev_predictions = make_predictions(X_dev, params)
dev_accuracy = get_accuracy(dev_predictions, Y_dev)
print("Development Set Accuracy:", dev_accuracy)


