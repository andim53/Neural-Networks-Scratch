import numpy as np
import pandas as pd

data = pd.read_csv("./dataset/train.csv")
data = np.array(data)
m, n = data.shape # row and column

# Def params
def init_params(layers, activation_func):
    params = {}
    for i in range(1, len(layers)):
        # Initialize weights
        W = np.random.rand(layers[i], layers [i - 1]) - 0.5  # Constrain the random value between -0.5 to 0.5
        params[f'W{i}'] = W

        # Initialize biases
        b = np.random.rand(layers[i], 1) - 0.5 
        params[f'b{i}'] = b
        
        # Initialize activation func
        params[f'activation{i}'] = activation_func[i - 1]
        
    return params

# Activation function
def ReLU(Z):
    return np.maximum(Z, 0)
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Forward propagation (prediction function)
def forward_prop(params, X):
    
    # Initialize input
    A = X
    
    num_layers = len(params) // 2
    
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
        

layer_sizes = [784, 10, 10] # First layer has 784 neuron, second layer has 10 neuron, third layer has 10 neuron. First layer is the input layer, last layer is the output layer.
activation_func = ['ReLu', 'softmax']
params = init_params(layer_sizes, activation_func)

print(params['W1'])