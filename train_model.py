import numpy as np
import pandas as pd

data = pd.read_csv("./dataset/train.csv")
data = np.array(data)
m, n = data.shape # row and column

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] # Feature, number of data

X_train = X_train / 255.
_,m_train = X_train.shape

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
def backward_prop(Z_vals, A_vals, params, X, Y):
    
    m = Y.size # Number of data
    one_hot_Y = one_hot(Y)
    
    # Initialize dict to hold grad for weight and bias
    dW = {}
    db = {}
    
    # Compute grad for the last layer
    A_last = A_vals[-1]
    dZ_last = A_last - one_hot_Y # Derivative of the output layer
    
    num_layers = sum(1 for key in params.keys() if key.startswith('W'))
    
    # Backpropagation for the last layer
    W_last = params[f'W{num_layers}']
    dW[num_layers] = (1 / m) * dZ_last.dot(A_vals[-2].T)  # Gradient for weights of the last layer
    db[num_layers] = (1 / m) * np.sum(dZ_last, axis=1, keepdims=True)  # Gradient for biases of the last layer

    


# One Hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    
    return one_hot_Y

# Derivative
def ReLU_deriv(Z):
    return Z > 0
        

layer_sizes = [n-1, 100, 100, 10] # First layer has 784 neuron, second layer has 10 neuron, third layer has 10 neuron. First layer is the input layer, last layer is the output layer.
# gives n-1, so that it could match the size of the input feature

activation_func = ['ReLu','ReLu','softmax']
params = init_params(layer_sizes, activation_func)

num_layers = sum(1 for key in params.keys() if key.startswith('W'))  # Count weights to determine the number of layers


print(params['W1'])