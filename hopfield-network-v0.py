import numpy as np

class HopfieldNetwork:
    """
    Implementation of a Hopfield Neural Network.
    
    A Hopfield network is a form of recurrent artificial neural network that serves
    as a content addressable memory system. The network converges to the stored
    pattern that most closely resembles the input pattern.
    """
    
    def __init__(self, num_neurons):
        """
        Initialize the Hopfield Network.
        
        Args:
            num_neurons (int): Number of neurons in the network
        """
        self.num_neurons = num_neurons
        # Initialize weight matrix with zeros
        # W[i][j] represents the weight of connection from neuron i to neuron j
        self.weights = np.zeros((num_neurons, num_neurons))
        
    def train(self, patterns):
        """
        Train the network using Hebbian learning rule.
        
        The weight matrix W is computed using the formula:
        W[i][j] = Σ(ξᵘ[i] * ξᵘ[j]) for all patterns u, where i ≠ j
        ξᵘ represents the u-th training pattern
        
        Args:
            patterns (list): List of patterns to store. Each pattern should be
                           a numpy array of shape (num_neurons,)
        """
        # Clear existing weights
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        
        # For each pattern
        for pattern in patterns:
            # Outer product of pattern with itself
            # This implements the Hebbian learning rule
            self.weights += np.outer(pattern, pattern)
            
        # Set diagonal elements to 0 (no self-connections)
        np.fill_diagonal(self.weights, 0)
        
        # Divide by number of patterns to normalize
        self.weights /= len(patterns)
    
    def update(self, state, num_iterations=10, mode='async'):
        """
        Update network state according to Hopfield dynamics.
        
        The update rule for each neuron is:
        s[i](t+1) = sign(Σ W[i][j] * s[j](t))
        where s[i] is the state of neuron i
        
        Args:
            state (numpy.array): Initial state of the network
            num_iterations (int): Number of iterations to run
            mode (str): Update mode - 'async' (asynchronous) or 'sync' (synchronous)
            
        Returns:
            numpy.array: Final state of the network
        """
        state = state.copy()
        
        for _ in range(num_iterations):
            if mode == 'async':
                # Update neurons one at a time in random order
                for i in np.random.permutation(self.num_neurons):
                    # Calculate input to neuron i
                    activation = np.dot(self.weights[i], state)
                    # Update state using sign activation function
                    state[i] = np.sign(activation)
            else:  # synchronous update
                # Update all neurons simultaneously
                activation = np.dot(self.weights, state)
                state = np.sign(activation)
                
        return state
    
    def energy(self, state):
        """
        Calculate energy of the network for given state.
        
        The energy function is given by:
        E = -1/2 * Σᵢⱼ W[i][j] * s[i] * s[j]
        
        Args:
            state (numpy.array): Current state of the network
            
        Returns:
            float: Energy of the network
        """
        return -0.5 * np.dot(np.dot(state, self.weights), state)

# Example usage
def example():
    # Create training patterns (simple 3x3 patterns as 1D arrays)
    pattern1 = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1])  # horizontal stripes
    pattern2 = np.array([1, -1, 1, 1, -1, 1, 1, -1, 1])  # vertical stripes
    
    # Initialize network
    network = HopfieldNetwork(9)
    
    # Train network
    network.train([pattern1, pattern2])
    
    # Test with noisy pattern (corrupted version of pattern1)
    noisy_pattern = pattern1.copy()
    noisy_pattern[2] = -1  # flip one bit
    
    # Recover pattern
    recovered_pattern = network.update(noisy_pattern, num_iterations=5)
    
    return pattern1, noisy_pattern, recovered_pattern

if __name__ == "__main__":
    original, noisy, recovered = example()
    print("Original pattern:", original)
    print("Noisy pattern:", noisy)
    print("Recovered pattern:", recovered)