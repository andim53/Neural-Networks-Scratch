import numpy as np
from typing import Tuple, List

class AtomicHopfieldNetwork:
    def __init__(self, n_atoms: int, dimensions: int = 3, learning_rate: float = 0.01):
        """
        Initialize Hopfield Network for atomic structure optimization.
        
        Args:
            n_atoms: Number of atoms in the system
            dimensions: Number of spatial dimensions (default: 3 for 3D space)
            learning_rate: Learning rate for updates
        """
        self.n_atoms = n_atoms
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.n_neurons = n_atoms * dimensions
        
        # Initialize weight matrix
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        # Initialize bias
        self.bias = np.zeros(self.n_neurons)
        
    def lennard_jones_potential(self, positions: np.ndarray) -> float:
        """
        Calculate Lennard-Jones potential energy for given atomic positions.
        
        Args:
            positions: Array of shape (n_atoms, dimensions) containing atomic positions
            
        Returns:
            Total potential energy of the system
        """
        epsilon = 1.0  # Energy parameter
        sigma = 1.0    # Distance parameter
        
        energy = 0.0
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                r = np.linalg.norm(positions[i] - positions[j])
                if r > 0:
                    energy += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
        return energy
    
    def calculate_forces(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate forces on atoms using numerical differentiation.
        
        Args:
            positions: Array of shape (n_atoms, dimensions) containing atomic positions
            
        Returns:
            Forces array of shape (n_atoms, dimensions)
        """
        delta = 1e-6
        forces = np.zeros_like(positions)
        
        for i in range(self.n_atoms):
            for d in range(self.dimensions):
                # Calculate numerical derivative
                positions[i, d] += delta
                energy_plus = self.lennard_jones_potential(positions)
                positions[i, d] -= 2*delta
                energy_minus = self.lennard_jones_potential(positions)
                positions[i, d] += delta
                
                forces[i, d] = -(energy_plus - energy_minus)/(2*delta)
        
        return forces
    
    def update_weights(self, positions: np.ndarray):
        """
        Update weights based on current atomic positions.
        
        Args:
            positions: Array of shape (n_atoms, dimensions) containing atomic positions
        """
        forces = self.calculate_forces(positions)
        positions_flat = positions.flatten()
        forces_flat = forces.flatten()
        
        # Update weights using outer product of positions and forces
        delta_weights = np.outer(forces_flat, positions_flat)
        self.weights += self.learning_rate * (delta_weights + delta_weights.T)/2
        
        # Update bias
        self.bias += self.learning_rate * forces_flat
    
    def optimize_structure(self, 
                         initial_positions: np.ndarray, 
                         max_iterations: int = 1000, 
                         tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize atomic structure using Hopfield network dynamics.
        
        Args:
            initial_positions: Initial atomic positions array of shape (n_atoms, dimensions)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for energy change
            
        Returns:
            Tuple of (optimized positions, energy history)
        """
        positions = initial_positions.copy()
        energy_history = []
        prev_energy = float('inf')
        
        for iteration in range(max_iterations):
            # Calculate current energy
            current_energy = self.lennard_jones_potential(positions)
            energy_history.append(current_energy)
            
            # Check convergence
            if abs(current_energy - prev_energy) < tolerance:
                break
                
            # Update weights based on current configuration
            self.update_weights(positions)
            
            # Update positions using Hopfield dynamics
            positions_flat = positions.flatten()
            activation = np.dot(self.weights, positions_flat) + self.bias
            positions_flat += self.learning_rate * activation
            positions = positions_flat.reshape(self.n_atoms, self.dimensions)
            
            prev_energy = current_energy
            
        return positions, energy_history

# Example usage
def create_random_structure(n_atoms: int, dimensions: int = 3, box_size: float = 10.0) -> np.ndarray:
    """Create random initial atomic structure."""
    return box_size * (np.random.random((n_atoms, dimensions)) - 0.5)

# Initialize the network and optimize a structure
network = AtomicHopfieldNetwork(n_atoms=5)
initial_positions = create_random_structure(n_atoms=5)
optimized_positions, energy_history = network.optimize_structure(initial_positions)