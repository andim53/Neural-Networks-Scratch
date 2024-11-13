import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

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
        
        # Store trajectory for visualization
        self.trajectory = []
        
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
        
        # Clear previous trajectory
        self.trajectory = [positions.copy()]
        
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
            
            # Store positions for visualization
            self.trajectory.append(positions.copy())
            
            prev_energy = current_energy
            
        return positions, energy_history

    def visualize_trajectory(self, interval: int = 100, save_animation: bool = False):
        """
        Visualize the optimization trajectory in 3D.
        
        Args:
            interval: Time interval between frames in milliseconds
            save_animation: Whether to save the animation as a gif
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            positions = self.trajectory[frame]
            
            # Plot atoms
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                      c='b', marker='o', s=100, alpha=0.6)
            
            # Plot bonds (connect atoms within a cutoff distance)
            cutoff = 2.5  # Adjust this value based on your system
            for i in range(self.n_atoms):
                for j in range(i+1, self.n_atoms):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < cutoff:
                        ax.plot([positions[i,0], positions[j,0]],
                               [positions[i,1], positions[j,1]],
                               [positions[i,2], positions[j,2]], 'k-', alpha=0.3)
            
            # Set axis labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set consistent axis limits
            max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                                positions[:, 1].max() - positions[:, 1].min(),
                                positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
            mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
            mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
            mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.set_title(f'Frame {frame}/{len(self.trajectory)-1}')
        
        ani = FuncAnimation(fig, update, frames=len(self.trajectory),
                          interval=interval, blit=False)
        
        if save_animation:
            ani.save('atomic_optimization.gif', writer='pillow')
        
        plt.show()
    
    def plot_energy_convergence(self):
        """Plot the energy convergence history."""
        energy_history = [self.lennard_jones_potential(pos) for pos in self.trajectory]
        
        plt.figure(figsize=(10, 6))
        plt.plot(energy_history, '-b', label='Total Energy')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Energy Convergence During Optimization')
        plt.grid(True)
        plt.legend()
        plt.yscale('log')
        plt.show()

# Example usage
def create_random_structure(n_atoms: int, dimensions: int = 3, box_size: float = 10.0) -> np.ndarray:
    """Create random initial atomic structure."""
    return box_size * (np.random.random((n_atoms, dimensions)) - 0.5)

# Run optimization with visualization
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize network and create random structure
    network = AtomicHopfieldNetwork(n_atoms=2)
    # initial_positions = create_random_structure(n_atoms=2)
    
    # Position of MgO
    initial_positions = np.array([[0,0,2.29897550000000], [0,0,2.10102550000000]])
    
    # Optimize structure
    optimized_positions, energy_history = network.optimize_structure(initial_positions)
    
    # Visualize results
    network.visualize_trajectory(interval=100, save_animation=True)
    network.plot_energy_convergence()