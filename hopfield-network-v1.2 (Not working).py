#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:10:29 2024

@author: andisyamsul
"""
import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
import colorsys
import seaborn as sns

class VisualizationStyle:
    """Class to handle different visualization styles."""
    
    @staticmethod
    def get_atom_colors(style: str, n_atoms: int) -> np.ndarray:
        """Generate atom colors based on different color schemes."""
        if style == 'rainbow':
            return np.array([colorsys.hsv_to_rgb(i/n_atoms, 0.8, 0.9) for i in range(n_atoms)])
        elif style == 'charge':
            # Simulate charge-based coloring (red for positive, blue for negative)
            charges = np.linspace(-1, 1, n_atoms)
            return np.array([(1-c)/2, 0, (1+c)/2] for c in charges)
        elif style == 'energy':
            # Energy-based coloring (higher energy = warmer color)
            energies = np.linspace(0, 1, n_atoms)
            return np.array([(e, 1-e, 0) for e in energies])
        else:  # default
            return np.array([[0.1, 0.4, 0.8] for _ in range(n_atoms)])
    
    @staticmethod
    def get_bond_style(style: str) -> Dict:
        """Get bond visualization parameters."""
        styles = {
            'solid': {'linestyle': '-', 'alpha': 0.3, 'color': 'k'},
            'dashed': {'linestyle': '--', 'alpha': 0.4, 'color': 'gray'},
            'gradient': {'linestyle': '-', 'alpha': 0.5, 'color': 'lightblue'},
            'none': {'linestyle': 'none', 'alpha': 0, 'color': 'none'}
        }
        return styles.get(style, styles['solid'])

class AtomicHopfieldNetwork:
    # ... (previous initialization code remains the same) ...
    
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

    def visualize_trajectory(self, 
                           interval: int = 100,
                           save_animation: bool = True,
                           style: str = 'default',
                           bond_style: str = 'solid',
                           background_color: str = 'white',
                           show_labels: bool = False,
                           atom_scale: float = 100,
                           view_angle: Optional[Tuple[float, float]] = None,
                           show_unit_cell: bool = False,
                           highlight_atoms: Optional[List[int]] = None):
        """
        Visualize the optimization trajectory in 3D with enhanced styling.
        
        Args:
            interval: Time interval between frames in milliseconds
            save_animation: Whether to save the animation as a gif
            style: Color scheme for atoms ('default', 'rainbow', 'charge', 'energy')
            bond_style: Style for bonds ('solid', 'dashed', 'gradient', 'none')
            background_color: Color of the plot background
            show_labels: Whether to show atom labels
            atom_scale: Size scaling factor for atoms
            view_angle: Tuple of (elevation, azimuth) angles for the view
            show_unit_cell: Whether to show the unit cell box
            highlight_atoms: List of atom indices to highlight
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        ax.set_facecolor(background_color)
        if background_color != 'white':
            fig.patch.set_facecolor(background_color)
        
        # Get atom colors based on style
        atom_colors = VisualizationStyle.get_atom_colors(style, self.n_atoms)
        bond_params = VisualizationStyle.get_bond_style(bond_style)
        
        # Calculate energy per atom for color scaling
        if style == 'energy':
            energies = np.array([self.lennard_jones_potential(pos) for pos in self.trajectory])
            energy_norm = plt.Normalize(energies.min(), energies.max())
        
        def update(frame):
            ax.clear()
            positions = self.trajectory[frame]
            
            # Plot atoms
            for i, pos in enumerate(positions):
                color = atom_colors[i]
                if highlight_atoms and i in highlight_atoms:
                    # Add highlight effect
                    ax.scatter(pos[0], pos[1], pos[2], 
                             c='yellow', marker='o', s=atom_scale*1.5, alpha=0.3)
                
                ax.scatter(pos[0], pos[1], pos[2],
                          c=[color], marker='o', s=atom_scale, alpha=0.8)
                
                if show_labels:
                    ax.text(pos[0], pos[1], pos[2], f'A{i}', 
                           fontsize=8, ha='center', va='center')
            
            # Plot bonds
            if bond_style != 'none':
                cutoff = 2.5
                for i in range(self.n_atoms):
                    for j in range(i+1, self.n_atoms):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < cutoff:
                            if bond_style == 'gradient':
                                # Create gradient-colored bonds
                                points = np.array([positions[i], positions[j]])
                                ax.plot(points[:, 0], points[:, 1], points[:, 2],
                                       c=np.mean([atom_colors[i], atom_colors[j]], axis=0),
                                       **bond_params)
                            else:
                                ax.plot([positions[i,0], positions[j,0]],
                                       [positions[i,1], positions[j,1]],
                                       [positions[i,2], positions[j,2]],
                                       **bond_params)
            
            # Show unit cell if requested
            if show_unit_cell:
                box_size = np.max([pos.max() - pos.min() for pos in positions.T]) * 1.2
                self._draw_unit_cell(ax, box_size)
            
            # Set axis labels and limits
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            
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
            
            # Set view angle if specified
            if view_angle is not None:
                ax.view_init(*view_angle)
            
            # Add title with energy information
            current_energy = self.lennard_jones_potential(positions)
            ax.set_title(f'Frame {frame}/{len(self.trajectory)-1}\nEnergy: {current_energy:.2f}')
        
        ani = FuncAnimation(fig, update, frames=len(self.trajectory),
                          interval=interval, blit=False)
        
        if save_animation:
            ani.save('atomic_optimization.gif', writer='pillow')
        
        plt.show()
    
    def _draw_unit_cell(self, ax, size):
        """Draw unit cell box."""
        # Define the vertices of a cube
        r = [-size/2, size/2]
        points = np.array([[x, y, z] for x in r for y in r for z in r])
        
        # Define the edges
        edges = [
            # Bottom face
            (0,1), (1,3), (3,2), (2,0),
            # Top face
            (4,5), (5,7), (7,6), (6,4),
            # Vertical edges
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        # Plot edges
        for edge in edges:
            ax.plot3D(*zip(points[edge[0]], points[edge[1]]),
                     color='gray', linestyle=':', alpha=0.3)
    
    def plot_energy_convergence(self, style='default'):
        """
        Plot the energy convergence history with enhanced styling.
        
        Args:
            style: Plotting style ('default', 'dark', 'publication')
        """
        energy_history = [self.lennard_jones_potential(pos) for pos in self.trajectory]
        
        if style == 'dark':
            plt.style.use('dark_background')
            color = 'cyan'
        else:
            plt.style.use('default')
            color = 'blue'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear scale plot
        ax1.plot(energy_history, color=color, label='Total Energy')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Convergence (Linear Scale)')
        ax1.grid(True)
        ax1.legend()
        
        # Log scale plot
        ax2.plot(energy_history, color=color, label='Total Energy')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Energy (log scale)')
        ax2.set_title('Energy Convergence (Log Scale)')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
def create_random_structure(n_atoms: int, dimensions: int = 3, box_size: float = 10.0) -> np.ndarray:
    """Create random initial atomic structure."""
    return box_size * (np.random.random((n_atoms, dimensions)) - 0.5)

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize network and create random structure
    network = AtomicHopfieldNetwork(n_atoms=5)
    initial_positions = create_random_structure(n_atoms=5)
    
    # Optimize structure
    optimized_positions, energy_history = network.optimize_structure(initial_positions)
    
    # # Visualize with different styles
    # # Rainbow colored atoms with gradient bonds
    # network.visualize_trajectory(interval=100, 
    #                            style='rainbow', 
    #                            bond_style='gradient',
    #                            show_labels=True,
    #                            atom_scale=150,
    #                            view_angle=(45, 45),
    #                            show_unit_cell=True,
    #                            highlight_atoms=[0, 2])
    
    # # Energy-colored atoms with dark background
    # network.visualize_trajectory(interval=100,
    #                            style='energy',
    #                            bond_style='solid',
    #                            background_color='black',
    #                            atom_scale=120)
    
    # # Plot energy convergence with publication style
    # network.plot_energy_convergence()
