"""
Feature selection using Crow Search Algorithm (CSA).
"""

import numpy as np
from typing import List, Tuple, Callable
import torch
from torch import nn

class CSAFeatureSelector:
    """Crow Search Algorithm for feature selection."""
    
    def __init__(self,
                 n_crows: int = 50,
                 max_iter: int = 100,
                 flight_length: float = 2.0,
                 awareness_prob: float = 0.1,
                 objective_fn: Callable = None):
        """
        Initialize CSA feature selector.
        
        Args:
            n_crows (int): Number of crows (population size)
            max_iter (int): Maximum number of iterations
            flight_length (float): Flight length parameter
            awareness_prob (float): Awareness probability
            objective_fn (Callable): Objective function to optimize
        """
        self.n_crows = n_crows
        self.max_iter = max_iter
        self.flight_length = flight_length
        self.awareness_prob = awareness_prob
        self.objective_fn = objective_fn or self._default_objective
        
    def _default_objective(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Default objective function: classification accuracy."""
        with torch.no_grad():
            # Use a simple linear classifier
            classifier = nn.Linear(features.shape[1], labels.max().item() + 1)
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean()
            return accuracy.item()
    
    def _initialize_crows(self, n_features: int) -> np.ndarray:
        """Initialize crow positions randomly."""
        return np.random.rand(self.n_crows, n_features) > 0.5
    
    def _update_position(self,
                        current_pos: np.ndarray,
                        memory: np.ndarray,
                        best_crow: np.ndarray) -> np.ndarray:
        """Update crow position based on memory and best crow."""
        new_pos = current_pos.copy()
        
        for i in range(self.n_crows):
            # Select a random crow to follow
            j = np.random.randint(0, self.n_crows)
            while j == i:
                j = np.random.randint(0, self.n_crows)
            
            # Update position
            if np.random.rand() > self.awareness_prob:
                # Follow the selected crow
                step = self.flight_length * np.random.rand() * (memory[j] - current_pos[i])
                new_pos[i] = current_pos[i] + step
            else:
                # Random position
                new_pos[i] = np.random.rand(n_features) > 0.5
        
        return new_pos
    
    def select_features(self,
                       features: torch.Tensor,
                       labels: torch.Tensor) -> Tuple[np.ndarray, List[float]]:
        """
        Select features using CSA.
        
        Args:
            features (torch.Tensor): Input features
            labels (torch.Tensor): Target labels
            
        Returns:
            Tuple[np.ndarray, List[float]]: Selected feature mask and fitness history
        """
        n_features = features.shape[1]
        
        # Initialize crows
        positions = self._initialize_crows(n_features)
        memory = positions.copy()
        fitness = np.zeros(self.n_crows)
        
        # Calculate initial fitness
        for i in range(self.n_crows):
            selected_features = features[:, positions[i]]
            fitness[i] = self.objective_fn(selected_features, labels)
        
        best_fitness = np.max(fitness)
        best_crow = positions[np.argmax(fitness)]
        fitness_history = [best_fitness]
        
        # Main optimization loop
        for _ in range(self.max_iter):
            # Update positions
            new_positions = self._update_position(positions, memory, best_crow)
            
            # Calculate new fitness
            new_fitness = np.zeros(self.n_crows)
            for i in range(self.n_crows):
                selected_features = features[:, new_positions[i]]
                new_fitness[i] = self.objective_fn(selected_features, labels)
            
            # Update memory
            update_mask = new_fitness > fitness
            memory[update_mask] = new_positions[update_mask]
            fitness[update_mask] = new_fitness[update_mask]
            
            # Update best
            if np.max(fitness) > best_fitness:
                best_fitness = np.max(fitness)
                best_crow = memory[np.argmax(fitness)]
            
            fitness_history.append(best_fitness)
            
            # Update positions
            positions = new_positions
        
        return best_crow, fitness_history
    
    def apply_selection(self,
                       features: torch.Tensor,
                       mask: np.ndarray) -> torch.Tensor:
        """
        Apply feature selection mask to features.
        
        Args:
            features (torch.Tensor): Input features
            mask (np.ndarray): Feature selection mask
            
        Returns:
            torch.Tensor: Selected features
        """
        return features[:, mask] 