from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


class BaseEstimator(object):
    def __init__(self):
        self.data = None

    def get_parameters(self):
        raise NotImplementedError

    def add_data(self, data: torch.Tensor):
        """
        Adds new data to the existing data. If no existing data is present, initializes the data attribute.

        Args:
            data (torch.Tensor): New data to add.
        """
        if not isinstance(data, torch.Tensor):
            raise ValueError("The input data must be a torch.Tensor")

        if self.data is None:
            # If no data is present, set self.data to the new data
            self.data = data
        else:
            # If data exists, concatenate the new data to the existing data
            self.data = torch.cat((self.data, data), dim=0)

    def set_data(self, data: torch.Tensor):
        """Sets the data, replacing any existing data."""
        self.data = data

    @staticmethod
    def plot_gaussian_contours_torch(
        mu: torch.Tensor, sigma: torch.Tensor, pairs: List
    ):
        """
        Plot contour plots for pairs of dimensions from a multivariate Gaussian distribution
        with mu and sigma as torch tensors (supporting CUDA).

        Args:
            mu (torch.Tensor): Mean vector of shape [dim] (can be on CUDA).
            sigma (torch.Tensor): Covariance matrix of shape [dim, dim] (can be on CUDA).
            pairs (list of tuples): List of dimension pairs to plot, e.g., [(0, 1), (2, 3)].

        Returns:
            None: Displays the contour plots.
        """
        # Move mu and sigma to CPU and convert to NumPy arrays if they are torch tensors
        if isinstance(mu, torch.Tensor):
            mu = mu.cpu().numpy()
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.cpu().numpy()

        sigma_copy = sigma.copy()  # Make a copy to modify
        np.fill_diagonal(sigma_copy, sigma_copy.diagonal() + 1e-8)

        # Create grid for plotting
        x, y = np.mgrid[0:1:0.01, 0:1:0.01]
        pos = np.dstack((x, y))

        # Handle the case where pairs has only one tuple
        if len(pairs) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            axes = [ax]  # Convert to list for consistent handling below
        else:
            fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5))

        for i, (d1, d2) in enumerate(pairs):
            # Extract the 2D mean and covariance for the current pair
            mu_2d = mu[[d1, d2]]
            sigma_2d = sigma_copy[[d1, d2], :][
                :, [d1, d2]
            ]  # Extract the 2x2 covariance matrix

            # Create a multivariate normal distribution for this 2D case
            rv = multivariate_normal(mean=mu_2d, cov=sigma_2d, allow_singular=True)

            # Plot contour on the corresponding axis
            ax = axes[i] if len(pairs) > 1 else axes[0]
            ax.contour(x, y, rv.pdf(pos), levels=5, alpha=0.7)
            ax.scatter(mu_2d[0], mu_2d[1], marker="o", color="r")
            ax.set_title(f"Dims {d1}, {d2}")
            ax.set_xlabel(f"Dim {d1}")
            ax.set_ylabel(f"Dim {d2}")
            ax.grid(True)

        plt.suptitle("Contour Plots for Different Pairs of Dimensions")
        plt.tight_layout()
        plt.show()
