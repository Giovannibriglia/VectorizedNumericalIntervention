from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
DISTRIBUTIONS_MAP = None
InterventionalDistribution = None
BaseVNI = None


class OverallVNI(BaseVNI):
    def __init__(
        self,
        X_prior: torch.Tensor,
        Y_prior: torch.Tensor,
        intervention_indices: List = None,
        n_samples_Y: int = 100,
        **kwargs,
    ):
        super().__init__(X_prior, Y_prior, intervention_indices, n_samples_Y, **kwargs)

        self.interventional_dist = InterventionalDistribution(
            DISTRIBUTIONS_MAP["Uniform"]
        )

    def query(
        self,
        X_obs: torch.Tensor,
        intervention_values: torch.Tensor = None,
        Y_seen: torch.Tensor = None,
    ):
        y_values = Y_seen.view(-1, 1) if Y_seen is not None else self.Y_values
        x_value_expanded = X_obs.repeat(y_values.shape[0], 1)

        if self.intervention_indices:

            low = torch.min(self.X_prior[:, self.intervention_indices], dim=0).values
            high = torch.max(self.X_prior[:, self.intervention_indices], dim=0).values

            self.interventional_dist.set_dist_parameters(low=low, high=high)

            samples = self.interventional_dist.sample((y_values.shape[0],))

            samples = samples.to(x_value_expanded.dtype)
            x_value_expanded[:, self.intervention_indices] = samples

        return samples, self._return_cpds(X_obs, x_value_expanded, y_values)

    @staticmethod
    def plot_surface(intervention_values, target_values, conditional_pdf):
        """
        Plot a 3D surface plot of CPD values across intervention and target values.

        Args:
            intervention_values (torch.Tensor): 1D tensor of intervention values.
            target_values (torch.Tensor): 1D tensor of target values.
            conditional_pdf (np.ndarray): 2D array of CPD values with shape (len(target_values), len(intervention_values)).
        """
        print("intervention_values.shape: ", intervention_values.shape)
        print("target_values.shape: ", target_values.shape)
        print("cpd_values.shape: ", conditional_pdf.shape)

        # Check that intervention_values and target_values are 1D
        if intervention_values.ndim != 1 or target_values.ndim != 1:
            raise ValueError(
                "intervention_values and target_values must be 1D tensors."
            )

        # Ensure conditional_pdf is 2D with the correct shape for plotting
        expected_shape = (len(target_values), len(intervention_values))
        if conditional_pdf.shape != expected_shape:
            raise ValueError(
                f"conditional_pdf must be a 2D matrix with shape {expected_shape}, but got {conditional_pdf.shape}"
            )

        # Create a meshgrid for the intervention and target values
        mesh_intervention, mesh_target = np.meshgrid(
            intervention_values.cpu().numpy(), target_values.cpu().numpy()
        )

        # Plot the surface with the provided CPD values
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            mesh_intervention,
            mesh_target,
            conditional_pdf,
            cmap="viridis",
            edgecolor="none",
            alpha=0.7,
        )
        ax.set_xlabel("Intervention Feature Value")
        ax.set_ylabel("Target Feature Value")
        ax.set_zlabel("CPD Value")
        ax.set_title("3D Surface Plot of CPD Values")
        plt.show()
