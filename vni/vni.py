from typing import List

import torch
from matplotlib import pyplot as plt

from vni.kde.gaussian_kernel import GaussianKDE


class VNI:
    def __init__(
        self,
        X_prior: torch.Tensor,
        Y_prior: torch.Tensor,
        intervention_indices: List = None,
        n_samples_Y: int = 100,
        **kwargs
    ):
        assert (
            X_prior.shape[0] == Y_prior.shape[0]
        ), "X_prior and Y_prior must have the same number of samples"
        assert (
            X_prior.device == Y_prior.device
        ), "X_prior and Y_prior must have the same device"

        assert isinstance(
            intervention_indices, list
        ), "intervention_indices_X must be a list"

        self.device = "cuda"  # "cuda" if torch.cuda.is_available() else "cpu"
        self.n_samples_Y = n_samples_Y

        self.intervention_indices = intervention_indices or []
        self.intervention_std = 0.1

        self.kde = GaussianKDE(bandwidth=0.5, device=self.device)

        self.current_samples = X_prior.shape[0]

        # Allocate space for max_samples and copy initial data
        self.X_prior = torch.empty(
            (self.current_samples, X_prior.shape[1]), device=self.device
        )
        self.Y_prior = torch.empty(
            (self.current_samples, Y_prior.shape[1]), device=self.device
        )
        self.X_prior[: self.current_samples] = X_prior
        self.Y_prior[: self.current_samples] = Y_prior

        self.Y_values = torch.linspace(
            self.Y_prior.min(), self.Y_prior.max(), self.n_samples_Y, device=self.device
        ).view(
            -1, 1
        )  # self.kde.sample(self.Y_prior, self.n_samples_Y)

    def update_data(self, X_prior: torch.Tensor, Y_prior: torch.Tensor):
        new_samples = X_prior.shape[0]
        assert (
            new_samples == Y_prior.shape[0]
        ), "New X_prior and Y_prior must have the same number of samples"
        assert (
            X_prior.device == Y_prior.device
        ), "X_prior and Y_prior must have the same device"
        assert (
            X_prior.shape[1] == self.X_prior.shape[1]
        ), "New X_prior must match feature dimension"
        assert (
            Y_prior.shape[1] == self.Y_prior.shape[1]
        ), "New Y_prior must match feature dimension"

        # Check if new data fits within allocated space
        end_index = self.current_samples + new_samples

        # Add new data into the allocated space
        self.X_prior[self.current_samples : end_index] = X_prior
        self.Y_prior[self.current_samples : end_index] = Y_prior

        self.current_samples += new_samples

        self.Y_values = torch.linspace(
            self.Y_prior.min(), self.Y_prior.max(), self.n_samples_Y, device=self.device
        ).view(
            -1, 1
        )  # self.kde.sample(self.Y_prior, self.n_samples_Y)

    def set_intervention_indices(self, intervention_indices_X: List):
        assert isinstance(
            intervention_indices_X, list
        ), "intervention_indices_X must be a list"

        self.intervention_indices = intervention_indices_X or []

    def compute_conditional_pdf(
        self,
        X_obs: torch.Tensor,
        intervention_values: torch.Tensor = None,
        Y_seen: torch.Tensor = None,
    ):
        y_values = Y_seen if Y_seen is not None else self.Y_values

        x_value_expanded = X_obs.repeat(y_values.shape[0], 1)

        # Apply Gaussian-distributed interventions to x_value_expanded
        if intervention_values is not None and self.intervention_indices:
            for idx, intervention_value in zip(
                self.intervention_indices, intervention_values
            ):
                # Replace each intervention index in x_value_expanded with Gaussian samples
                gaussian_samples = torch.normal(
                    mean=intervention_value,
                    std=self.intervention_std,
                    size=(y_values.shape[0],),
                    device=self.device,
                )
                x_value_expanded[:, idx] = gaussian_samples

        batch_size, x_dim = x_value_expanded.shape
        _, y_dim = y_values.shape
        # Pre-allocate the joint_points tensor
        joint_points = torch.empty(
            (batch_size, x_dim + y_dim), device=x_value_expanded.device
        )
        # Assign `x_value_expanded` and `y_values` to slices in `joint_points`
        joint_points[:, :x_dim] = x_value_expanded
        joint_points[:, x_dim:] = y_values

        n_samples, n_features_X = self.X_prior.shape
        n_features_Y = self.Y_prior.shape[1]

        # Preallocate XY with the combined number of features
        XY_prior = torch.empty(
            (n_samples, n_features_X + n_features_Y), device=self.X_prior.device
        )

        # Assign X_prior and Y_prior to XY without torch.cat
        XY_prior[:, :n_features_X] = self.X_prior
        XY_prior[:, n_features_X:] = self.Y_prior

        joint_density = self.kde.evaluate(XY_prior, joint_points)  # X-Y

        # Evaluate marginal KDE P(X = x)
        marginal_density = self.kde.evaluate(
            self.X_prior, X_obs.unsqueeze(0)
        ).item()  # scalar X

        # Compute conditional density P(Y | X = x) = P(X = x, Y) / P(X = x)
        conditional_pdf = joint_density / (
            marginal_density + 1e-8
        )  # Add epsilon to avoid division by zero
        return conditional_pdf, self.Y_values

    @staticmethod
    def plot_pdf(conditional_pdf, y_values, y_true=None, do: bool = False):
        # Plot the result
        plt.plot(
            y_values.cpu().numpy(),
            conditional_pdf.cpu().numpy(),
            label="P(Y|X=x)" if not do else "P(Y|do(W=w),X=x)",
        )
        if y_true is not None:
            plt.vlines(
                y_true.cpu().numpy(),
                min(conditional_pdf.cpu().numpy()),
                max(conditional_pdf.cpu().numpy()),
                colors="r",
                linestyles="dashed",
                label="Ground Truth",
            )
        plt.xlabel("Y")
        plt.ylabel("P(Y|X=x)" if not do else "P(Y|do(W=w),X=x)")
        plt.legend()
        plt.show()
