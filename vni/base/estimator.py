from abc import abstractmethod
from typing import List, Tuple

import torch
from torch.distributions import Uniform


class BaseEstimator(object):
    def __init__(
        self,
        X_indices: List[int],
        Y_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        self.XY_prior = None
        self.XY_prior_intervention = None

        assert not (
            set(X_indices) & set(Y_indices)
        ), "X_indices and Y_indices have overlapping items"
        self.X_indices = X_indices
        self.Y_indices = Y_indices

        self.intervention_indices = (
            None if len(intervention_indices) == 0 else intervention_indices
        )

        self.local_intervention = None
        self.global_intervention = None

    @abstractmethod
    def fit(
        self,
        XY: torch.Tensor,
    ):
        raise NotImplementedError

    def _fit(
        self,
        XY: torch.Tensor,
    ):
        """
        Fits the model to the given data and checks index coverage and overlap.

        Args:
            XY (torch.Tensor): The dataset to fit, with shape (n_features, n_samples).

        Raises:
            ValueError: If not all indices are covered or if there are overlapping indices.
        """
        self.XY_prior = XY

        if self.intervention_indices is not None:
            # Clone the original tensor
            new_XY = XY.clone()

            # Get the slice of the tensor corresponding to the intervention indices
            intervention_values = XY[self.intervention_indices, :]

            # Calculate the low and high for the uniform distribution
            low = intervention_values.min(
                dim=1, keepdim=True
            ).values  # Shape: [len(intervention_indices), 1]
            high = intervention_values.max(
                dim=1, keepdim=True
            ).values  # Shape: [len(intervention_indices), 1]

            # Create a uniform distribution with the calculated bounds
            uniform_dist = Uniform(low, high)

            # Generate uniform random values within the calculated range
            # The shape should match [len(intervention_indices), new_XY.shape[1]]
            new_samples = uniform_dist.sample(
                (new_XY.shape[1],)
            ).T  # Transpose to align dimensions

            # Update the specified rows of the tensor with the generated samples
            new_XY[self.intervention_indices, :] = new_samples

            self.XY_prior_intervention = new_XY

    @abstractmethod
    def predict(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the CPD of Y given X in batch mode using MultivariateNormal for PDF computation.

        Args:
            X_query (torch.Tensor): Batch of X values, shape [batch_size, n_features_X-n_features_X_do].
            Y_query (torch.Tensor, optional): Batch of Y query values, shape [batch_size, n_features_Y].
                                               If provided, the CPD will be evaluated for these values.
            X_do (torch.Tensor, optional): Interventional values for X. Defaults to None. [batch_size, n_features_X_do]
            n_samples (int): Number of samples to generate if Y_query is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - PDF values for the given or generated Y_query, shape [batch_size, n_features_Y, n_samples].
                - Y_values used for evaluation (generated or provided), shape [batch_size, n_features_Y, n_samples].
        """
        raise NotImplementedError

    def _define_Y_values(self, Y_query, n_samples, batch_size):
        if Y_query is None:
            # Calculate min and max for each feature in Y
            Y_min = torch.min(
                self.XY_prior[self.Y_indices, :], dim=1
            ).values  # Shape: [n_target_features]
            Y_max = torch.max(
                self.XY_prior[self.Y_indices, :], dim=1
            ).values  # Shape: [n_target_features]

            # Create a linspace template
            linspace_template = torch.linspace(
                0, 1, n_samples, device=self.XY_prior.device
            ).unsqueeze(
                0
            )  # [1, n_samples]

            # Scale the linspace to each feature's range
            Y_values = Y_min.unsqueeze(1) + linspace_template * (
                Y_max - Y_min
            ).unsqueeze(
                1
            )  # [n_target_features, n_samples]

            # Expand Y_values to match batch size
            Y_values = Y_values.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # [batch_size, n_target_features, n_samples]
        else:
            Y_values = Y_query.clone().unsqueeze(
                -1
            )  # [batch_size, n_target_features, 1]

        return Y_values

    def _check_output(
        self,
        pdf: torch.Tensor,
        Y_values: torch.Tensor,
        Y_query: torch.Tensor,
        batch_size: int,
        n_samples: int,
    ):
        assert (
            pdf.shape
            == Y_values.shape
            == (
                batch_size,
                len(self.Y_indices),
                n_samples if Y_query is None else 1,
            )
        ), print(
            f"pdf and/or y_values shape are wrong: must be "
            f"{(batch_size, len(self.Y_indices), n_samples if Y_query is None else 1)}, "
            f"instead pdf.shape: {pdf.shape} and y_values.shape: {Y_values.shape}"
        )


class BaseParametricEstimator(BaseEstimator):
    def __init__(
        self,
        X_indices: List[int],
        Y_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        super().__init__(X_indices, Y_indices, intervention_indices)
        self.prior_parameters = None
        self.prior_parameters_after_interventions = None

    def fit(self, XY: torch.Tensor):
        self._fit(XY)
        self.prior_parameters = self._compute_prior_parameters(self.XY_prior)

        if self.intervention_indices is not None:
            self.prior_parameters_after_interventions = self._compute_prior_parameters(
                self.XY_prior_intervention
            )

    @abstractmethod
    def predict(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _compute_prior_parameters(self, XY: torch.Tensor):
        raise NotImplementedError


class BaseNonParametricEstimator(BaseEstimator):
    def __init__(
        self,
        X_indices: List[int],
        Y_indices: List[int],
        intervention_indices: List[int] = None,
    ):
        super().__init__(X_indices, Y_indices, intervention_indices)

    def fit(self, XY: torch.Tensor):
        self._fit(XY)

    @abstractmethod
    def predict(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
