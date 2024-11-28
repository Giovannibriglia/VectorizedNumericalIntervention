from abc import abstractmethod
from typing import List, Tuple

import torch
from torch.distributions import MultivariateNormal, Uniform


class BaseEstimator(object):
    def __init__(
        self,
    ):
        self.XY_prior = None

        self.local_intervention = None
        self.global_intervention = None

        self.X_indices = None
        self.Y_indices = None
        self.intervention_indices = None

    def define_intervention(self):
        self.local_intervention = MultivariateNormal
        self.global_intervention = Uniform

    def set_intervention_indices(self, intervention_indices):
        self.intervention_indices = intervention_indices

    @abstractmethod
    def fit(self, XY: torch.Tensor, X_indices: List, Y_indices: List):
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the CPD of Y given X in batch mode using MultivariateNormal for PDF computation.

        Args:
            X_query (torch.Tensor): Batch of X values, shape [batch_size, n_features_X].
            Y_query (torch.Tensor, optional): Batch of Y query values, shape [batch_size, n_features_Y].
                                               If provided, the CPD will be evaluated for these values.
            X_do (torch.Tensor, optional): Interventional values for X. Defaults to None.
            n_samples (int): Number of samples to generate if Y_query is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - PDF values for the given or generated Y_query, shape [batch_size, n_features_Y, n_samples].
                - Y_values used for evaluation (generated or provided), shape [batch_size, n_features_Y, n_samples].
        """
        raise NotImplementedError


class BaseParametricEstimator(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.prior_parameters = None

    def fit(self, XY: torch.Tensor, X_indices: List[int], Y_indices: List[int]):
        """
        Fits the model to the given data and checks index coverage and overlap.

        Args:
            XY (torch.Tensor): The dataset to fit, with shape (n_features, n_samples).
            X_indices (List[int]): Indices for X features.
            Y_indices (List[int]): Indices for Y features.

        Raises:
            ValueError: If not all indices are covered or if there are overlapping indices.
        """
        self.XY_prior = XY

        self.prior_parameters = self._compute_prior_parameters(self.XY_prior)

        # Check for overlapping indices
        overlap = set(X_indices) & set(Y_indices)
        if overlap:
            raise ValueError(
                f"Overlap detected between X_indices and Y_indices: {overlap}"
            )

        # Set indices if checks pass
        self.X_indices = X_indices
        self.Y_indices = Y_indices

    @abstractmethod
    def predict(
        self,
        X_query: torch.Tensor,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _compute_prior_parameters(self, XY: torch.Tensor):
        raise NotImplementedError
