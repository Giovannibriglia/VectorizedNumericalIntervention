from abc import abstractmethod
from typing import List, Tuple

import torch


class BaseEstimator(object):
    def __init__(
        self,
    ):
        self.XY_prior = None

        self.X_indices = None
        self.X_do_indices = None
        self.Y_indices = None

        self.all_X_indices = None

    def set_indices(self, X_indices: List, Y_indices: List, X_do_indices: List = []):
        assert not (
            set(X_indices) & set(Y_indices)
        ), "X_indices and Y_indices have overlapping items"

        self.X_indices = X_indices
        self.Y_indices = Y_indices

        assert not (
            set(X_indices) & set(X_do_indices)
        ), "X_indices and X_do_indices have overlapping items"

        self.X_do_indices = X_do_indices

        self.all_X_indices = X_indices + X_do_indices

    def fit(
        self,
        XY: torch.Tensor,
    ):
        self.XY_prior = XY
        self._fit()
        # TODO: update data

    def predict(
        self,
        X_query: torch.Tensor = None,
        Y_query: torch.Tensor = None,
        X_do: torch.Tensor = None,
        n_samples_x: int = 1024,
        n_samples_y: int = 1024,
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
        assert (
            self.X_indices is not None
        ), "X_indices must be set before calling predict"
        assert (
            self.Y_indices is not None
        ), "Y_indices must be set before calling predict"
        assert (
            self.X_do_indices is not None
        ), "X_do_indices must be set before calling predict"

        batch_size = (
            X_query.shape[0]
            if X_query is not None
            else (
                Y_query.shape[0]
                if Y_query is not None
                else X_do.shape[0] if X_do is not None else 1
            )
        )

        if X_query is None:
            X_query_values = self._define_values(
                self.X_indices, n_samples_x, batch_size
            )
        else:
            X_query_values = X_query.unsqueeze(
                -1
            )  # Avoid clone unless modification is needed

        if X_do is None:
            X_do_query_values = self._define_values(
                self.X_do_indices, n_samples_x, batch_size
            )
        else:
            X_do_query_values = X_do.unsqueeze(
                -1
            )  # Avoid clone unless modification is needed

        # Combine indices dynamically
        all_used_indices = torch.tensor(
            sorted(set(self.X_indices + self.X_do_indices)),
            device=X_query_values.device,
        )

        if all_used_indices.numel() == 0:
            raise ValueError("No valid indices found in X_indices or X_do_indices.")

        # Dynamically allocate the tensor based on the used indices
        X_query_concat_values = torch.zeros(
            (
                batch_size,
                all_used_indices.numel(),
                max(X_query_values.size(-1), X_do_query_values.size(-1)),
            ),
            device=X_query_values.device,
            dtype=X_query_values.dtype,
        )

        # Create a mapping tensor for X_indices and X_do_indices to their positions in all_used_indices
        X_indices_tensor = torch.tensor(self.X_indices, device=X_query_values.device)
        X_do_indices_tensor = torch.tensor(
            self.X_do_indices, device=X_query_values.device
        )

        X_indices_mapped = (
            X_indices_tensor.unsqueeze(1) == all_used_indices.unsqueeze(0)
        ).nonzero(as_tuple=True)[1]
        X_do_indices_mapped = (
            X_do_indices_tensor.unsqueeze(1) == all_used_indices.unsqueeze(0)
        ).nonzero(as_tuple=True)[1]

        # Assign values using advanced indexing
        X_query_concat_values[:, X_indices_mapped, : X_query_values.size(-1)] = (
            X_query_values
        )
        X_query_concat_values[:, X_do_indices_mapped, : X_do_query_values.size(-1)] = (
            X_do_query_values
        )

        if Y_query is None:
            Y_query_values = self._define_values(
                self.Y_indices, n_samples_y, batch_size
            )  # [batch_size, n_features_Y, n_samples_y]
        else:
            Y_query_values = Y_query.clone().unsqueeze(
                -1
            )  # [batch_size, n_features_Y, 1]

        pdf, y_values = self._predict(X_query_concat_values, Y_query_values)

        self._check_output(pdf, y_values, Y_query, batch_size, n_samples_y)

        return pdf, y_values

    @abstractmethod
    def _predict(self, X_values: torch.Tensor, Y_values: torch.Tensor):
        raise NotImplementedError

    def _define_values(self, indices, n_samples, batch_size):
        min_value = torch.min(
            self.XY_prior[indices, :], dim=1
        ).values  # Shape: [n_features]
        max_value = torch.max(
            self.XY_prior[indices, :], dim=1
        ).values  # Shape: [n_features]

        # Create a linspace template
        linspace_template = torch.linspace(
            0, 1, n_samples, device=self.XY_prior.device
        ).unsqueeze(
            0
        )  # [1, n_samples]

        # Scale the linspace to each feature's range
        values_linspace = min_value.unsqueeze(1) + linspace_template * (
            max_value - min_value
        ).unsqueeze(
            1
        )  # [n_features, n_samples]

        # Expand values_linspace to match batch size
        values_linspace = values_linspace.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, n_features, n_samples]

        return values_linspace

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

    @abstractmethod
    def _fit(self):
        raise NotImplementedError


class BaseParametricEstimator(BaseEstimator):
    def __init__(self):
        super().__init__()

    def _fit(self):
        self.prior_parameters = self._compute_prior_parameters(self.XY_prior)

    @abstractmethod
    def _compute_prior_parameters(self, XY: torch.Tensor):
        raise NotImplementedError


class BaseNonParametricEstimator(BaseEstimator):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _fit(self):
        raise NotImplementedError
