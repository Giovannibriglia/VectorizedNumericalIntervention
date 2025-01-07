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
            n_samples_x (int): Number of samples to generate if X_query is None.
            n_samples_y (int): Number of samples to generate if Y_query is None.

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

        # TODO: handle X_do_indices empty
        if len(self.X_do_indices) > 0:
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

        # self._ensure_compatibility(pdf, y_values)

        return pdf, y_values

    @abstractmethod
    def _predict(self, X_values: torch.Tensor, Y_values: torch.Tensor):
        raise NotImplementedError

    def _define_values(self, indices, n_samples, batch_size):
        # Extract data for the specified indices
        data = self.XY_prior[indices, :]  # Shape: [n_features, n_samples_prior]

        # Compute unique values and determine if features are continuous
        unique_values_list = [torch.unique(data[i, :]) for i in range(data.shape[0])]
        unique_lengths = torch.tensor(
            [len(vals) for vals in unique_values_list], device=data.device
        )
        is_continuous = unique_lengths > 10

        # Initialize variables for continuous and discrete handling
        continuous_indices = torch.empty(0, dtype=torch.long, device=data.device)
        discrete_indices = torch.empty(0, dtype=torch.long, device=data.device)
        continuous_values = None
        discrete_values = None

        # Prepare a tensor to hold all unique values for discrete features
        max_unique = max(unique_lengths)
        padded_unique_values = torch.zeros(
            (data.shape[0], max_unique), device=data.device
        )
        for i, vals in enumerate(unique_values_list):
            padded_unique_values[i, : len(vals)] = vals

        # Generate continuous values
        if torch.any(is_continuous):
            continuous_indices = is_continuous.nonzero(as_tuple=True)[0]
            min_values = torch.min(
                data[continuous_indices, :], dim=1
            ).values  # [n_continuous_features]
            max_values = torch.max(
                data[continuous_indices, :], dim=1
            ).values  # [n_continuous_features]

            linspace_template = torch.linspace(
                0, 1, n_samples, device=data.device
            ).unsqueeze(
                0
            )  # [1, n_samples]
            continuous_values = min_values.unsqueeze(1) + linspace_template * (
                max_values - min_values
            ).unsqueeze(
                1
            )  # [n_continuous_features, n_samples]

            continuous_values = continuous_values.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # [batch_size, n_continuous_features, n_samples]

        # Generate discrete values
        if torch.any(~is_continuous):
            discrete_indices = (~is_continuous).nonzero(as_tuple=True)[0]
            discrete_unique_values = padded_unique_values[
                discrete_indices
            ]  # [n_discrete_features, max_unique]
            discrete_probs = torch.ones_like(discrete_unique_values) / unique_lengths[
                discrete_indices
            ].unsqueeze(
                1
            )  # Normalize probabilities

            # Sample indices for discrete features
            sampled_indices = torch.multinomial(
                discrete_probs, n_samples, replacement=True
            )  # [n_discrete_features, n_samples]

            # Map sampled indices to values
            discrete_values = discrete_unique_values.gather(
                1, sampled_indices
            )  # [n_discrete_features, n_samples]

            # Expand to batch size
            discrete_values = discrete_values.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # [batch_size, n_discrete_features, n_samples]

        # Combine continuous and discrete features
        combined_values = torch.zeros(
            (batch_size, len(indices), n_samples), device=data.device
        )
        if continuous_values is not None:
            combined_values[:, continuous_indices, :] = continuous_values
        if discrete_values is not None:
            combined_values[:, discrete_indices, :] = discrete_values

        return combined_values

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

        # Check for NaN values
        assert not torch.isnan(pdf).any(), "The PDF tensor contains NaN values."
        # Check that all values are >= 0
        assert (pdf >= 0).all(), "The PDF tensor contains negative values."

    @abstractmethod
    def _fit(self):
        raise NotImplementedError

    def _ensure_compatibility(self, pdf: torch.Tensor, y_values: torch.Tensor):
        """
        Ensure compatibility with the real data for discrete or continuous target features.

        :param pdf: [batch_size, n_features_Y, n_samples]
        :param y_values: [batch_size, n_features_Y, n_samples]
        :return: Adjusted pdf and y_values
        """
        true_values = self.XY_prior[self.Y_indices, :]  # [n_features_Y, n_samples_data]
        batch_size, n_features_Y, n_samples = pdf.shape

        # Identify unique values and their counts for each feature
        unique_values_list = [
            torch.unique(true_values[f, :]) for f in range(n_features_Y)
        ]
        unique_counts = torch.tensor(
            [len(u) for u in unique_values_list], device=pdf.device
        )

        # Create a mask for features with fewer than 20 unique values
        is_discrete = unique_counts < 20

        # Temporary copy of y_values and pdf for discrete features
        updated_y_values = y_values.clone()
        updated_pdf = torch.zeros(
            (batch_size, n_features_Y, n_samples), device=pdf.device
        )

        for feature_idx in range(n_features_Y):
            if is_discrete[feature_idx]:
                unique_values = unique_values_list[
                    feature_idx
                ]  # Discrete values for this feature

                # Round y_values to nearest discrete unique values
                y_values_feature = y_values[
                    :, feature_idx, :
                ]  # [batch_size, n_samples]
                expanded_unique_values = unique_values.view(
                    1, -1, 1
                )  # [1, n_unique, 1]
                distances = torch.abs(
                    y_values_feature.unsqueeze(1) - expanded_unique_values
                )  # [batch_size, n_unique, n_samples]
                nearest_indices = torch.argmin(
                    distances, dim=1
                )  # [batch_size, n_samples]
                rounded_values = unique_values[
                    nearest_indices
                ]  # [batch_size, n_samples]

                # Update the temporary y_values
                updated_y_values[:, feature_idx, :] = rounded_values

                # Adjust pdf: aggregate probabilities for each unique value
                matches = rounded_values.unsqueeze(
                    -1
                ) == expanded_unique_values.squeeze(
                    -1
                )  # [batch_size, n_samples, n_unique]
                matches = (
                    matches.float()
                )  # Convert boolean mask to float for multiplication

                aggregated_pdf = torch.matmul(
                    matches.transpose(1, 2), pdf[:, feature_idx, :].unsqueeze(-1)
                ).squeeze(
                    -1
                )  # [batch_size, n_unique]

                # Resize updated_pdf for discrete features
                updated_pdf[:, feature_idx, : aggregated_pdf.shape[1]] = aggregated_pdf
            else:
                # For continuous features, retain the original pdf
                updated_pdf[:, feature_idx, :] = pdf[:, feature_idx, :]

        return updated_pdf, updated_y_values


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
