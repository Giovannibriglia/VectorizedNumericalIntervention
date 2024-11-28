from typing import List

import torch
from matplotlib import pyplot as plt

from vni.base import DISTRIBUTIONS_MAP
from vni.base.intervention import InterventionalDistribution
from vni.base.vni import BaseVNI


class PreciseVNI(BaseVNI):
    def __init__(
        self,
        X_prior: torch.Tensor,
        Y_prior: torch.Tensor,
        intervention_indices: List = None,
        n_samples_Y: int = 100,
        **kwargs
    ):
        super().__init__(X_prior, Y_prior, intervention_indices, n_samples_Y, **kwargs)

        self.interventional_dist = InterventionalDistribution(
            DISTRIBUTIONS_MAP["Uniform"]
        )
        self.interventional_std = 1e-4

    def query(
        self,
        X_obs: torch.Tensor,
        intervention_values: torch.Tensor = None,
        Y_seen: torch.Tensor = None,
    ):
        y_values = Y_seen.view(-1, 1) if Y_seen is not None else self.Y_values
        x_value_expanded = X_obs.repeat(y_values.shape[0], 1)

        self.interventional_dist.set_dist_parameters(
            loc=intervention_values,
            scale=torch.full_like(
                intervention_values,
                self.interventional_std,
                dtype=torch.float32,
            ),
        )
        samples = self.interventional_dist.sample((y_values.shape[0],))

        samples = samples.to(x_value_expanded.dtype)
        x_value_expanded[:, self.intervention_indices] = samples

        return self._return_cpds(X_obs, x_value_expanded, y_values)

    @staticmethod
    def plot_pdf(conditional_pdf, y_values, y_true=None, do: bool = False):

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
