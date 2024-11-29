from typing import Dict, List

import yaml

from vni.base.estimator import BaseEstimator
from vni.non_parametric_estimators.kde import GaussianKDE

# from vni.parametric_estimators.mean_covariance import MeanCovarianceEstimator


def create_estimator(
    config_params: Dict,
    X_indices: List[int],
    Y_indices: List[int],
    intervention_indices: List[int] = None,
) -> BaseEstimator:
    return GaussianKDE(X_indices, Y_indices, intervention_indices)


def yaml_to_dict(file_path: str) -> dict:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)  # Use safe_load for security
    return data
