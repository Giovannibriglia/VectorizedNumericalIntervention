from typing import Dict

import yaml

from vni.base.estimator import BaseEstimator
from vni.parametric_estimators.mean_covariance import MeanCovarianceEstimator


def create_estimator(config_params: Dict) -> BaseEstimator:
    return MeanCovarianceEstimator()


def yaml_to_dict(file_path: str) -> dict:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)  # Use safe_load for security
    return data
