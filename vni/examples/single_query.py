import pandas as pd
import torch

from vni.utils import setup_vni, single_query

if __name__ == "__main__":
    df = pd.read_pickle("../../data/FrozenLake-v1.pkl")
    df = df.loc[:1024, ["obs_0", "action", "reward"]]
    target_features = ["action"]
    intervention_features = ["reward"]

    estimator_config = {
        "estimator": "multivariate_gaussian_kde"
    }  # mean_covariance, multivariate_gaussian_kde

    device = "cuda"

    vni, XY_prior_tensor, X_indices, Y_indices, intervention_indices = setup_vni(
        df, target_features, intervention_features, estimator_config, device
    )

    X_query = torch.tensor(
        df.iloc[:, X_indices].values, dtype=torch.float32, device=device
    )

    single_query(vni, X_query, show_res=True)

    """
    vni.set_indices(X_indices, Y_indices, intervention_indices)

    pdf, y_values = vni.query(X_query, n_samples_x=64, n_samples_y=64)

    vni.plot_result(pdf, y_values)"""
