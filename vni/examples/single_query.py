import pandas as pd
import torch

from vni.utils import setup_vni, single_query

if __name__ == "__main__":
    df = pd.read_pickle("../../data/FrozenLake-v1.pkl")
    df = df.loc[:1024]
    target_features = ["reward"]
    intervention_features = ["action"]

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

    Y_query = torch.tensor(
        df.iloc[:, Y_indices].values, dtype=torch.float32, device=device
    )

    y_pred = single_query(vni, X_query, show_res=True)

    # print(y_pred)
