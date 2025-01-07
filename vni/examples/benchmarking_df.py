import numpy as np
import pandas as pd

from vni.utils import benchmarking_df, print_metrics

batch_size = 128
n_samples_x = 128
n_samples_y = 128

df = pd.read_pickle("../../data/df_navigation_pomdp_discrete_actions_0.pkl")
agent0_columns = [col for col in df.columns if "agent_0" in col]
df = df.loc[:10000, agent0_columns]
target_features = ["agent_0_reward"]
intervention_features = ["agent_0_action_0"]

"""df = pd.read_pickle("../../data/MountainCar-v0.pkl")
df = df.loc[:5000]
target_features = ["reward"]
intervention_features = ["action"]"""

estimator_config = {
    "estimator": "multivariate_gaussian_kde"
}  # mean_covariance, multivariate_gaussian_kde

y_true, y_pred = benchmarking_df(
    df,
    estimator_config,
    target_features,
    intervention_features,
    batch_size,
    n_samples_x,
    n_samples_y,
    show_res=True,
    density_value=False,
    device="cuda",
)
# print("True: ", y_true[:10])
# print("Pred: ", y_pred[:10])
if isinstance(y_pred, (np.ndarray, list)) and isinstance(y_true, (np.ndarray, list)):
    print_metrics(y_pred, y_true)
