import pandas as pd

from vni.utils import benchmarking_df, print_metrics

batch_size = 128
n_samples_x = 1024
n_samples_y = 1024

"""df = pd.read_pickle("../../data/df_navigation_pomdp_discrete_actions_0.pkl")
agent0_columns = [col for col in df.columns if "agent_0" in col]
df = df.loc[:10000, agent0_columns]
target_features = ["agent_0_reward"]
intervention_features = ["agent_0_action_0"]"""

df = pd.read_pickle("../../data/FrozenLake-v1.pkl")
df = df.loc[:1000, ["obs_0", "action", "reward"]]

target_features = ["reward"]
intervention_features = ["action"]

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
    density_value=True,
    device="cuda",
)

print(y_true.shape, y_pred.shape)
print_metrics(y_pred, y_true)
