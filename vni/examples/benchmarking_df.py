import pandas as pd

from vni.utils import benchmarking_df

batch_size = 64
n_samples_y = 64

df = pd.read_pickle("../../data/df_navigation_pomdp_discrete_actions_0.pkl")
agent0_columns = [col for col in df.columns if "agent_0" in col]
df = df.loc[:, agent0_columns]

target_features = ["agent_0_reward"]
intervention_features = ["agent_0_action_0"]

"""df = pd.read_csv("../../data/lalonde_cps_sample0.csv")

target_features = [s for s in df.columns if s[0] == "y"]
intervention_features = [s for s in df.columns if s[0] == "t"]"""

y_true, y_pred = benchmarking_df(
    df, target_features, intervention_features, batch_size, n_samples_y, False
)

print(y_true.shape, y_pred.shape)
