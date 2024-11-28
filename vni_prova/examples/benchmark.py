import pandas as pd

from vni.usage.surface import benchmark_dataframe

df = pd.read_pickle("../../data/df_navigation_pomdp_discrete_actions_0.pkl")
agent0_columns = [col for col in df.columns if "agent_0" in col]
df = df.loc[:9191, agent0_columns]

metrics = benchmark_dataframe(
    df,
    "agent_0_action_0",
    ["agent_0_reward"],
    if_plot=True,
)
print(metrics)
