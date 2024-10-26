import pandas as pd
import torch
from tqdm import tqdm

from vni.base.pipeline import Pipeline
from vni.utils import df_to_batched_tensor_with_index

# set dataframe
df = pd.read_pickle("df_example.pkl")

# setup dataframe
agent0_columns = [col for col in df.columns if "agent_0" in col]
df = df.loc[:, agent0_columns]

# set feature
feature_target_names = ["agent_0_reward"]

n_evaluations = 10
batch_size = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"

n_features = len(df.columns)

df_torch, feature_name_index = df_to_batched_tensor_with_index(
    df, n_evaluations, device=device
)

df_torch = df_torch.to(device)

target_indexes = [feature_name_index[feat] for feat in feature_target_names]
obs_indexes = [
    feature_name_index[feat]
    for feat in feature_name_index
    if feature_name_index[feat] not in target_indexes
]

pipeline = Pipeline(device=device)

pipeline.set_data(df_torch)

total_elements = df_torch.shape[2]
min_values_target = [df[feat].min() for feat in feature_target_names]
max_values_target = [df[feat].max() for feat in feature_target_names]

with tqdm(total=total_elements, desc="Processing Elements") as pbar:
    for i in range(0, total_elements, batch_size):
        # Select N batches at a time, with shape (N, n_features, n_samples)
        batch = df_torch[:, :, i : i + batch_size]

        # Prepare dict_input for pipeline
        dict_input = {
            "target_indexes": torch.tensor(target_indexes, device=device),
            "obs_indexes": torch.tensor(obs_indexes, device=device),
            "obs_values": batch[:, obs_indexes],  # Select obs_values for this chunk
        }

        mean_value, prediction_uncertainty = pipeline.get_cpds(**dict_input)

        assert (
            mean_value.shape[0] == batch[:, target_indexes].shape[1]
        ), "output must has the same shape of df"

        if i == 0:
            kwargs = {
                "mu": mean_value,
                "sigma": prediction_uncertainty,
                "feature_names": feature_target_names,
                "min_values": min_values_target,
                "max_values": max_values_target,
            }

            pipeline.estimator_plot_contours(**kwargs)

        pbar.update(min(batch_size, total_elements - i))
