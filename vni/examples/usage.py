# import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from vni.vni import VNI

device = "cuda"

# Load DataFrame and filter columns for agent_0
df = pd.read_pickle("../../data/df_navigation_pomdp_discrete_actions_0.pkl")
agent0_columns = [col for col in df.columns if "agent_0" in col]
df = df.loc[:9191, agent0_columns]

# Specify the target column and number of samples
target_column = "agent_0_action_0"  # Replace with your actual target column name
n_samples = 4096

# Separate observed X data and Y target based on specified target column
observed_X = torch.tensor(
    df.drop(columns=[target_column]).iloc[:n_samples].values,
    dtype=torch.float16,
    device="cpu",
)
observed_Y = torch.tensor(
    df[target_column].iloc[:n_samples].values, dtype=torch.float16, device="cpu"
).view(-1, 1)

# Initialize VNI object with the observed data
vni = VNI(observed_X, observed_Y)

mean_error = torch.tensor(0, dtype=torch.float32, device=device)

# Evaluate predictor on each row in df
for n, row in tqdm(df.iterrows(), total=len(df)):
    x_value = torch.tensor(
        row.drop(target_column).values, dtype=torch.float16, device=device
    )  # Current X values as tensor
    y_true = torch.tensor(
        row[target_column], dtype=torch.float16, device=device
    )  # True Y value

    # Compute conditional PDF P(Y | X = x_value) over y_values
    conditional_pdf, y_values = vni.compute_conditional_pdf(x_value)
    # print((max(conditional_pdf.cpu().numpy()) - y_true) ** 2)
    mean_error += (max(conditional_pdf.cpu().numpy()) - y_true) ** 2

    """# Plot the result
    plt.plot(
        y_values.cpu().numpy(), conditional_pdf.cpu().numpy(), label=f"P(Y | X = x)"
    )
    plt.vlines(
        y_true.cpu().numpy(),
        min(conditional_pdf.cpu().numpy()),
        max(conditional_pdf.cpu().numpy()),
        colors="r",
        linestyles="dashed",
        label="Ground Truth",
    )
    plt.xlabel("Y")
    plt.ylabel("P(Y | X = x)")
    plt.title("Conditional PDF P(Y | X = x) over Possible Values of Y")
    plt.legend()
    plt.show()"""

print("MSE error: ", mean_error.item() / len(df))


"""# Process data in batches
for i in tqdm(range(0, len(df), batch_size)):
    # Select batch of X values
    x_values_batch = torch.tensor(
        df.drop(columns=[target_column]).iloc[i : i + batch_size].values,
        dtype=torch.float16,
        device=device,
    )

    # Compute conditional PDF P(Y | X = x_values_batch) for the entire batch
    conditional_pdf_batch = vni.compute_conditional_pdf(x_values_batch)"""