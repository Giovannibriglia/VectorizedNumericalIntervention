import pandas as pd
import torch
from matplotlib import pyplot as plt


class KDE:
    def __init__(self, bandwidth=1.0, device: str = "cpu"):
        self.device = device
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit KDE to data."""
        self.data = data.to(self.device)

    def evaluate(self, points):
        """Evaluate KDE for the provided points."""
        if self.data is None:
            raise ValueError("KDE must be fitted with data before evaluation.")

        # Pairwise distance squared between `points` and `self.data`
        diff = points[:, None, :] - self.data[None, :, :]
        dist_sq = torch.sum(diff**2, dim=-1)

        # KDE estimate
        kernel_vals = torch.exp(-0.5 * dist_sq / self.bandwidth**2)
        return kernel_vals.mean(dim=1) / (
            self.bandwidth * (2 * torch.pi) ** (points.shape[1] / 2)
        )


class VNI:
    def __init__(self, observed, target, bandwidth=1.0):
        # Fit KDE on joint data [observed, target]
        joint_data = torch.cat([observed, target], dim=1)
        self.joint_kde = KDE(bandwidth=bandwidth)
        self.joint_kde.fit(joint_data)

        # Fit KDE on observed data only to get P(X = x)
        self.obs_kde = KDE(bandwidth=bandwidth)
        self.obs_kde.fit(observed)

    def compute_conditional_pdf(self, x_value, y_range_values):
        """
        Compute the conditional PDF P(Y | X = x_value) over possible values of Y using KDE.
        Args:
            observed (torch.Tensor): Observed data tensor of shape [n_samples, n_obs_features].
            target (torch.Tensor): Target data tensor of shape [n_samples, n_target_features].
            x_value (torch.Tensor): Specific observed value at which to condition (1D tensor).
            y_range_values (torch.Tensor): Grid of possible target values for Y.
            bandwidth (float): Bandwidth for KDE.

        Returns:
            torch.Tensor: Conditional PDF values for each point in y_values.
        """
        # Evaluate joint KDE P(X = x, Y = y) over the range of y_values
        x_value_expanded = x_value.expand(y_range_values.shape[0], -1)
        joint_points = torch.cat([x_value_expanded, y_range_values], dim=1)
        joint_density = self.joint_kde.evaluate(joint_points)

        # Evaluate marginal KDE P(X = x)
        marginal_density = self.obs_kde.evaluate(x_value.unsqueeze(0)).item()  # scalar

        # Compute conditional density P(Y | X = x) = P(X = x, Y) / P(X = x)
        conditional_pdf = joint_density / (
            marginal_density + 1e-8
        )  # Add epsilon to avoid division by zero
        return conditional_pdf


"""# Generate some example data (replace with actual data)
observed = torch.randn(100, 2)  # Observed features (e.g., X)
target = torch.randn(100, 1)    # Target features (e.g., Y)
x_value = torch.tensor([0.53, -0.21])  # Conditioned value for observed features
y_values = torch.linspace(-3, 3, 100).view(-1, 1)  # Range of Y values for PDF computation

# Compute conditional PDF P(Y | X = x_value) over y_values
conditional_pdf = compute_conditional_pdf(observed, target, x_value, y_values, bandwidth=0.5)

# Plot the result
plt.plot(y_values.cpu().numpy(), conditional_pdf.cpu().numpy(), label=f'P(Y | X = {x_value.tolist()})')
plt.xlabel('Y')
plt.ylabel('P(Y | X = x)')
plt.title('Conditional PDF P(Y | X = x) over Possible Values of Y')
plt.legend()
plt.show()"""

# Load DataFrame and filter columns for agent_0
df = pd.read_pickle("data/df_navigation_pomdp_discrete_actions_0.pkl")
agent0_columns = [col for col in df.columns if "agent_0" in col]
df = df.loc[:, agent0_columns]

# Specify the target column and number of samples
target_column = "agent_0_action_0"  # Replace with your actual target column name
n_samples = 1024

# Separate observed X data and Y target based on specified target column
observed_X = torch.tensor(
    df.drop(columns=[target_column]).iloc[:n_samples].values,
    dtype=torch.float32,
    device="cpu",
)
observed_Y = torch.tensor(
    df[target_column].iloc[:n_samples].values, dtype=torch.float32, device="cpu"
).view(-1, 1)

# Define a range of Y values for evaluating the conditional PDF
y_values = torch.linspace(
    observed_Y.min(), observed_Y.max(), n_samples, device="cpu"
).view(-1, 1)

# Initialize VNI object with the observed data
vni = VNI(observed_X, observed_Y, bandwidth=0.5)

# Evaluate predictor on each row in df
for n, row in df.iterrows():
    x_value = torch.tensor(
        row.drop(target_column).values, dtype=torch.float32, device="cpu"
    )  # Current X values as tensor
    y_true = torch.tensor(
        row[target_column], dtype=torch.float32, device="cpu"
    )  # True Y value

    # Compute conditional PDF P(Y | X = x_value) over y_values
    conditional_pdf = vni.compute_conditional_pdf(x_value, y_values)

    # Plot the result
    plt.plot(
        y_values.cpu().numpy(), conditional_pdf.cpu().numpy(), label="P(Y | X = x)"
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
    plt.show()
