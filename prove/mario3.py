import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# from mpl_toolkits.mplot3d import Axes3D


def estimate_conditional_pdf(
    intervention_samples,
    observation_samples,
    target_samples,
    intervention_range,
    observation_value,
    target_range,
    bandwidth=0.1,
    tol=0.1,
):
    """
    Estimate the conditional PDF of the target feature given interventions and observations using KDE.

    Args:
        intervention_samples (torch.Tensor): Tensor of intervention values (expanded to match target_samples).
        observation_samples (torch.Tensor): Tensor of observed values (expanded to match target_samples).
        target_samples (torch.Tensor): Tensor of resulting target values.
        intervention_range (torch.Tensor): Range of intervention values for PDF estimation.
        observation_value (float): Fixed observation value to condition on.
        target_range (torch.Tensor): Range of target values for PDF estimation.
        bandwidth (float): Bandwidth for KDE, controlling the smoothness.
        tol (float): Tolerance range for matching intervention and observation values.

    Returns:
        np.ndarray: 2D array of conditional PDF values with shape (len(target_range), len(intervention_range)).
    """
    # Convert to numpy arrays for KDE
    intervention_samples = intervention_samples.cpu().numpy()
    observation_samples = observation_samples.cpu().numpy()
    target_samples = target_samples.cpu().numpy()
    target_range = target_range.cpu().numpy()
    intervention_range = intervention_range.cpu().numpy()

    # Prepare matrix to hold conditional PDF values
    conditional_pdf_matrix = np.zeros((len(target_range), len(intervention_range)))

    # Estimate PDF for each intervention value in the range
    for i, intervention_value in enumerate(intervention_range):
        # Select target samples within tolerance of the intervention and observation values
        selected_targets = target_samples[
            (np.abs(intervention_samples - intervention_value) < tol)
            & (np.abs(observation_samples - observation_value) < tol)
        ]

        if len(selected_targets) > 1:  # Check to ensure enough samples for KDE
            kde = gaussian_kde(selected_targets, bw_method=bandwidth)
            conditional_pdf_matrix[:, i] = kde.evaluate(target_range)
        else:
            # If not enough samples, set to zeros (or some minimum density)
            conditional_pdf_matrix[:, i] = np.zeros(len(target_range))

    return conditional_pdf_matrix


def plot_surface(intervention_values, target_values, conditional_pdf):
    """
    Plot a 3D surface plot of CPD values across intervention and target values.
    """
    mesh_intervention, mesh_target = np.meshgrid(
        intervention_values.cpu().numpy(), target_values.cpu().numpy()
    )

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        mesh_intervention,
        mesh_target,
        conditional_pdf,
        cmap="viridis",
        edgecolor="none",
        alpha=0.7,
    )
    ax.set_xlabel("Intervention Feature Value")
    ax.set_ylabel("Target Feature Value")
    ax.set_zlabel("CPD Value")
    ax.set_title("3D Surface Plot of Conditional PDF")
    plt.show()


# Example usage
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define synthetic ranges and samples
n_samples = 1000
intervention_samples = torch.linspace(-2.0, 2.0, n_samples, device=device)
observation_samples = torch.rand(
    n_samples, device=device
)  # Example fixed observation values
target_samples = (
    torch.randn(n_samples, device=device) * 0.5 + intervention_samples
)  # Random target values

# Define ranges for intervention and target values for KDE evaluation
intervention_range = torch.linspace(-2.0, 2.0, 20, device=device)
target_range = torch.linspace(-3, 3, 100, device=device)

# Set a fixed observation value for conditioning
fixed_observation_value = 0.5

# Estimate conditional PDF given the observation and intervention
conditional_pdf = estimate_conditional_pdf(
    intervention_samples,
    observation_samples,
    target_samples,
    intervention_range,
    fixed_observation_value,
    target_range,
    bandwidth=0.5,
)

# Plot the conditional PDF as a 3D surface
plot_surface(intervention_range, target_range, conditional_pdf)
