from torch.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
    Exponential,
    Gamma,
    Geometric,
    LogNormal,
    MultivariateNormal,
    Normal,
    Poisson,
    Uniform,
)

# Define string constants for each distribution
NORMAL = "normal"
MULTIVARIATE_NORMAL = "multivariate_normal"
BERNOULLI = "bernoulli"
CATEGORICAL = "categorical"
UNIFORM = "uniform"
EXPONENTIAL = "exponential"
GAMMA = "gamma"
BETA = "beta"
POISSON = "poisson"
DIRICHLET = "dirichlet"
LOGNORMAL = "lognormal"
GEOMETRIC = "geometric"
BINOMIAL = "binomial"

# Dictionary to map distribution names to classes
DISTRIBUTIONS = {
    NORMAL: Normal,
    MULTIVARIATE_NORMAL: MultivariateNormal,
    BERNOULLI: Bernoulli,
    CATEGORICAL: Categorical,
    UNIFORM: Uniform,
    EXPONENTIAL: Exponential,
    GAMMA: Gamma,
    BETA: Beta,
    POISSON: Poisson,
    DIRICHLET: Dirichlet,
    LOGNORMAL: LogNormal,
    GEOMETRIC: Geometric,
    BINOMIAL: Binomial,
}
