import torch

DISTRIBUTIONS_MAP = {
    "Normal": torch.distributions.Normal,
    "Bernoulli": torch.distributions.Bernoulli,
    "Binomial": torch.distributions.Binomial,
    "Beta": torch.distributions.Beta,
    "Categorical": torch.distributions.Categorical,
    "Chi2": torch.distributions.Chi2,
    "Dirichlet": torch.distributions.Dirichlet,
    "Exponential": torch.distributions.Exponential,
    "Gamma": torch.distributions.Gamma,
    "Geometric": torch.distributions.Geometric,
    "Gumbel": torch.distributions.Gumbel,
    "Independent": torch.distributions.Independent,
    "Kumaraswamy": torch.distributions.Kumaraswamy,
    "Laplace": torch.distributions.Laplace,
    "LogNormal": torch.distributions.LogNormal,
    "Multinomial": torch.distributions.Multinomial,
    "MultivariateNormal": torch.distributions.MultivariateNormal,
    "NegativeBinomial": torch.distributions.NegativeBinomial,
    "Poisson": torch.distributions.Poisson,
    "RelaxedBernoulli": torch.distributions.RelaxedBernoulli,
    "RelaxedOneHotCategorical": torch.distributions.RelaxedOneHotCategorical,
    "StudentT": torch.distributions.StudentT,
    "Uniform": torch.distributions.Uniform,
    "Weibull": torch.distributions.Weibull,
    "Cauchy": torch.distributions.Cauchy,
    "HalfCauchy": torch.distributions.HalfCauchy,
    "HalfNormal": torch.distributions.HalfNormal,
    "NormalInverseGaussian": (
        torch.distributions.NormalInverseGaussian
        if hasattr(torch.distributions, "NormalInverseGaussian")
        else None
    ),  # Check if available
    "Pareto": torch.distributions.Pareto,
    "LogisticNormal": (
        torch.distributions.LogisticNormal
        if hasattr(torch.distributions, "LogisticNormal")
        else None
    ),  # Check if available
}
