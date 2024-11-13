# KDE - Kernel Density Estimation

## Nomenclature

- \( N \): Number of samples
- \( h \): Bandwidth parameter, controlling the smoothing effect of the KDE
- \( d \): Dimensionality of the input space
- \( K \): Kernel function

## Estimation Methods

### 1. Joint KDE Estimation for \( P(X=x, Y=y) \)

The joint probability density estimation, given observations for variables \( X \) and \( Y \), is computed as:

\[
\hat{P}(X=x, Y=y) = \frac{1}{N h^{d+1}} \sum_{i=1}^{N} K\left(\frac{\| X - x_i, Y - y_i \|}{h}\right)
\]

In this equation:
- The kernel function \( K \) evaluates the density contribution from each sample \((x_i, y_i)\) around the point \((X, Y)\).
- \( h^{d+1} \) normalizes the density estimate over the dimensions of \( X \) and \( Y \).

### 2. Marginal KDE Estimation for \( P(X=x) \)

The marginal probability density estimation for \( X \), ignoring \( Y \), is calculated as:

\[
\hat{P}(X=x) = \frac{1}{N h^d} \sum_{i=1}^N K\left(\frac{X - x_i}{h}\right)
\]

Here:
- \( K \) is applied to each \( x_i \) with respect to the target \( X \), considering only the \( d \)-dimensional space of \( X \).
- This marginal density provides a smoothed estimate of the distribution of \( X \) alone.

### 3. Conditional Probability Density Function \( P(Y|X=x) \)

The conditional probability density function of \( Y \) given \( X = x \) can be derived using the joint and marginal densities:

\[
\hat{P}(Y|X=x) = \frac{\hat{P}(X=x, Y=y)}{\hat{P}(X=x)}
\]

In this form:
- The KDE for \( P(Y|X=x) \) leverages both joint and marginal estimates to produce a conditional density for \( Y \) at given values of \( X \).
