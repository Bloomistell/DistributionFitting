# Distribution Fitting with Deep Neural Networks and Variational Inference

This project is focused on fitting complex data distributions (P(Y|X, D)) using a deep neural network. The network is designed to output a Gaussian mixture, and we employ the variational inference technique for training.

## Project Overview

The goal of this project is to accurately model complex data distributions. By using a deep neural network that outputs a Gaussian mixture, we can capture the intricate patterns and relationships within the data. The variational inference technique allows us to train the network in a probabilistic manner, providing a robust and flexible model.

## Methodology

1. **Deep Neural Networks**: We use deep learning to capture the non-linear relationships within the data. The network is designed to output a Gaussian mixture, allowing us to model a wide range of data distributions.

2. **Gaussian Mixture Models**: The output of the network is a Gaussian mixture, which can model complex and multi-modal data distributions.

3. **Variational Inference**: We use variational inference for training, which allows us to optimize the network in a probabilistic manner. This provides a robust and flexible model that can adapt to various data distributions.

## Math

### Bayes' Theorem

The goal is to predict the distribution of a variable y given an input x and the dataset D, P(y|x, D). We call this distribution the posterior distribution. Given a model with a set of parameters $\theta$, the bayesian formula can be written as:

$$P(y|x, D) = \int P(y|x, \theta) P(\theta|D) d\theta$$

Here:
- $P(y|x, \theta)$ is the likelihood of observing $y$ given $x$ and model parameters $\theta$. It describes how likely the output $y$ is for a given input $x$ and parameters $\theta$.
- $P(\theta|D)$ is the posterior distribution of the parameters $\theta$ given the dataset $D$. It represents our updated belief about the parameters after observing the data.
- The integral $\int P(y|x, \theta) P(\theta|D) d\theta$ aggregates over all possible parameter configurations, weighted by their posterior probability, to give the overall probability of $y$ given $x$ and the observed data $D$.

In variational inference, direct computation of $P(\theta|D)$ and hence $P(y|x, D)$ is often intractable due to the complex integral over $\theta$. Variational inference approaches this problem by approximating the true posterior $P(\theta|D)$ with a simpler, parameterized distribution $q_\theta(x)$. This approximation allows for more tractable computations. The objective is to choose $q_\theta(x)$ such that it is as close as possible to $P(\theta|D)$, typically by minimizing some form of divergence (e.g., the Kullback-Leibler divergence) between $q_\theta(x)$ and $P(\theta|D)$.

Using this approximation, the formula for $P(y|x, D)$ in the context of variational inference becomes:

$$P(y|x, D) \approx \int P(y|x, \theta) q_\theta(x) d\theta$$

This approximation allows for practical computation of the posterior predictive distribution, enabling predictions of new outputs $y$ given new inputs $x$ based on the learned model from the data $D$.

### Gaussian Mixture

In our case, the parametrized distribution will be a Gaussian Mixture:

$$q_\theta(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x | \mu_k, \Sigma_k)$$

where:
- $x$ is the data point for which you are calculating the density.
- $\pi_k$ are the mixture weights for each Gaussian component, with $\sum_{k=1}^{K} \pi_k = 1$ and $\pi_k \geq 0$ for all $k$, ensuring that the weights form a valid probability distribution.
- $\mathcal{N}(x | \mu_k, \Sigma_k)$ denotes the PDF of the $k$-th Gaussian component in the mixture, which is defined for a data point $x$ as:

In order to fit the parametrized distribution on our data, we need to evaluate the "distance" between $q_\theta(x)$ and $P(Y|X)$. To do that, we can use the Kullback-Leibler (KL) divergence:

$$D_{KL}(P \| Q) = \sum_{x \in X} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$

- The KL divergence quantifies the amount of information lost when $Q$ is used to approximate $P$. It's often described as a measure of the "distance" between two distributions, though it's not a true distance metric since it's not symmetric and does not satisfy the triangle inequality.
- In the context of variational inference, the KL divergence is used to measure how well the variational distribution $q_\theta(x)$ approximates the true posterior distribution $P(x|D)$, guiding the optimization of the variational parameters $\theta$.

### ELBO Loss

Based on the KL loss, we can create an appropriate loss for our use case; the ELBO loss:

$$\text{ELBO}(\theta) = \mathbb{E}_{q_\theta(x)}[\log P(D|x)] - \text{KL}(q_\theta(x) \| P(x))$$

where:
- $\mathbb{E}_{q_\theta(x)}[\log P(D|x)]$ is the expected log-likelihood of the data given the latent variables $x$, averaged over the variational distribution $q_\theta(x)$.
- $\text{KL}(q_\theta(x) \| P(x))$ is the Kullback-Leibler divergence between the variational distribution $q_\theta(x)$ and the prior distribution $P(x)$ over the latent variables. This term acts as a regularization component, encouraging $q_\theta(x)$ to be close to the prior $P(x)$.

### Applying the ELBO to a Gaussian Mixture Model (GMM)

When $q_\theta(x)$ is parameterized as a GMM, both terms of the ELBO involve computations specific to this choice:

1. **Expected Log-Likelihood**:
   For a GMM, this involves integrating the log-likelihood of the data under each component of the mixture, weighted by the component's mixture weight. This can be challenging to compute directly and may often involve Monte Carlo estimation for practical implementation.

2. **KL Divergence**:
   Computing the KL divergence between a GMM and a simpler prior (often chosen to be a standard Gaussian for each component for tractability) can also be complex. For certain choices of priors and specific forms of the GMM (e.g., isotropic Gaussians), analytical expressions might be derived, but in general, this may require approximation.

Given these components, the ELBO for a GMM $q_\theta(x)$ can be approximated as:

$$\text{ELBO}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \log \left(\sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(D_i | \mu_k, \Sigma_k)\right) - \text{KL}(q_\theta(x) \| P(x))$$

where $N$ is the number of data points in $D$, and $D_i$ represents the $i$-th data point.

### Practical Considerations

- **Monte Carlo Estimation**: The expected log-likelihood can be estimated using samples from the variational distribution, especially when direct integration is not feasible.
- **KL Divergence Approximation**: For complex models like GMMs, the KL divergence may not have a closed-form solution and might require numerical integration or approximation techniques, unless simplifying assumptions are made about the form of the GMM and the prior.