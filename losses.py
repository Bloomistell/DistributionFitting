import torch
import torch.nn as nn



class GaussianMixtureLoss(nn.Module):
    """
    Gaussian Mixture Loss class for handling the loss computation in a Gaussian Mixture Model.
    """
    def __init__(self, prior_mu, prior_sigma, device='cpu'):
        """
        Initialize the GaussianMixtureLoss class.

        Arguments:
            - prior_mu: the prior mean.
            - prior_sigma: the prior standard deviation.
            - device: the device to use for computations.
        """
        super(GaussianMixtureLoss, self).__init__()
        self.prior_mu = torch.tensor(prior_mu, device=device)
        self.prior_sigma = torch.tensor(prior_sigma, device=device)
        self.device = device

    def forward(self, y_pred, y, mu, log_var, weights):
        """
        Compute the forward pass of the loss.

        Arguments:
            - y_pred: the predicted output.
            - y: the actual output.
            - mu: the means of the Gaussian components.
            - log_var: the log variances of the Gaussian components.
            - weights: the weights of the Gaussian components.

        Returns:
            - The computed loss.
        """
        # likelihood of observing y given Variational mu and sigma
        likelihood = self._nll_gm(y, mu, log_var, weights)
        
        # prior probability of y_pred
        log_prior = self._nll_gaussian(y_pred, self.prior_mu, torch.log(self.prior_sigma))
        
        # variational probability of y_pred
        log_p_q = self._nll_gm(y_pred, mu, log_var, weights)
        
        # by taking the mean we approximate the expectation
        return (likelihood + (log_prior - log_p_q).abs()).mean()

    def _nll_gm(self, y, mu, log_var, weights):
        """
        Compute the negative log likelihood for the Gaussian Mixture.

        Arguments:
            - y: the actual output.
            - mu: the means of the Gaussian components.
            - log_var: the log variances of the Gaussian components.
            - weights: the weights of the Gaussian components.

        Returns:
            - The computed negative log likelihood.
        """
        sigma = torch.exp(0.5 * log_var)
        return -torch.log(torch.sum(weights * torch.exp(-0.5 * ((y - mu)**2 / sigma**2)) / torch.sqrt(2 * torch.pi * sigma**2), dim=1) + 1e-8)

    def _nll_gaussian(self, y, mu, log_var):
        """
        Compute the negative log likelihood for the Gaussian.

        Arguments:
            - y: the actual output.
            - mu: the mean of the Gaussian.
            - log_var: the log variance of the Gaussian.

        Returns:
            - The computed negative log likelihood.
        """
        sigma = torch.exp(0.5 * log_var)
        return 0.5 * torch.log(2 * torch.pi * sigma**2) + (1 / (2 * sigma**2))* (y-mu)**2
    
    def loss_eval(self, y_pred, y, mu, log_var, weights):
        """
        Evaluate the loss and print the components.

        Arguments:
            - y_pred: the predicted output.
            - y: the actual output.
            - mu: the means of the Gaussian components.
            - log_var: the log variances of the Gaussian components.
            - weights: the weights of the Gaussian components.
        """
        print('likelihood:', self._nll_gm(y, mu, log_var, weights).mean())
        print('log_prior:', self._nll_gaussian(y_pred, self.prior_mu, torch.log(self.prior_sigma)).mean())
        print('log_p_q:', self._nll_gm(y_pred, mu, log_var, weights).mean())
        print('elbo:', self.forward(y_pred, y, mu, log_var, weights).mean())



def KL_loss(y, y_pred):
    """
    Compute the Kullback-Leibler divergence for PDFs.

    Arguments:
        - y: the actual output.
        - y_pred: the predicted output.

    Returns:
        - The computed Kullback-Leibler divergence.
    """
    return torch.mean(y * (torch.log(y + 1e-8) - torch.log(y_pred + 1e-8)))


def JS_loss(y, y_pred):
    """
    Compute the Jensen-Shannon divergence for PDFs.

    Arguments:
        - y: the actual output.
        - y_pred: the predicted output.

    Returns:
        - The computed Jensen-Shannon divergence.
    """
    m = 0.5 * (y + y_pred)
    return 0.5 * KL_loss(y, m) + 0.5 * KL_loss(y_pred, m)