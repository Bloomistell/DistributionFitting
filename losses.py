import torch
import torch.nn as nn



def nll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return 0.5 * torch.log(2 * torch.pi * sigma**2) + (1 / (2 * sigma**2))* (y-mu)**2

def elbo(y_pred, y, mu, log_var):
    # likelihood of observing y given Variational mu and sigma
    likelihood = nll_gaussian(y, mu, log_var)
    
    # prior probability of y_pred
    log_prior = nll_gaussian(y_pred, 2.5, torch.log(torch.tensor(1.4)))
    
    # variational probability of y_pred
    log_p_q = nll_gaussian(y_pred, mu, log_var)
    
    # by taking the mean we approximate the expectation
    return (likelihood + log_prior - log_p_q).mean()

def det_loss(y_pred, y, mu, log_var):
    return elbo(y_pred, y, mu, log_var)



def nll_gm(y, mu, log_var, weights):
    sigma = torch.exp(0.5 * log_var)
    return -torch.log(torch.sum(weights * torch.exp(-0.5 * ((y - mu)**2 / sigma**2)) / torch.sqrt(2 * torch.pi * sigma**2), dim=1) + 1e-8)

def elbo_gm(y_pred, y, mu, log_var, weights):
    # likelihood of observing y given Variational mu and sigma
    likelihood = nll_gm(y, mu, log_var, weights)
    
    # prior probability of y_pred
    log_prior = nll_gaussian(y_pred, -0.5, torch.log(torch.tensor(0.9)))
    
    # variational probability of y_pred
    log_p_q = nll_gm(y_pred, mu, log_var, weights)
    
    # by taking the mean we approximate the expectation
    return (likelihood + (log_prior - log_p_q).abs()).mean()

def elbo_gm_eval(y_pred, y, mu, log_var, weights):
        # sigma = torch.exp(0.5 * log_var)
        # print((torch.sum(weights * torch.exp(-0.5 * ((y - mu)**2 / sigma**2)) / torch.sqrt(2 * torch.pi * sigma**2), dim=1) + 1e-8).mean())
        print('likelihood:', nll_gm(y, mu, log_var, weights).mean())
        print('log_prior:', nll_gaussian(y_pred, 2.5, torch.log(torch.tensor(1.75))).mean())
        print('log_p_q:', nll_gm(y_pred, mu, log_var, weights).mean())
        print('elbo:', elbo_gm(y_pred, y, mu, log_var, weights).mean())



class GaussianMixtureLoss(nn.Module):
    def __init__(self, prior_mu, prior_sigma, device='cpu'):
        super(GaussianMixtureLoss, self).__init__()
        self.prior_mu = torch.tensor(prior_mu, device=device)
        self.prior_sigma = torch.tensor(prior_sigma, device=device)
        self.device = device

    def forward(self, y_pred, y, mu, log_var, weights):
        # likelihood of observing y given Variational mu and sigma
        likelihood = self._nll_gm(y, mu, log_var, weights)
        
        # prior probability of y_pred
        log_prior = self._nll_gaussian(y_pred, self.prior_mu, torch.log(self.prior_sigma))
        
        # variational probability of y_pred
        log_p_q = self._nll_gm(y_pred, mu, log_var, weights)
        
        # by taking the mean we approximate the expectation
        return (likelihood + (log_prior - log_p_q).abs()).mean()

    def _nll_gm(self, y, mu, log_var, weights):
        sigma = torch.exp(0.5 * log_var)
        return -torch.log(torch.sum(weights * torch.exp(-0.5 * ((y - mu)**2 / sigma**2)) / torch.sqrt(2 * torch.pi * sigma**2), dim=1) + 1e-8)

    def _nll_gaussian(self, y, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        return 0.5 * torch.log(2 * torch.pi * sigma**2) + (1 / (2 * sigma**2))* (y-mu)**2
    
    def loss_eval(self, y_pred, y, mu, log_var, weights):
        print('likelihood:', self._nll_gm(y, mu, log_var, weights).mean())
        print('log_prior:', self._nll_gaussian(y_pred, self.prior_mu, torch.log(self.prior_sigma)).mean())
        print('log_p_q:', self._nll_gm(y_pred, mu, log_var, weights).mean())
        print('elbo:', self.forward(y_pred, y, mu, log_var, weights).mean())