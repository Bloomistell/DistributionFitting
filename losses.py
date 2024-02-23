import torch



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
    