import torch
import torch.nn as nn



class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
        self.q_weights = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
    
    def reparameterize(self, mu, log_var, weights):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return ((mu + sigma * eps) * weights).sum(dim=1, keepdim=True)

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        weights = torch.softmax(self.q_weights(x), dim=1)
        return self.reparameterize(mu, log_var, weights), mu, log_var, weights