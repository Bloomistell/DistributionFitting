import torch
import torch.nn as nn



class VI(nn.Module):
    def __init__(self, n_gaussians=3):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, n_gaussians)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, n_gaussians)
        )
        self.q_weights = nn.Sequential(
            nn.Linear(1, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, n_gaussians)
        )

    def sample_from_gmm(self, mu, log_var, weights):
        batch_size = weights.shape[0]
        # Select components based on weights for each item in the batch
        components = torch.multinomial(weights, 1).squeeze()
        # Gather the selected means and std devs
        selected_mu = mu[torch.arange(batch_size), components]
        selected_sigma = torch.exp(0.5 * log_var[torch.arange(batch_size), components])
        # Sample from the selected components
        eps = torch.randn_like(selected_mu)
        return selected_mu + selected_sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        weights = torch.softmax(self.q_weights(x), dim=1)
        return self.sample_from_gmm(mu, log_var, weights).unsqueeze(1), mu, log_var, weights