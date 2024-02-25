from torch.utils.data import DataLoader, Dataset
import torch

import matplotlib.pyplot as plt




# dataset where each output is drawn from a Gaussian distribution
class GaussianDataset(Dataset):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def generate_data(self, n, n_samples):
        self.n = n
        self.n_samples = n_samples
        self.X = torch.rand(n)
        self.X = self.X.repeat(n_samples).reshape(-1, 1).to(self.device)
        weights = [0.5, 0.5]
        mu_1 = torch.sin(self.X * 2 * 3.14159) * 3
        mu_2 = torch.cos(self.X * 2 * 3.14159) * 3
        mus = torch.cat([mu_1, mu_2], dim=1)
        sigma_1 = torch.full((n_samples * n, 1), 0.5)
        sigma_2 = torch.full((n_samples * n, 1), 0.5)
        sigmas = torch.cat([sigma_1, sigma_2], dim=1)
        comp = torch.multinomial(torch.tensor(weights), n_samples * n, replacement=True)
        selected_mu = mus[torch.arange(n_samples * n), comp]
        selected_sigma = sigmas[torch.arange(n_samples * n), comp]
        self.y = torch.normal(selected_mu, selected_sigma).reshape(-1, 1).to(self.device)

    def __len__(self):
        return self.n * self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def plot(self):
        plt.scatter(self.X.cpu(), self.y.cpu())
        plt.show()