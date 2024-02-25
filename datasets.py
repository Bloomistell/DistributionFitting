from torch.utils.data import Dataset
import torch

import matplotlib.pyplot as plt



class GaussianDataset(Dataset):
    def __init__(self, device: torch.device = torch.device('cpu')):
        """
        Initialize the GaussianDataset object.
        """
        super().__init__()
        self.device = device

    def generate_data(self, n: int, n_samples: int):
        """
        Generate data for the Gaussian distribution.

        Arguments:
         - n: the number of data points.
         - n_samples: the number of samples for each data point.
        """
        self.n = n
        self.n_samples = n_samples
        self.X = torch.rand(n, device=self.device)
        self.X = self.X.repeat(n_samples).reshape(-1, 1)
        weights = torch.tensor([0.5, 0.5], device=self.device)
        mu_1 = torch.sin(self.X * 2 * 3.14159) * 3
        mu_2 = torch.cos(self.X * 2 * 3.14159) * 3
        mus = torch.cat([mu_1, mu_2], dim=1)
        sigma_1 = torch.full((n_samples * n, 1), 0.5, device=self.device)
        sigma_2 = torch.full((n_samples * n, 1), 0.5, device=self.device)
        sigmas = torch.cat([sigma_1, sigma_2], dim=1)
        comp = torch.multinomial(weights, n_samples * n, replacement=True)
        selected_mu = mus[torch.arange(n_samples * n, device=self.device), comp]
        selected_sigma = sigmas[torch.arange(n_samples * n, device=self.device), comp]
        self.y = torch.normal(selected_mu, selected_sigma).reshape(-1, 1)

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
         - int: the total number of samples.
        """
        return self.n * self.n_samples
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get the data and target at the given index.

        Arguments:
         - idx: the index of the data and target.

        Returns:
         - tuple: the data and target at the given index.
        """
        return self.X[idx], self.y[idx]
    
    def plot(self):
        """
        Plot the data and target.
        """
        plt.scatter(self.X.cpu(), self.y.cpu())
        plt.show()