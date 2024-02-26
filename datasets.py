from torch.utils.data import Dataset
import torch

import numpy as np

import matplotlib.pyplot as plt



class GaussianDataset(Dataset):
    def __init__(self, mu_transforms: list, sigmas: list, weights: list, poly: int=0, device: torch.device = 'cpu'):
        """
        Initialize the GaussianDataset object.

        Arguments:
         - mu_transforms: the list of functions to transform the input data.
         - sigmas: the list of standard deviations for the Gaussian components.
         - weights: the list of weights for the Gaussian components.
         - poly: the degree of the polynomial features to add to the input data (0 for no polynomial features).
         - device: the device to use for computations.
        """
        super().__init__()
        self.device = device
        self.mu_transforms = mu_transforms
        self.sigmas = sigmas
        self.weights = weights
        self.poly = poly

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
        self.X = self.X.repeat(n_samples).reshape(-1, 1).to(self.device)
        
        self.X_gm_transform(self.X, n, n_samples)

        if self.poly:
            self.X = self.X_poly_transform(self.X, n, n_samples)

    def X_gm_transform(self, X: torch.tensor, n: int, n_samples: int) -> torch.tensor:
        """
        Transform the input data.

        Arguments:
            - X: the input data.
            - n: the number of data points.
            - n_samples: the number of samples for each data point.

        Returns:
            - y: the transformed input data.
        """
        mus, sigmas = [], []
        for i in range(len(self.mu_transforms)):
            mus.append(self.mu_transforms[i](X))
            sigmas.append(torch.full((n_samples * n, 1), self.sigmas[i]))

        mus = torch.cat(mus, dim=1)
        sigmas = torch.cat(sigmas, dim=1)

        # sample from the Gaussian Mixture
        comp = torch.multinomial(self.weights, n_samples * n, replacement=True)
        selected_mu = mus[torch.arange(n_samples * n, device=self.device), comp]
        selected_sigma = sigmas[torch.arange(n_samples * n, device=self.device), comp]

        self.y = torch.normal(selected_mu, selected_sigma).reshape(-1, 1).to(self.device)

        return self.y
    
    def X_poly_transform(self, X: torch.tensor, n: int, n_samples: int) -> torch.tensor:
        """
        Augment the data with polynomial features.

        Arguments:
         - X: the input data.
         - n: the number of data points.
         - n_samples: the number of samples for each data point.
        
        Returns:
            - X_poly: the augmented input data.
        """
        X_poly = torch.cat([X**i for i in range(1, self.poly+1)], dim=1)
        X_poly = X_poly.repeat(n_samples, 1)
        
        return X_poly
    
    def y_curve(self, X: float, y: np.array) -> np.array:
        """
        Compute the curve for the given input.

        Arguments:
         - X: point for distribution inference.
         - y: the range for the curve.

        Returns:
         - y_curve: the curve for the given input.
        """
        y_curve = np.sum([self._gaussian(y, mu(X), sigma) * weight for mu, sigma, weight in zip(self.mu_transforms, self.sigmas, self.weights)], axis=0)
        
        return y_curve
    
    def _gaussian(self, y: np.array, mu: np.array, sigma: np.array) -> np.array:
        """
        Compute the gaussian distribution.

        Arguments:
         - y: the actual output.
         - mu: the mean of the Gaussian.
         - sigma: the standard deviation of the Gaussian.

        Returns:
         - The computed gaussian distribution.
        """
        return np.exp(-0.5 * ((y - mu)**2 / sigma**2)) / np.sqrt(2 * np.pi * sigma**2)



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