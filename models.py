import torch
import torch.nn as nn




class GMNN(nn.Module): 
    """
    Gaussian Mixture Neural Network (GMNN) class. This class is a subclass of the PyTorch nn.Module class.
    It represents a neural network that models a Gaussian Mixture Model (GMM).
    """
    def __init__(self, n_gaussians: int = 3):
        """
        Initialize the GMNN.

        Arguments:
         - n_gaussians: the number of Gaussian components in the mixture model.
        """
        super().__init__()

        # Define the neural network for the means of the Gaussian components
        self.q_mu = nn.Sequential(
            nn.Linear(1, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, n_gaussians)
        )

        # Define the neural network for the log variances of the Gaussian components
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, n_gaussians)
        )

        # Define the neural network for the weights of the Gaussian components
        self.q_weights = nn.Sequential(
            nn.Linear(1, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, n_gaussians)
        )

    def sample_from_gmm(self, mu: torch.Tensor, log_var: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Sample from the Gaussian Mixture Model (GMM).

        Arguments:
            - mu: the means of the Gaussian components.
            - log_var: the log variances of the Gaussian components.
            - weights: the weights of the Gaussian components.

        Returns:
            - A sample from the GMM.
        """
        batch_size = weights.shape[0]
        # Select components based on weights for each item in the batch
        components = torch.multinomial(weights, 1).squeeze()
        # Gather the selected means and std devs
        selected_mu = mu[torch.arange(batch_size), components]
        selected_sigma = torch.exp(0.5 * log_var[torch.arange(batch_size), components])
        # Sample from the selected components
        eps = torch.randn_like(selected_mu)
        return selected_mu + selected_sigma * eps

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the GMNN.

        Arguments:
         - x: the input tensor.

        Returns:
         - A tuple containing the sample from the GMM, the means, the log variances, and the weights.
        """
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        weights = torch.softmax(self.q_weights(x), dim=1)
        return self.sample_from_gmm(mu, log_var, weights).unsqueeze(1), mu, log_var, weights