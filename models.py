import torch
import torch.nn as nn



class GMNN(nn.Module):
    """
    Gaussian Mixture Neural Network (GMNN) class. This class is a subclass of the PyTorch nn.Module class.
    It represents a neural network that models a Gaussian Mixture Model (GMM).
    """
    def __init__(self, n_input: int = 1, n_gaussians: int = 3, n_hidden: int = 3, n_neurons: int = 10):
        """
        Initialize the GMNN.

        Arguments:
         - n_gaussians: the number of Gaussian components in the mixture model.
         - n_hidden: the number of hidden layers in the neural networks for the means, log variances, and weights.
         - n_neurons: the number of neurons in each hidden layer of the neural networks for the means, log variances, and weights.
        """
        super().__init__()

        # Define the neural network for the means of the Gaussian components
        self.q_mu = nn.Sequential(
            nn.Linear(n_input, n_neurons),
            nn.ReLU()
        )
        for _ in range(n_hidden):
            self.q_mu.add_module('hidden', nn.Linear(n_neurons, n_neurons))
            self.q_mu.add_module('hidden_relu', nn.ReLU())

        self.q_mu.add_module('output', nn.Linear(n_neurons, n_gaussians))

        # Define the neural network for the log variances of the Gaussian components
        self.q_log_var = nn.Sequential(
            nn.Linear(n_input, n_neurons),
            nn.ReLU()
        )
        for _ in range(n_hidden):
            self.q_log_var.add_module('hidden', nn.Linear(n_neurons, n_neurons))
            self.q_log_var.add_module('hidden_relu', nn.ReLU())

        self.q_log_var.add_module('output', nn.Linear(n_neurons, n_gaussians))

        # Define the neural network for the weights of the Gaussian components
        self.q_weights = nn.Sequential(
            nn.Linear(n_input, n_neurons),
            nn.ReLU()
        )
        for _ in range(n_hidden):
            self.q_weights.add_module('hidden', nn.Linear(n_neurons, n_neurons))
            self.q_weights.add_module('hidden_relu', nn.ReLU())
        
        self.q_weights.add_module('output', nn.Linear(n_neurons, n_gaussians))

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
    


class GSM(nn.Module):
    """
    Gaussian Sampling Model (GSM) class. This class is a subclass of the PyTorch nn.Module class.

    The idea is to sample from a single Gaussian distribution centered on the input data. Hopefully, this will
    help to create very complex distributions since the simple gaussian distribution is going through a complex
    non-linear transformation.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transform = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GSM.

        Arguments:
         - x: the input tensor.

        Returns:
         - The sample from the GSM.
        """
        

    