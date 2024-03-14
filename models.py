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
    


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(input_dim, latent_dim*2)  # *2 for mean and log variance

    def forward(self, x):
        x = self.linear(x)
        mu = x[:, :self.latent_dim]  # Mean
        log_var = x[:, self.latent_dim:]  # Log variance for numerical stability
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return torch.sigmoid(self.linear(z))  # Use sigmoid to output values between 0 and 1

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, 1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z)



class GaussianTransform(nn.Module):
    """
    A family of Gaussian are parametrized by the input tensor and then transformed by a neural network.

    Hopefully, the complexe transformation will enable the model to capture complexe distribution.
    """
    def __init__(
            self, n_input: int,
            encoder_n_hidden: list,
            n_latent: int,
            decoder_n_hidden: list,
            n_output: int = 1
        ):
        """
        Initialize the GaussianTransform.

        Arguments:
         - n_input: the number of input features.
         - encoder_layers: the list of number of neurones in the layers for the encoder.
         - decoder_layers: the list of number of neurones in the layers for the decoder.
         - n_output: the number of output features.
        """
        super().__init__()
        self.n_latent = n_latent

        self.mu_encoder = nn.Sequential(
            nn.Linear(n_input, encoder_n_hidden[0]),
            nn.ReLU()
        )
        self.log_var_encoder = nn.Sequential(
            nn.Linear(n_input, encoder_n_hidden[0]),
            nn.ReLU()
        )
        for i in encoder_n_hidden[1:]:
            self.mu_encoder.add_module('hidden', nn.Linear(encoder_n_hidden[i-1], encoder_n_hidden[i]))
            self.mu_encoder.add_module('hidden_relu', nn.ReLU())
            self.log_var_encoder.add_module('hidden', nn.Linear(encoder_n_hidden[i-1], encoder_n_hidden[i]))
            self.log_var_encoder.add_module('hidden_relu', nn.ReLU())

        self.mu_encoder.add_module('output', nn.Linear(encoder_n_hidden[-1], n_latent))
        self.log_var_encoder.add_module('output', nn.Linear(encoder_n_hidden[-1], n_latent))

        self.decoder = nn.Sequential(
            nn.Linear(n_latent, decoder_n_hidden[0]),
            nn.ReLU()
        )
        for i in decoder_n_hidden[1:]:
            self.decoder.add_module('hidden', nn.Linear(decoder_n_hidden[i-1], decoder_n_hidden[i]))
            self.decoder.add_module('hidden_relu', nn.ReLU())
        
        self.decoder.add_module('output', nn.Linear(decoder_n_hidden[-1], n_output))

    def reparametrize(self, mu, log_var):
        """
        Reparametrize the Gaussian distribution.

        Arguments:
         - mu: the mean of the Gaussian distribution.
         - log_var: the log variance of the Gaussian distribution.

        Returns:
         - A sample from the Gaussian distribution.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GaussianTransform.

        Arguments:
         - x: the input tensor.

        Returns:
         - The output tensor.
        """
        mu = self.mu_encoder(x)
        log_var = self.log_var_encoder(x)
        z = self.reparametrize(mu, log_var)
        return self.decoder(z)