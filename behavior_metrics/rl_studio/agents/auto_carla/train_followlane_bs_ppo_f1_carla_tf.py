from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn

class CustomActorCriticPolicy(ActorCriticPolicy): # No se esta usando ahora mismo!!!!
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

        # Define a custom flatten extractor
        #self.features_extractor = FlattenExtractor(self.observation_space)

        # Ensure the feature dimension is correctly calculated
        # self.feature_dim = int(torch.prod(torch.tensor(self.observation_space.shape)))

        # Optional: If feature_dim is not matching, add a linear layer to match
        # self.feature_transform = nn.Sequential(
        #    nn.Linear(self.feature_dim, 256),  # Transform to the correct size
        #    nn.ReLU()
        # )

        # Policy network
        self.action_net = nn.Sequential(
            #nn.Linear(256, 256),  # Hidden layer
            #nn.ReLU(),  # Intermediate activation
            #nn.Linear(256, 256),  # Hidden layer
            #nn.ReLU(),  # Intermediate activation
            #nn.Linear(256, 256),  # Hidden layer
            #nn.ReLU(),  # Intermediate activation
            #nn.Linear(256, self.action_space.shape[0]),  # Output layer
            self.action_net,
            nn.Tanh()  # Tanh activation for output layer
        )

        # Value network
        # self.value_net = nn.Sequential(
        #     nn.Linear(256, 256),  # Hidden layer for value function
        #     nn.ReLU(),  # Intermediate activation
        #     nn.Linear(256, 256),  # Hidden layer
        #     nn.ReLU(),  # Intermediate activation
        #     nn.Linear(256, 256),  # Hidden layer
        #     nn.ReLU(),  # Intermediate activation
        #     nn.Linear(256, 1)  # Output for value function
        # )

    # def calculate_log_probs(self, action_logits):
    #     # Assuming you have a mean and stddev for your actions
    #     # Use a normal distribution to compute log probabilities
    #     mean = action_logits  # or some transformation based on your model
    #     stddev = torch.exp(self.log_std)  # log_std should be a learnable parameter
    #
    #     # Create a normal distribution
    #     dist = torch.distributions.Normal(mean, stddev)
    #
    #     # Calculate log probabilities
    #     log_probs = dist.log_prob(action_logits)
    #     return log_probs.sum(dim=-1)

    # def forward(self, obs):
    #     features = self.features_extractor(obs)  # Extract features
    #     features_transformed = self.feature_transform(features)  # Transform features to expected size
    #     action_logits = self.action_net(features_transformed)  # Use the modified action_net
    #     value = self.value_net(features_transformed)  # Get the value using the transformed features
    #     log_probs = self.calculate_log_probs(action_logits)  # Implement this method based on your action space
    #     return action_logits, value, log_probs
