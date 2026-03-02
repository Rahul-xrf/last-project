"""
Module 2: Deep Recurrent Q-Network (DRQN) Model
Implements the neural network architecture with LSTM layers
for handling temporal dependencies in spectrum allocation decisions.
Uses agent ID conditioning for multi-agent parameter sharing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_SIZE, LSTM_LAYERS, OBS_DIM, ACTION_DIM, NUM_VEHICLES, DEVICE


class DRQNNetwork(nn.Module):
    """
    Deep Recurrent Q-Network (DRQN) with LSTM and Agent ID conditioning.

    Architecture:
        Observation → Linear(obs_dim, hidden) → ReLU
        Agent ID → Embedding(num_agents, embed_dim)
        [obs_features, agent_embed] → Linear → ReLU
        → LSTM(hidden, hidden, num_layers)
        → Linear(hidden, hidden) → ReLU
        → Linear(hidden, action_dim) → Q-values

    The agent ID embedding allows parameter-shared agents to learn
    different channel preferences based on their identity.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 hidden_size=HIDDEN_SIZE, lstm_layers=LSTM_LAYERS,
                 n_agents=NUM_VEHICLES):
        super(DRQNNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.n_agents = n_agents

        # Agent ID embedding (key for multi-agent differentiation)
        embed_dim = 32
        self.agent_embedding = nn.Embedding(n_agents, embed_dim)

        # Input embedding layer (obs + agent embedding)
        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim + embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # LSTM layer for temporal dependency modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Output layers: fully connected to Q-values
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, obs, agent_ids, hidden_state=None):
        """
        Forward pass through the network.

        Args:
            obs: tensor of shape (batch, seq_len, obs_dim) or (batch, obs_dim)
            agent_ids: tensor of shape (batch,) with agent IDs
            hidden_state: tuple (h, c) for LSTM, or None for initial state

        Returns:
            q_values: tensor of shape (batch, seq_len, action_dim)
            hidden_state: updated (h, c) tuple
        """
        # Handle single observation (no sequence dimension)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)

        batch_size = obs.size(0)
        seq_len = obs.size(1)

        # Get agent embeddings and expand for sequence
        agent_embed = self.agent_embedding(agent_ids)  # (batch, embed_dim)
        agent_embed = agent_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, embed_dim)

        # Concatenate observation with agent embedding
        x = torch.cat([obs, agent_embed], dim=-1)  # (batch, seq, obs_dim + embed_dim)

        # Process through input embedding
        x = x.reshape(batch_size * seq_len, -1)
        x = self.input_layer(x)
        x = x.reshape(batch_size, seq_len, self.hidden_size)

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, hidden_state)

        # Process through output layers
        out = lstm_out.reshape(batch_size * seq_len, self.hidden_size)
        q_values = self.output_layer(out)
        q_values = q_values.reshape(batch_size, seq_len, -1)

        return q_values, new_hidden

    def init_hidden(self, batch_size=1):
        """Initialize LSTM hidden state with zeros."""
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(DEVICE)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(DEVICE)
        return (h, c)

    def get_action_values(self, obs, agent_id, hidden_state=None):
        """
        Get Q-values for a single observation (no sequence dim).
        Used during execution/inference.

        Args:
            obs: tensor of shape (obs_dim,) or (1, obs_dim)
            agent_id: int, the agent's ID
            hidden_state: optional (h, c) tuple

        Returns:
            q_values: tensor of shape (action_dim,)
            hidden_state: updated (h, c) tuple
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (1, obs_dim)

        agent_ids = torch.LongTensor([agent_id]).to(DEVICE)
        q_values, new_hidden = self.forward(obs, agent_ids, hidden_state)
        return q_values.squeeze(0).squeeze(0), new_hidden
