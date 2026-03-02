"""
Module 2: Intelligent Agent Design
Implements the DRQN-based vehicle agent with experience replay,
epsilon-greedy exploration, and target network for stable learning.
Each agent has its own ID for differentiated behavior via parameter sharing.
"""

import numpy as np
import random
from collections import deque
import torch
import torch.optim as optim
import torch.nn.functional as F

from drqn_model import DRQNNetwork
from config import (
    DEVICE, OBS_DIM, ACTION_DIM, HIDDEN_SIZE, LSTM_LAYERS,
    LEARNING_RATE, GAMMA, TAU, SEQUENCE_LENGTH, NUM_VEHICLES,
    REPLAY_BUFFER_SIZE, BATCH_SIZE,
    EPSILON_START, EPSILON_END, EPSILON_DECAY
)


class EpisodeMemory:
    """Stores a single episode's transitions for sequential replay."""

    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []

    def add(self, obs, action, reward, next_obs, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)

    def __len__(self):
        return len(self.observations)


class ReplayBuffer:
    """
    Experience replay buffer that stores complete episodes.
    Samples sequences of fixed length for LSTM training.
    """

    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, episode_memory):
        """Add a complete episode to the buffer."""
        if len(episode_memory) >= SEQUENCE_LENGTH:
            self.buffer.append(episode_memory)

    def sample(self, batch_size=BATCH_SIZE):
        """
        Sample a batch of sequences from stored episodes.

        Returns:
            obs_batch: (batch, seq_len, obs_dim)
            action_batch: (batch, seq_len)
            reward_batch: (batch, seq_len)
            next_obs_batch: (batch, seq_len, obs_dim)
            done_batch: (batch, seq_len)
            agent_id_batch: (batch,)
        """
        episodes = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        agent_id_batch = []

        for episode in episodes:
            ep_len = len(episode)
            start = random.randint(0, ep_len - SEQUENCE_LENGTH)
            end = start + SEQUENCE_LENGTH

            obs_batch.append(episode.observations[start:end])
            action_batch.append(episode.actions[start:end])
            reward_batch.append(episode.rewards[start:end])
            next_obs_batch.append(episode.next_observations[start:end])
            done_batch.append(episode.dones[start:end])
            agent_id_batch.append(episode.agent_id)

        obs_batch = torch.FloatTensor(np.array(obs_batch)).to(DEVICE)
        action_batch = torch.LongTensor(np.array(action_batch)).to(DEVICE)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(DEVICE)
        next_obs_batch = torch.FloatTensor(np.array(next_obs_batch)).to(DEVICE)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(DEVICE)
        agent_id_batch = torch.LongTensor(np.array(agent_id_batch)).to(DEVICE)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, agent_id_batch

    def __len__(self):
        return len(self.buffer)

    def is_ready(self):
        """Check if buffer has enough episodes for training."""
        return len(self.buffer) >= BATCH_SIZE


class DRQNAgent:
    """
    Intelligent vehicle agent using Deep Recurrent Q-Network.
    Implements epsilon-greedy exploration with target network updates.
    Uses agent ID for differentiated behavior in parameter-shared setting.
    """

    def __init__(self, agent_id, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Policy network (for action selection)
        self.policy_net = DRQNNetwork(obs_dim, action_dim).to(DEVICE)

        # Target network (for stable Q-value computation)
        self.target_net = DRQNNetwork(obs_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Exploration parameters
        self.epsilon = EPSILON_START

        # Hidden state for sequential decision making (per agent)
        self.hidden_states = {}
        for i in range(NUM_VEHICLES):
            self.hidden_states[i] = None

    def reset_hidden(self):
        """Reset LSTM hidden state for all agents at the start of each episode."""
        for i in range(NUM_VEHICLES):
            self.hidden_states[i] = self.policy_net.init_hidden(batch_size=1)

    def select_action(self, observation, agent_id=None, evaluate=False):
        """
        Select an action using epsilon-greedy policy.

        Args:
            observation: numpy array of shape (obs_dim,)
            agent_id: which agent is selecting (for hidden state tracking)
            evaluate: if True, use greedy policy (no exploration)

        Returns:
            action: int, selected channel index
        """
        if agent_id is None:
            agent_id = self.agent_id

        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).to(DEVICE)
            hidden = self.hidden_states.get(agent_id)
            q_values, new_hidden = self.policy_net.get_action_values(
                obs_tensor, agent_id, hidden
            )
            self.hidden_states[agent_id] = new_hidden
            return q_values.argmax().item()

    def update(self, replay_buffer):
        """
        Update DRQN weights using a batch of sequences from replay buffer.

        Returns:
            loss: float, training loss value
        """
        if not replay_buffer.is_ready():
            return 0.0

        obs, actions, rewards, next_obs, dones, agent_ids = replay_buffer.sample()

        # Compute Q-values for current observations
        q_values, _ = self.policy_net(obs, agent_ids)
        q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_obs, agent_ids)
            max_next_q = next_q_values.max(dim=2)[0]
            targets = rewards + GAMMA * max_next_q * (1 - dones)

        # Compute loss and update
        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def soft_update_target(self):
        """Soft update target network: θ_target = τ*θ_policy + (1-τ)*θ_target"""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                TAU * policy_param.data + (1.0 - TAU) * target_param.data
            )

    def hard_update_target(self):
        """Hard update: copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
