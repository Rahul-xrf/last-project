"""
Configuration for Multi-Agent Reinforcement Learning Dynamic Spectrum Allocation.
All hyperparameters and simulation settings are centralized here.
"""

import torch

# =============================================================================
# Device Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Environment Settings
# =============================================================================
GRID_SIZE = 10                # Urban grid dimension (10x10 intersections)
NUM_VEHICLES = 10             # Number of vehicle agents
NUM_CHANNELS = 5              # Available wireless channels
MAX_EPISODE_STEPS = 200       # Maximum steps per episode
VEHICLE_SPEED_RANGE = (1, 3)  # Min/max speed (grid units per step)
PAYLOAD_SIZE_RANGE = (1, 5)   # Min/max payload size (packets)
TIME_BUDGET_RANGE = (5, 20)   # Min/max time budget for delivery

# Channel model parameters
NOISE_POWER = 0.1             # Background noise power
INTERFERENCE_FACTOR = 0.5     # Interference scaling factor
SNR_THRESHOLD = 5.0           # Minimum SNR for successful transmission
TRANSMISSION_POWER = 1.0      # Default transmission power
PATH_LOSS_EXPONENT = 3.0      # Path loss exponent for urban environment

# =============================================================================
# DRQN Model Settings
# =============================================================================
HIDDEN_SIZE = 64              # Hidden layer size
LSTM_LAYERS = 1               # Number of LSTM layers
SEQUENCE_LENGTH = 8           # Sequence length for LSTM training
LEARNING_RATE = 1e-3          # Adam optimizer learning rate
GAMMA = 0.95                  # Discount factor
TAU = 0.01                    # Soft update rate for target network

# =============================================================================
# Training Settings
# =============================================================================
NUM_EPISODES = 500            # Total training episodes
BATCH_SIZE = 32               # Batch size for replay sampling
REPLAY_BUFFER_SIZE = 5000     # Maximum replay buffer capacity
EPSILON_START = 1.0           # Starting exploration rate
EPSILON_END = 0.05            # Minimum exploration rate
EPSILON_DECAY = 0.995         # Epsilon decay factor per episode
TARGET_UPDATE_FREQ = 10       # Episodes between target network updates
SAVE_MODEL_FREQ = 50          # Episodes between model checkpoints

# =============================================================================
# Reward Settings
# =============================================================================
REWARD_SUCCESS = 1.0          # Reward for successful packet delivery
REWARD_COLLISION = -1.0       # Penalty for channel collision
REWARD_IDLE = 0.0             # Reward when no action needed

# =============================================================================
# SUMO Simulator Settings
# =============================================================================
USE_SUMO = True               # Toggle SUMO integration (True=SUMO, False=standalone)
SUMO_CONFIG = "sumo_config/urban.sumocfg"  # Path to SUMO config file
SUMO_GUI = True              # Use sumo-gui instead of sumo (for visualization)
SUMO_STEP_LENGTH = 1.0        # SUMO simulation step length in seconds

# =============================================================================
# Evaluation Settings
# =============================================================================
EVAL_EPISODES = 100           # Episodes for evaluation
RESULTS_DIR = "results"       # Directory for saving results
MODEL_DIR = "models"          # Directory for saving trained models

# =============================================================================
# Observation & Action Dimensions (computed)
# =============================================================================
# Observation: channel_occupancy(NUM_CHANNELS) + interference(NUM_CHANNELS)
#            + position(2) + speed(1) + remaining_payload(1) + time_budget(1)
#            + previous_action_onehot(NUM_CHANNELS) + agent_id_onehot(NUM_VEHICLES)
OBS_DIM = NUM_CHANNELS * 2 + 5 + NUM_CHANNELS + NUM_VEHICLES
# Action: select one of NUM_CHANNELS
ACTION_DIM = NUM_CHANNELS
