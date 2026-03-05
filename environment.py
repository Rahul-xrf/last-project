"""
Module 1: Simulation Environment
Simulates the Cognitive Internet of Vehicles (CIoV) spectrum allocation problem
as a multi-agent environment with urban traffic, wireless channels, and interference.
"""

import numpy as np
from config import (
    GRID_SIZE, NUM_VEHICLES, NUM_CHANNELS, MAX_EPISODE_STEPS,
    VEHICLE_SPEED_RANGE, PAYLOAD_SIZE_RANGE, TIME_BUDGET_RANGE,
    NOISE_POWER, INTERFERENCE_FACTOR, SNR_THRESHOLD,
    TRANSMISSION_POWER, PATH_LOSS_EXPONENT,
    REWARD_SUCCESS, REWARD_COLLISION, REWARD_IDLE,
    OBS_DIM, ACTION_DIM
)


class Vehicle:
    """Represents an electric vehicle agent in the urban environment."""

    def __init__(self, vehicle_id, grid_size):
        self.id = vehicle_id
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Reset vehicle to random state."""
        self.position = np.array([
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ], dtype=np.float32)
        self.speed = np.random.randint(
            VEHICLE_SPEED_RANGE[0], VEHICLE_SPEED_RANGE[1] + 1
        )
        self.direction = np.random.choice([0, 1, 2, 3])  # N, E, S, W
        self.remaining_payload = np.random.randint(
            PAYLOAD_SIZE_RANGE[0], PAYLOAD_SIZE_RANGE[1] + 1
        )
        self.time_budget = np.random.randint(
            TIME_BUDGET_RANGE[0], TIME_BUDGET_RANGE[1] + 1
        )
        self.total_payload = self.remaining_payload
        self.packets_sent = 0
        self.packets_delivered = 0
        self.collisions = 0
        self.total_latency = 0.0
        self.current_step = 0
        self.previous_action = -1  # No previous action

    def move(self):
        """Move vehicle along the urban grid."""
        direction_vectors = {
            0: np.array([0, 1]),    # North
            1: np.array([1, 0]),    # East
            2: np.array([0, -1]),   # South
            3: np.array([-1, 0])    # West
        }

        # Occasionally change direction at intersections
        if np.random.random() < 0.3:
            self.direction = np.random.choice([0, 1, 2, 3])

        # Move
        delta = direction_vectors[self.direction] * self.speed
        self.position = self.position + delta

        # Wrap around grid boundaries (toroidal grid)
        self.position[0] = self.position[0] % self.grid_size
        self.position[1] = self.position[1] % self.grid_size

        # Update time budget
        self.time_budget = max(0, self.time_budget - 1)
        self.current_step += 1

        # Renew payload if delivered or timed out
        if self.remaining_payload <= 0 or self.time_budget <= 0:
            self.remaining_payload = np.random.randint(
                PAYLOAD_SIZE_RANGE[0], PAYLOAD_SIZE_RANGE[1] + 1
            )
            self.time_budget = np.random.randint(
                TIME_BUDGET_RANGE[0], TIME_BUDGET_RANGE[1] + 1
            )
            self.total_payload = self.remaining_payload

    def get_distance_to(self, other_position):
        """Calculate toroidal distance to another position."""
        diff = np.abs(self.position - other_position)
        diff = np.minimum(diff, self.grid_size - diff)
        return np.sqrt(np.sum(diff ** 2))


class WirelessChannel:
    """Models wireless channel characteristics including noise and interference."""

    def __init__(self, channel_id):
        self.id = channel_id
        self.base_noise = NOISE_POWER
        self.users = []  # Vehicles currently using this channel

    def reset(self):
        """Reset channel state."""
        self.users = []

    def calculate_interference(self, vehicle, all_vehicles):
        """
        Calculate interference at a vehicle's position on this channel.
        Interference comes from other vehicles using the same channel.
        """
        interference = 0.0
        for other_id in self.users:
            if other_id == vehicle.id:
                continue
            other_vehicle = all_vehicles[other_id]
            distance = vehicle.get_distance_to(other_vehicle.position)
            # Path loss model: interference decreases with distance
            if distance < 0.1:
                distance = 0.1  # Minimum distance to avoid division by zero
            path_loss = 1.0 / (distance ** PATH_LOSS_EXPONENT)
            interference += INTERFERENCE_FACTOR * TRANSMISSION_POWER * path_loss
        return interference

    def calculate_snr(self, vehicle, all_vehicles):
        """Calculate Signal-to-Noise-plus-Interference Ratio."""
        interference = self.calculate_interference(vehicle, all_vehicles)
        sinr = TRANSMISSION_POWER / (self.base_noise + interference)
        return sinr

    def get_num_users(self):
        """Return number of current users."""
        return len(self.users)

    def is_busy(self):
        """Check if channel has multiple users (collision potential)."""
        return len(self.users) > 1


class SpectrumEnvironment:
    """
    Multi-agent environment for dynamic spectrum allocation.
    Implements the Dec-POMDP formulation where each vehicle agent
    must independently select a wireless channel for communication.
    """

    def __init__(self, n_agents=NUM_VEHICLES, n_channels=NUM_CHANNELS,
                 grid_size=GRID_SIZE, max_steps=MAX_EPISODE_STEPS):
        self.n_agents = n_agents
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Create vehicles and channels
        self.vehicles = [Vehicle(i, grid_size) for i in range(n_agents)]
        self.channels = [WirelessChannel(i) for i in range(n_channels)]

        # State tracking
        self.current_step = 0
        self.episode_metrics = {
            'total_transmissions': 0,
            'successful_transmissions': 0,
            'collisions': 0,
            'total_latency': 0.0,
            'latency_count': 0
        }

    def reset(self):
        """
        Reset environment for a new episode.
        Returns: list of initial observations for each agent.
        """
        self.current_step = 0
        self.episode_metrics = {
            'total_transmissions': 0,
            'successful_transmissions': 0,
            'collisions': 0,
            'total_latency': 0.0,
            'latency_count': 0
        }

        # Reset all vehicles and channels
        for vehicle in self.vehicles:
            vehicle.reset()
        for channel in self.channels:
            channel.reset()

        # Return initial observations
        return [self._get_observation(i) for i in range(self.n_agents)]

    def step(self, actions):
        """
        Execute one environment step with simultaneous actions from all agents.

        Args:
            actions: list of channel selections, one per agent (int 0..n_channels-1)

        Returns:
            observations: list of new observations per agent
            rewards: list of rewards per agent
            done: bool, whether episode is finished
            info: dict with episode metrics
        """
        assert len(actions) == self.n_agents, \
            f"Expected {self.n_agents} actions, got {len(actions)}"

        # Clear channel assignments
        for channel in self.channels:
            channel.reset()

        # Assign vehicles to channels based on actions
        for agent_id, action in enumerate(actions):
            channel_id = int(action)
            self.channels[channel_id].users.append(agent_id)
            self.vehicles[agent_id].previous_action = channel_id

        # Compute rewards for each agent
        rewards = []
        for agent_id, action in enumerate(actions):
            channel_id = int(action)
            vehicle = self.vehicles[agent_id]
            channel = self.channels[channel_id]

            vehicle.packets_sent += 1
            self.episode_metrics['total_transmissions'] += 1

            num_users_on_channel = channel.get_num_users()

            if num_users_on_channel == 1:
                # Clear channel - successful transmission
                reward = REWARD_SUCCESS
                vehicle.packets_delivered += 1
                vehicle.remaining_payload -= 1
                self.episode_metrics['successful_transmissions'] += 1
                latency = vehicle.current_step / max(1, vehicle.total_payload)
                vehicle.total_latency += latency
                self.episode_metrics['total_latency'] += latency
                self.episode_metrics['latency_count'] += 1
            else:
                # Collision - multiple vehicles on same channel
                snr = channel.calculate_snr(vehicle, self.vehicles)
                if snr >= SNR_THRESHOLD:
                    # Signal strong enough despite interference
                    reward = REWARD_SUCCESS * 0.5
                    vehicle.packets_delivered += 1
                    vehicle.remaining_payload -= 1
                    self.episode_metrics['successful_transmissions'] += 1
                    latency = vehicle.current_step / max(1, vehicle.total_payload)
                    vehicle.total_latency += latency
                    self.episode_metrics['total_latency'] += latency
                    self.episode_metrics['latency_count'] += 1
                else:
                    # Collision - signal too weak
                    reward = REWARD_COLLISION
                    vehicle.collisions += 1
                    self.episode_metrics['collisions'] += 1

            rewards.append(reward)

        # Move vehicles
        for vehicle in self.vehicles:
            vehicle.move()

        # Increment step counter
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Get new observations
        observations = [self._get_observation(i) for i in range(self.n_agents)]

        # Compile info
        info = self._get_metrics()

        return observations, rewards, done, info

    def _get_observation(self, agent_id):
        """
        Construct local observation for a specific agent.
        Observation vector:
          - Channel occupancy (NUM_CHANNELS): number of users normalized
          - Interference levels (NUM_CHANNELS): estimated interference on each channel
          - Position (2): x, y normalized
          - Speed (1): normalized
          - Remaining payload (1): normalized
          - Time budget (1): normalized
          - Previous action one-hot (NUM_CHANNELS): last channel selected
          - Agent ID one-hot (NUM_VEHICLES): unique agent identity
        """
        vehicle = self.vehicles[agent_id]
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Channel occupancy - deterministic count (normalized)
        for ch_id, channel in enumerate(self.channels):
            obs[ch_id] = channel.get_num_users() / max(1, self.n_agents)

        # Interference levels on each channel
        for ch_id, channel in enumerate(self.channels):
            interference = channel.calculate_interference(vehicle, self.vehicles)
            # Normalize interference to [0, 1] range
            obs[NUM_CHANNELS + ch_id] = min(1.0, interference / 2.0)

        # Normalized position
        idx = NUM_CHANNELS * 2
        obs[idx] = vehicle.position[0] / self.grid_size
        obs[idx + 1] = vehicle.position[1] / self.grid_size

        # Normalized speed
        obs[idx + 2] = (vehicle.speed - VEHICLE_SPEED_RANGE[0]) / \
                        max(1, VEHICLE_SPEED_RANGE[1] - VEHICLE_SPEED_RANGE[0])

        # Normalized remaining payload
        obs[idx + 3] = vehicle.remaining_payload / PAYLOAD_SIZE_RANGE[1]

        # Normalized time budget
        obs[idx + 4] = vehicle.time_budget / TIME_BUDGET_RANGE[1]

        # Previous action one-hot encoding
        idx2 = idx + 5
        if vehicle.previous_action >= 0:
            obs[idx2 + vehicle.previous_action] = 1.0

        # Agent ID one-hot encoding (allows parameter-shared agents to differentiate)
        idx3 = idx2 + NUM_CHANNELS
        obs[idx3 + agent_id] = 1.0

        return obs

    def get_global_state(self):
        """
        Get the full global state (used during centralized training).
        Returns a concatenation of all agents' observations + channel states.
        """
        global_obs = []
        for i in range(self.n_agents):
            global_obs.append(self._get_observation(i))

        # Add full channel state
        channel_state = np.zeros(self.n_channels * 2, dtype=np.float32)
        for ch_id, channel in enumerate(self.channels):
            channel_state[ch_id] = len(channel.users) / self.n_agents
            channel_state[self.n_channels + ch_id] = 1.0 if channel.is_busy() else 0.0

        global_obs.append(channel_state)
        return np.concatenate(global_obs)

    def _get_metrics(self):
        """Calculate current episode metrics."""
        total = max(1, self.episode_metrics['total_transmissions'])
        successful = self.episode_metrics['successful_transmissions']
        collisions = self.episode_metrics['collisions']
        latency_count = max(1, self.episode_metrics['latency_count'])

        return {
            'throughput': successful,
            'packet_delivery_ratio': successful / total,
            'avg_latency': self.episode_metrics['total_latency'] / latency_count,
            'collision_rate': collisions / total,
            'total_transmissions': total,
            'step': self.current_step
        }

    def action_space_sample(self):
        """Sample a random action (channel selection)."""
        return np.random.randint(0, self.n_channels)

    def get_obs_dim(self):
        """Return observation dimension."""
        return OBS_DIM

    def get_action_dim(self):
        """Return action dimension."""
        return ACTION_DIM


class SUMOSpectrumEnvironment:
    """
    SUMO-integrated spectrum allocation environment.
    Uses TraCI to get realistic vehicle positions and speeds from SUMO,
    while keeping the same observation/action/reward interface.
    """

    def __init__(self, n_agents=NUM_VEHICLES, n_channels=NUM_CHANNELS,
                 max_steps=MAX_EPISODE_STEPS,
                 sumo_cfg=None, use_gui=False):
        self.n_agents = n_agents
        self.n_channels = n_channels
        self.max_steps = max_steps
        self.sumo_cfg = sumo_cfg or "sumo_config/urban.sumocfg"
        self.use_gui = use_gui
        self.sumo_running = False
        self._traci_label = "marl_dsa"
        self._connection = None

        # Wireless channels (same as standalone)
        self.channels = [WirelessChannel(i) for i in range(n_channels)]

        # Agent state tracking
        self.agent_data = {}  # {agent_id: {position, speed, payload, time_budget, ...}}
        self.current_step = 0
        self.episode_metrics = {}

    def _start_sumo(self):
        """Start or reload SUMO simulation."""
        import traci
        import os

        # If SUMO is already running, reload or restart
        if self.sumo_running:
            if not self.use_gui:
                # Headless: reload simulation (faster, no process restart)
                try:
                    traci.load(['-c', self.sumo_cfg, '--no-warnings', '--no-step-log'])
                    return
                except Exception:
                    pass
            # GUI mode or reload failed: full restart
            try:
                traci.close()
            except Exception:
                pass
            self.sumo_running = False

        sumo_home = os.environ.get('SUMO_HOME', '')
        if self.use_gui:
            sumo_binary = os.path.join(sumo_home, 'bin', 'sumo-gui')
        else:
            sumo_binary = os.path.join(sumo_home, 'bin', 'sumo')

        sumo_cmd = [
            sumo_binary,
            '-c', self.sumo_cfg,
            '--no-warnings',
            '--start',
            '--no-step-log',
        ]

        # Add GUI-specific flags for visible vehicle movement
        if self.use_gui:
            sumo_cmd += ['--delay', '50']

        import random
        port = random.randint(10000, 60000)
        traci.start(sumo_cmd, port=port, numRetries=3)
        self.sumo_running = True

    def _stop_sumo(self):
        """Stop the SUMO simulation."""
        import traci
        if self.sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            self.sumo_running = False

    def reset(self):
        """Reset environment with a fresh SUMO simulation."""
        import traci

        self._start_sumo()
        self.current_step = 0
        self.episode_metrics = {
            'total_transmissions': 0,
            'successful_transmissions': 0,
            'collisions': 0,
            'total_latency': 0.0,
            'latency_count': 0
        }

        # Advance SUMO until we have enough vehicles
        for _ in range(50):
            traci.simulationStep()
            veh_ids = traci.vehicle.getIDList()
            if len(veh_ids) >= self.n_agents:
                break

        # Initialize agent data from SUMO vehicles
        veh_ids = list(traci.vehicle.getIDList())[:self.n_agents]
        self.sumo_vehicle_ids = veh_ids

        for i, vid in enumerate(veh_ids):
            pos = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
            self.agent_data[i] = {
                'position': np.array([pos[0], pos[1]], dtype=np.float32),
                'speed': speed,
                'remaining_payload': np.random.randint(
                    PAYLOAD_SIZE_RANGE[0], PAYLOAD_SIZE_RANGE[1] + 1),
                'time_budget': np.random.randint(
                    TIME_BUDGET_RANGE[0], TIME_BUDGET_RANGE[1] + 1),
                'total_payload': 0,
                'packets_sent': 0,
                'packets_delivered': 0,
                'collisions_count': 0,
                'total_latency': 0.0,
                'previous_action': -1,
            }
            self.agent_data[i]['total_payload'] = self.agent_data[i]['remaining_payload']

        # Reset channels
        for ch in self.channels:
            ch.reset()

        return [self._get_observation(i) for i in range(self.n_agents)]

    def step(self, actions):
        """Execute one step: assign channels, compute rewards, advance SUMO."""
        import traci

        assert len(actions) == self.n_agents

        # Clear channels
        for ch in self.channels:
            ch.reset()

        # Assign agents to channels
        for agent_id, action in enumerate(actions):
            channel_id = int(action)
            self.channels[channel_id].users.append(agent_id)
            self.agent_data[agent_id]['previous_action'] = channel_id

        # Color vehicles by channel selection (visible in SUMO-GUI)
        channel_colors = {
            0: (0, 0, 255),      # Blue
            1: (255, 0, 0),      # Red
            2: (0, 255, 0),      # Green
            3: (255, 255, 0),    # Yellow
            4: (255, 0, 255)     # Purple
        }
        
        try:
            active_vehicles = set(traci.vehicle.getIDList())
            for agent_id, action in enumerate(actions):
                vid = self.sumo_vehicle_ids[agent_id]
                if vid in active_vehicles:
                    color = channel_colors.get(int(action), (255, 255, 255))
                    traci.vehicle.setColor(vid, color)
        except Exception:
            pass

        # Compute rewards (same logic as standalone)
        rewards = []
        for agent_id, action in enumerate(actions):
            channel_id = int(action)
            data = self.agent_data[agent_id]
            channel = self.channels[channel_id]

            data['packets_sent'] += 1
            self.episode_metrics['total_transmissions'] += 1
            num_users = channel.get_num_users()

            if num_users == 1:
                reward = REWARD_SUCCESS
                data['packets_delivered'] += 1
                data['remaining_payload'] -= 1
                self.episode_metrics['successful_transmissions'] += 1
                latency = self.current_step / max(1, data['total_payload'])
                data['total_latency'] += latency
                self.episode_metrics['total_latency'] += latency
                self.episode_metrics['latency_count'] += 1
            else:
                # Calculate distance-based interference
                interference = self._calc_interference(agent_id, channel_id)
                sinr = TRANSMISSION_POWER / (NOISE_POWER + interference)
                if sinr >= SNR_THRESHOLD:
                    reward = REWARD_SUCCESS * 0.5
                    data['packets_delivered'] += 1
                    data['remaining_payload'] -= 1
                    self.episode_metrics['successful_transmissions'] += 1
                    latency = self.current_step / max(1, data['total_payload'])
                    data['total_latency'] += latency
                    self.episode_metrics['total_latency'] += latency
                    self.episode_metrics['latency_count'] += 1
                else:
                    reward = REWARD_COLLISION
                    data['collisions_count'] += 1
                    self.episode_metrics['collisions'] += 1

            rewards.append(reward)

        # Advance SUMO simulation
        traci.simulationStep()
        self.current_step += 1

        # Update agent positions/speeds from SUMO
        current_vehs = traci.vehicle.getIDList()
        for i, vid in enumerate(self.sumo_vehicle_ids):
            if vid in current_vehs:
                pos = traci.vehicle.getPosition(vid)
                self.agent_data[i]['position'] = np.array(
                    [pos[0], pos[1]], dtype=np.float32)
                self.agent_data[i]['speed'] = traci.vehicle.getSpeed(vid)

            # Renew payload if exhausted
            if self.agent_data[i]['remaining_payload'] <= 0 or \
               self.agent_data[i]['time_budget'] <= 0:
                self.agent_data[i]['remaining_payload'] = np.random.randint(
                    PAYLOAD_SIZE_RANGE[0], PAYLOAD_SIZE_RANGE[1] + 1)
                self.agent_data[i]['time_budget'] = np.random.randint(
                    TIME_BUDGET_RANGE[0], TIME_BUDGET_RANGE[1] + 1)
                self.agent_data[i]['total_payload'] = \
                    self.agent_data[i]['remaining_payload']
            else:
                self.agent_data[i]['time_budget'] -= 1

        done = self.current_step >= self.max_steps
        if done:
            self._stop_sumo()

        observations = [self._get_observation(i) for i in range(self.n_agents)]
        info = self._get_metrics()

        return observations, rewards, done, info

    def _calc_interference(self, agent_id, channel_id):
        """Calculate distance-based interference from co-channel users."""
        interference = 0.0
        my_pos = self.agent_data[agent_id]['position']
        for other_id in self.channels[channel_id].users:
            if other_id == agent_id:
                continue
            other_pos = self.agent_data[other_id]['position']
            dist = np.sqrt(np.sum((my_pos - other_pos) ** 2))
            
            # Scale distance for real-world SUMO coordinates (meters)
            # In urban V2V, interference range is ~200-300m
            # Normalize so 30m = 1 grid unit for strong collisions
            scaled_dist = max(dist / 30.0, 0.5)
            
            path_loss = 1.0 / (scaled_dist ** PATH_LOSS_EXPONENT)
            interference += INTERFERENCE_FACTOR * TRANSMISSION_POWER * path_loss
        return interference

    def _get_observation(self, agent_id):
        """Construct observation for an agent (same format as standalone)."""
        data = self.agent_data[agent_id]
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Channel occupancy (normalized)
        for ch_id, ch in enumerate(self.channels):
            obs[ch_id] = ch.get_num_users() / max(1, self.n_agents)

        # Interference levels
        for ch_id in range(self.n_channels):
            obs[NUM_CHANNELS + ch_id] = min(
                1.0, self._calc_interference(agent_id, ch_id) / 2.0)

        # Normalized position (SUMO coordinates, normalize by grid extent)
        # 4x4 grid with 200m edges = ~800m extent
        grid_extent = 800.0
        idx = NUM_CHANNELS * 2
        obs[idx] = data['position'][0] / grid_extent
        obs[idx + 1] = data['position'][1] / grid_extent

        # Normalized speed (max SUMO speed ~15 m/s)
        obs[idx + 2] = data['speed'] / 15.0

        # Normalized remaining payload
        obs[idx + 3] = data['remaining_payload'] / PAYLOAD_SIZE_RANGE[1]

        # Normalized time budget
        obs[idx + 4] = data['time_budget'] / TIME_BUDGET_RANGE[1]

        # Previous action one-hot
        idx2 = idx + 5
        if data['previous_action'] >= 0:
            obs[idx2 + data['previous_action']] = 1.0

        # Agent ID one-hot
        idx3 = idx2 + NUM_CHANNELS
        obs[idx3 + agent_id] = 1.0

        return obs

    def _get_metrics(self):
        """Calculate current episode metrics."""
        total = max(1, self.episode_metrics['total_transmissions'])
        successful = self.episode_metrics['successful_transmissions']
        collisions = self.episode_metrics['collisions']
        latency_count = max(1, self.episode_metrics['latency_count'])
        return {
            'throughput': successful,
            'packet_delivery_ratio': successful / total,
            'avg_latency': self.episode_metrics['total_latency'] / latency_count,
            'collision_rate': collisions / total,
            'total_transmissions': total,
            'step': self.current_step
        }

    def action_space_sample(self):
        return np.random.randint(0, self.n_channels)

    def get_obs_dim(self):
        return OBS_DIM

    def get_action_dim(self):
        return ACTION_DIM

    def __del__(self):
        self._stop_sumo()


def create_environment(use_sumo=None):
    """
    Factory function to create the appropriate environment.
    Uses config.USE_SUMO by default, or overrides with argument.
    """
    from config import USE_SUMO, SUMO_CONFIG, SUMO_GUI
    if use_sumo is None:
        use_sumo = USE_SUMO

    if use_sumo:
        print("[ENV] Using SUMO-integrated environment")
        return SUMOSpectrumEnvironment(
            sumo_cfg=SUMO_CONFIG, use_gui=SUMO_GUI)
    else:
        print("[ENV] Using standalone environment (no SUMO)")
        return SpectrumEnvironment()
