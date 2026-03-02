"""
Module 3: Training Phase
Implements the Centralized Training and Distributed Execution (CTDE) loop
for multi-agent DRQN learning with shared parameters and agent ID conditioning.
"""

import os
import time
import numpy as np
from collections import deque
from tqdm import tqdm

from environment import create_environment
from agent import DRQNAgent, ReplayBuffer, EpisodeMemory
from config import (
    NUM_VEHICLES, NUM_EPISODES, TARGET_UPDATE_FREQ,
    SAVE_MODEL_FREQ, MODEL_DIR, SEQUENCE_LENGTH, BATCH_SIZE
)


def train(num_episodes=NUM_EPISODES, verbose=True):
    """
    Train MARL agents using CTDE architecture.

    All agents share the same DRQN model with agent ID conditioning.
    Each agent's ID is embedded to learn differentiated channel preferences.

    Args:
        num_episodes: number of training episodes
        verbose: whether to print training progress

    Returns:
        agent: trained DRQNAgent
        training_history: dict with training metrics over episodes
    """
    env = create_environment()
    agent = DRQNAgent(agent_id=0)
    replay_buffer = ReplayBuffer()

    history = {
        'episode_rewards': [],
        'episode_throughput': [],
        'episode_pdr': [],
        'episode_latency': [],
        'episode_collision_rate': [],
        'episode_loss': [],
        'epsilon': []
    }

    reward_window = deque(maxlen=50)
    best_avg_reward = -float('inf')
    warmup_episodes = max(BATCH_SIZE * 2, 64)

    os.makedirs(MODEL_DIR, exist_ok=True)

    start_time = time.time()
    pbar = tqdm(range(1, num_episodes + 1), desc="Training", disable=not verbose)

    for episode in pbar:
        observations = env.reset()
        agent.reset_hidden()

        # Each vehicle gets its own episode memory with its agent ID
        episode_memories = [EpisodeMemory(agent_id=i) for i in range(env.n_agents)]
        episode_reward = 0.0
        episode_loss = 0.0
        update_count = 0

        done = False
        while not done:
            # Each agent selects action using its own ID
            actions = []
            for i in range(env.n_agents):
                action = agent.select_action(observations[i], agent_id=i)
                actions.append(action)

            next_observations, rewards, done, info = env.step(actions)

            for i in range(env.n_agents):
                episode_memories[i].add(
                    observations[i], actions[i], rewards[i],
                    next_observations[i], float(done)
                )

            episode_reward += sum(rewards)
            observations = next_observations

        # Add episode experiences to replay buffer
        for memory in episode_memories:
            if len(memory) >= SEQUENCE_LENGTH:
                replay_buffer.add_episode(memory)

        # Post-episode training updates (after warmup)
        if episode > warmup_episodes and replay_buffer.is_ready():
            n_updates = min(4, len(replay_buffer) // BATCH_SIZE)
            for _ in range(n_updates):
                loss = agent.update(replay_buffer)
                episode_loss += loss
                update_count += 1
            agent.soft_update_target()

        # Hard update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.hard_update_target()

        agent.decay_epsilon()

        avg_reward = episode_reward / env.n_agents
        reward_window.append(avg_reward)
        avg_loss = episode_loss / max(1, update_count)

        history['episode_rewards'].append(avg_reward)
        history['episode_throughput'].append(info['throughput'])
        history['episode_pdr'].append(info['packet_delivery_ratio'])
        history['episode_latency'].append(info['avg_latency'])
        history['episode_collision_rate'].append(info['collision_rate'])
        history['episode_loss'].append(avg_loss)
        history['epsilon'].append(agent.epsilon)

        rolling_avg = np.mean(reward_window)
        pbar.set_postfix({
            'Avg Reward': f'{avg_reward:.1f}',
            'Rolling50': f'{rolling_avg:.1f}',
            'PDR': f'{info["packet_delivery_ratio"]:.2f}',
            'Collisions': f'{info["collision_rate"]:.2f}',
            'ε': f'{agent.epsilon:.3f}'
        })

        if rolling_avg > best_avg_reward and len(reward_window) >= 50:
            best_avg_reward = rolling_avg
            agent.save(os.path.join(MODEL_DIR, 'best_model.pth'))

        if episode % SAVE_MODEL_FREQ == 0:
            agent.save(os.path.join(MODEL_DIR, f'checkpoint_ep{episode}.pth'))

    agent.save(os.path.join(MODEL_DIR, 'final_model.pth'))

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"  Total time:       {elapsed:.1f} seconds")
        print(f"  Episodes:         {num_episodes}")
        print(f"  Final Epsilon:    {agent.epsilon:.4f}")
        print(f"  Best Avg Reward:  {best_avg_reward:.2f}")
        print(f"  Final PDR:        {info['packet_delivery_ratio']:.4f}")
        print(f"  Final Collision:  {info['collision_rate']:.4f}")
        print(f"  Model saved to:   {MODEL_DIR}/")
        print(f"{'='*60}")

    return agent, history


if __name__ == "__main__":
    agent, history = train()
