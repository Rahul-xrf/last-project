"""
Module 4: Evaluation
Evaluates the trained MARL agent against baseline methods
(Static Allocation, Random Allocation) using key performance metrics.
"""

import os
import numpy as np
from tqdm import tqdm

from environment import create_environment
from agent import DRQNAgent
from config import (
    NUM_VEHICLES, NUM_CHANNELS, EVAL_EPISODES, MODEL_DIR, ACTION_DIM
)


class StaticAllocation:
    """
    Baseline: Fixed channel assignment.
    Each vehicle is assigned a fixed channel based on its ID.
    This mimics traditional pre-planned spectrum allocation.
    """

    def __init__(self, n_agents, n_channels):
        self.n_agents = n_agents
        self.n_channels = n_channels
        self.assignments = [i % n_channels for i in range(n_agents)]

    def select_action(self, agent_id, observation=None):
        return self.assignments[agent_id]

    def reset_hidden(self):
        pass


class RandomAllocation:
    """
    Baseline: Random channel selection.
    Each vehicle randomly selects a channel at each time step.
    """

    def __init__(self, n_channels):
        self.n_channels = n_channels

    def select_action(self, agent_id=None, observation=None):
        return np.random.randint(0, self.n_channels)

    def reset_hidden(self):
        pass


def evaluate_method(env, method, method_name, n_episodes=EVAL_EPISODES,
                    is_marl=False, verbose=True):
    """
    Evaluate a spectrum allocation method over multiple episodes.

    Args:
        env: SpectrumEnvironment instance
        method: allocation method (MARL agent, Static, or Random)
        method_name: string name for display
        n_episodes: number of evaluation episodes
        is_marl: whether the method is the MARL agent
        verbose: show progress bar

    Returns:
        metrics: dict with averaged performance metrics
    """
    all_throughput = []
    all_pdr = []
    all_latency = []
    all_collision_rate = []

    pbar = tqdm(range(n_episodes), desc=f"Evaluating {method_name}",
                disable=not verbose)

    for ep in pbar:
        observations = env.reset()
        if is_marl:
            method.reset_hidden()

        done = False
        while not done:
            actions = []
            for i in range(env.n_agents):
                if is_marl:
                    action = method.select_action(
                        observations[i], agent_id=i, evaluate=True
                    )
                else:
                    action = method.select_action(i, observations[i])
                actions.append(action)

            observations, rewards, done, info = env.step(actions)

        all_throughput.append(info['throughput'])
        all_pdr.append(info['packet_delivery_ratio'])
        all_latency.append(info['avg_latency'])
        all_collision_rate.append(info['collision_rate'])

    metrics = {
        'throughput': {
            'mean': np.mean(all_throughput),
            'std': np.std(all_throughput),
            'all': all_throughput
        },
        'packet_delivery_ratio': {
            'mean': np.mean(all_pdr),
            'std': np.std(all_pdr),
            'all': all_pdr
        },
        'avg_latency': {
            'mean': np.mean(all_latency),
            'std': np.std(all_latency),
            'all': all_latency
        },
        'collision_rate': {
            'mean': np.mean(all_collision_rate),
            'std': np.std(all_collision_rate),
            'all': all_collision_rate
        }
    }

    return metrics


def run_evaluation(agent=None, model_path=None, n_episodes=EVAL_EPISODES,
                   verbose=True):
    """
    Run full evaluation comparing MARL vs Static vs Random allocation.

    Args:
        agent: pre-trained DRQNAgent (optional, loads from model_path if None)
        model_path: path to saved model weights
        n_episodes: evaluation episodes per method
        verbose: print results

    Returns:
        results: dict with metrics for each method
    """
    env = create_environment()

    if agent is None:
        agent = DRQNAgent(agent_id=0)
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, 'best_model.pth')
        if os.path.exists(model_path):
            agent.load(model_path)
            if verbose:
                print(f"Loaded model from {model_path}")
        else:
            if verbose:
                print(f"Warning: No model found at {model_path}, using untrained agent")

    agent.epsilon = 0.0
    agent.policy_net.eval()

    static = StaticAllocation(env.n_agents, env.n_channels)
    random_alloc = RandomAllocation(env.n_channels)

    results = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evaluation: {n_episodes} episodes per method")
        print(f"{'='*60}\n")

    results['MARL (DRQN)'] = evaluate_method(
        env, agent, "MARL (DRQN)", n_episodes, is_marl=True, verbose=verbose
    )

    results['Static'] = evaluate_method(
        env, static, "Static Allocation", n_episodes, is_marl=False, verbose=verbose
    )

    results['Random'] = evaluate_method(
        env, random_alloc, "Random Allocation", n_episodes, is_marl=False,
        verbose=verbose
    )

    if verbose:
        print(f"\n{'='*75}")
        print(f"  {'Metric':<25} {'MARL (DRQN)':>15} {'Static':>15} {'Random':>15}")
        print(f"  {'-'*70}")

        metric_names = {
            'throughput': 'Throughput',
            'packet_delivery_ratio': 'Packet Delivery Ratio',
            'avg_latency': 'Avg Latency',
            'collision_rate': 'Collision Rate'
        }

        for key, display_name in metric_names.items():
            marl_val = results['MARL (DRQN)'][key]['mean']
            static_val = results['Static'][key]['mean']
            random_val = results['Random'][key]['mean']
            print(f"  {display_name:<25} {marl_val:>15.4f} {static_val:>15.4f} "
                  f"{random_val:>15.4f}")

        print(f"  {'='*70}")

        print(f"\n  MARL Improvements over baselines:")
        pdr_marl = results['MARL (DRQN)']['packet_delivery_ratio']['mean']
        pdr_static = results['Static']['packet_delivery_ratio']['mean']
        pdr_random = results['Random']['packet_delivery_ratio']['mean']

        if pdr_static > 0:
            print(f"    vs Static:  PDR {((pdr_marl - pdr_static)/pdr_static)*100:+.1f}%")
        if pdr_random > 0:
            print(f"    vs Random:  PDR {((pdr_marl - pdr_random)/pdr_random)*100:+.1f}%")
        print()

    return results


if __name__ == "__main__":
    results = run_evaluation()
