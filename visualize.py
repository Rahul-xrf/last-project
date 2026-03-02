"""
Module 4: Visualization
Creates publication-quality plots for training curves,
metric comparisons, and performance analysis.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import RESULTS_DIR


# Styling configuration
COLORS = {
    'MARL (DRQN)': '#2196F3',     # Blue
    'Static': '#FF9800',           # Orange
    'Random': '#F44336',           # Red
}
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def plot_training_curves(history, save_dir=RESULTS_DIR):
    """
    Plot training reward curve with rolling average.

    Args:
        history: dict with 'episode_rewards', 'epsilon', etc.
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MARL Training Progress', fontsize=16, fontweight='bold', y=0.98)

    episodes = range(1, len(history['episode_rewards']) + 1)

    # 1. Reward curve
    ax = axes[0, 0]
    ax.plot(episodes, history['episode_rewards'], alpha=0.3, color='#2196F3',
            label='Per Episode')
    # Rolling average
    window = min(50, len(history['episode_rewards']))
    if window > 1:
        rolling = np.convolve(history['episode_rewards'],
                              np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(history['episode_rewards']) + 1), rolling,
                color='#1565C0', linewidth=2, label=f'Rolling Avg ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Reward')
    ax.legend()

    # 2. Packet Delivery Ratio
    ax = axes[0, 1]
    ax.plot(episodes, history['episode_pdr'], alpha=0.3, color='#4CAF50')
    if window > 1:
        rolling = np.convolve(history['episode_pdr'],
                              np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(history['episode_pdr']) + 1), rolling,
                color='#2E7D32', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('PDR')
    ax.set_title('Packet Delivery Ratio')
    ax.set_ylim(0, 1.05)

    # 3. Collision Rate
    ax = axes[1, 0]
    ax.plot(episodes, history['episode_collision_rate'], alpha=0.3, color='#F44336')
    if window > 1:
        rolling = np.convolve(history['episode_collision_rate'],
                              np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(history['episode_collision_rate']) + 1), rolling,
                color='#C62828', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Collision Rate')
    ax.set_title('Collision Rate')
    ax.set_ylim(0, 1.05)

    # 4. Epsilon decay
    ax = axes[1, 1]
    ax.plot(episodes, history['epsilon'], color='#9C27B0', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (ε)')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_metric_comparison(results, save_dir=RESULTS_DIR):
    """
    Create bar charts comparing all 4 metrics across methods.

    Args:
        results: dict from run_evaluation() with metrics for each method
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    methods = list(results.keys())
    metrics = {
        'throughput': ('Throughput (Successful Transmissions)', 'Count'),
        'packet_delivery_ratio': ('Packet Delivery Ratio', 'Ratio'),
        'avg_latency': ('Average Latency', 'Time Steps'),
        'collision_rate': ('Collision Rate', 'Rate')
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison: MARL vs Baselines',
                 fontsize=16, fontweight='bold', y=0.98)

    bar_width = 0.25
    x_positions = np.arange(len(methods))

    for idx, (metric_key, (title, ylabel)) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]

        means = [results[m][metric_key]['mean'] for m in methods]
        stds = [results[m][metric_key]['std'] for m in methods]
        colors = [COLORS.get(m, '#757575') for m in methods]

        bars = ax.bar(x_positions, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor='white',
                      linewidth=1.5, width=0.5)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold',
                    fontsize=10)

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, fontsize=10)

        # Highlight best performer
        if metric_key in ['throughput', 'packet_delivery_ratio']:
            best = np.argmax(means)
        else:
            best = np.argmin(means)
        bars[best].set_edgecolor('#FFD700')
        bars[best].set_linewidth(3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'metric_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_episode_trends(results, save_dir=RESULTS_DIR):
    """
    Plot per-episode metric trends across methods.

    Args:
        results: dict from run_evaluation()
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    methods = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Episode Performance Trends',
                 fontsize=16, fontweight='bold', y=0.98)

    metrics = {
        'throughput': ('Throughput per Episode', 'Count'),
        'packet_delivery_ratio': ('PDR per Episode', 'Ratio'),
        'avg_latency': ('Latency per Episode', 'Time Steps'),
        'collision_rate': ('Collision Rate per Episode', 'Rate')
    }

    for idx, (metric_key, (title, ylabel)) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]

        for method in methods:
            data = results[method][metric_key]['all']
            episodes = range(1, len(data) + 1)
            color = COLORS.get(method, '#757575')

            ax.plot(episodes, data, alpha=0.3, color=color)
            # Rolling average
            window = min(10, len(data))
            if window > 1:
                rolling = np.convolve(data, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(data) + 1), rolling,
                        color=color, linewidth=2, label=method)
            else:
                ax.plot(episodes, data, color=color, linewidth=2, label=method)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Evaluation Episode')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'episode_trends.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_all(history=None, results=None, save_dir=RESULTS_DIR):
    """
    Generate all visualization plots.

    Args:
        history: training history dict (from train())
        results: evaluation results dict (from run_evaluation())
        save_dir: directory to save all plots
    """
    print(f"\nGenerating visualizations...")
    print(f"{'='*40}")

    if history is not None:
        plot_training_curves(history, save_dir)

    if results is not None:
        plot_metric_comparison(results, save_dir)
        plot_episode_trends(results, save_dir)

    print(f"{'='*40}")
    print(f"All plots saved to: {save_dir}/\n")
