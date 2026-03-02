"""
Multi-Agent Reinforcement Learning for Dynamic Spectrum Allocation
Main Entry Point

Orchestrates the full pipeline:
  1. Train MARL agents (CTDE architecture)
  2. Evaluate against baselines (Static, Random)
  3. Generate visualization plots

Usage:
  python main.py                           # Full pipeline (train + evaluate + visualize)
  python main.py --train                   # Training only
  python main.py --evaluate                # Evaluation only (requires trained model)
  python main.py --visualize               # Visualization only (requires saved results)
  python main.py --train --episodes 100    # Train for 100 episodes
  python main.py --eval-episodes 50        # Evaluate with 50 episodes
"""

import argparse
import os
import json
import numpy as np

from config import NUM_EPISODES, EVAL_EPISODES, RESULTS_DIR, MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Agent Reinforcement Learning for Dynamic Spectrum Allocation'
    )
    parser.add_argument('--train', action='store_true',
                        help='Run training phase')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation phase')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                        help=f'Number of training episodes (default: {NUM_EPISODES})')
    parser.add_argument('--eval-episodes', type=int, default=EVAL_EPISODES,
                        help=f'Number of evaluation episodes (default: {EVAL_EPISODES})')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model for evaluation')
    return parser.parse_args()


def save_results(history=None, results=None, save_dir=RESULTS_DIR):
    """Save training history and evaluation results to JSON files."""
    os.makedirs(save_dir, exist_ok=True)

    if history is not None:
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) for v in value]
            else:
                history_serializable[key] = value

        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"  Training history saved to {save_dir}/training_history.json")

    if results is not None:
        results_serializable = {}
        for method, metrics in results.items():
            results_serializable[method] = {}
            for metric, values in metrics.items():
                results_serializable[method][metric] = {
                    'mean': float(values['mean']),
                    'std': float(values['std']),
                    'all': [float(v) for v in values['all']]
                }

        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"  Evaluation results saved to {save_dir}/evaluation_results.json")


def load_results(save_dir=RESULTS_DIR):
    """Load saved training history and evaluation results."""
    history = None
    results = None

    history_path = os.path.join(save_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"  Loaded training history from {history_path}")

    results_path = os.path.join(save_dir, 'evaluation_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"  Loaded evaluation results from {results_path}")

    return history, results


def main():
    args = parse_args()

    # If no specific phase selected, run everything
    run_all = not (args.train or args.evaluate or args.visualize)

    print()
    print("=" * 65)
    print("  Multi-Agent Reinforcement Learning")
    print("  for Dynamic Spectrum Allocation")
    print("=" * 65)
    print(f"  Modules:  {'Train' if (args.train or run_all) else ''}  "
          f"{'Evaluate' if (args.evaluate or run_all) else ''}  "
          f"{'Visualize' if (args.visualize or run_all) else ''}")
    print("=" * 65)
    print()

    history = None
    results = None
    agent = None

    # =========================================================================
    # Phase 1: Training
    # =========================================================================
    if args.train or run_all:
        print("[Phase 1] Training MARL Agents")
        print("-" * 40)
        from train import train
        agent, history = train(num_episodes=args.episodes, verbose=True)
        save_results(history=history)

    # =========================================================================
    # Phase 2: Evaluation
    # =========================================================================
    if args.evaluate or run_all:
        print("\n[Phase 2] Evaluating Against Baselines")
        print("-" * 40)
        from evaluate import run_evaluation
        results = run_evaluation(
            agent=agent,
            model_path=args.model_path,
            n_episodes=args.eval_episodes,
            verbose=True
        )
        save_results(results=results)

    # =========================================================================
    # Phase 3: Visualization
    # =========================================================================
    if args.visualize or run_all:
        print("\n[Phase 3] Generating Visualizations")
        print("-" * 40)
        from visualize import plot_all

        # Load saved results if not already in memory
        if history is None or results is None:
            loaded_history, loaded_results = load_results()
            if history is None:
                history = loaded_history
            if results is None:
                results = loaded_results

        if history is None and results is None:
            print("  Error: No training history or evaluation results found.")
            print("  Run with --train and/or --evaluate first.")
            return

        plot_all(history=history, results=results)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 65)
    print("  Pipeline Complete!")
    print("=" * 65)

    if results:
        print("\n  Final Results Summary:")
        print(f"  {'Method':<20} {'Throughput':>12} {'PDR':>10} "
              f"{'Latency':>10} {'Collision':>10}")
        print(f"  {'-'*62}")
        for method in results:
            t = results[method]['throughput']['mean']
            p = results[method]['packet_delivery_ratio']['mean']
            l = results[method]['avg_latency']['mean']
            c = results[method]['collision_rate']['mean']
            print(f"  {method:<20} {t:>12.2f} {p:>10.4f} {l:>10.4f} {c:>10.4f}")
        print()

    print(f"  Models saved to:   {MODEL_DIR}/")
    print(f"  Results saved to:  {RESULTS_DIR}/")
    print()


if __name__ == "__main__":
    main()
