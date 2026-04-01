"""
Enhanced SUMO visual demo for dynamic spectrum allocation.

Runs one GUI-visible episode, highlights a tracked vehicle, records a replay
trace, and generates polished post-run visual outputs.
"""

import argparse
import os
import sys

from agent import DRQNAgent
from config import MAX_EPISODE_STEPS, MODEL_DIR, RESULTS_DIR, SUMO_CONFIG
from enhanced_visualization import generate_visual_report
from environment import SUMOSpectrumEnvironment
from evaluate import RandomAllocation, StaticAllocation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the enhanced SUMO visualization demo'
    )
    parser.add_argument(
        '--mode',
        choices=['marl', 'static', 'random'],
        default='marl',
        help='Policy used for channel selection during the demo'
    )
    parser.add_argument(
        '--model-path',
        default=os.path.join(MODEL_DIR, 'best_model.pth'),
        help='Model path for MARL demo mode'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=MAX_EPISODE_STEPS,
        help='Number of SUMO steps to run'
    )
    parser.add_argument(
        '--tracked-agent',
        type=int,
        default=0,
        help='Agent ID to keep highlighted in the GUI and replay outputs'
    )
    parser.add_argument(
        '--trace-path',
        default=os.path.join(RESULTS_DIR, 'enhanced_trace.json'),
        help='Where to save the recorded replay trace'
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.join(RESULTS_DIR, 'enhanced_visuals'),
        help='Directory for the generated summary images and animation'
    )
    return parser.parse_args(argv)


def _build_policy(mode, model_path, n_agents, n_channels):
    if mode == 'static':
        return StaticAllocation(n_agents, n_channels), 'Static Allocation'
    if mode == 'random':
        return RandomAllocation(n_channels), 'Random Allocation'

    agent = DRQNAgent(agent_id=0)
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0.0
        agent.policy_net.eval()
        return agent, 'MARL (DRQN)'

    print(f"Model not found at {model_path}. Falling back to static allocation.")
    return StaticAllocation(n_agents, n_channels), 'Static Allocation'


def _select_actions(policy, policy_name, observations, env):
    actions = []
    is_marl = policy_name == 'MARL (DRQN)'
    for agent_id in range(env.n_agents):
        if is_marl:
            action = policy.select_action(observations[agent_id], agent_id=agent_id, evaluate=True)
        else:
            action = policy.select_action(agent_id, observations[agent_id])
        actions.append(action)
    return actions


def run_demo(argv=None):
    args = parse_args(argv)
    if not os.environ.get('SUMO_HOME'):
        print("ERROR: SUMO_HOME environment variable not set.")
        sys.exit(1)

    env = SUMOSpectrumEnvironment(
        max_steps=args.steps,
        sumo_cfg=SUMO_CONFIG,
        use_gui=True,
        trace_path=args.trace_path,
        tracked_agent_id=args.tracked_agent,
    )

    policy, policy_name = _build_policy(args.mode, args.model_path, env.n_agents, env.n_channels)
    if hasattr(policy, 'reset_hidden'):
        policy.reset_hidden()

    print("=" * 72)
    print("  Enhanced SUMO Visualization Demo")
    print("=" * 72)
    print(f"  Policy:           {policy_name}")
    print(f"  SUMO Config:      {SUMO_CONFIG}")
    print(f"  Tracked Vehicle:  Agent {args.tracked_agent}")
    print(f"  Trace Output:     {args.trace_path}")
    print(f"  Visual Output:    {args.output_dir}")
    print("=" * 72)
    print("  SUMO controls: scroll to zoom, drag to pan, the tracked vehicle will auto-follow.")
    print()

    observations = env.reset()
    done = False
    last_info = None

    try:
        while not done:
            actions = _select_actions(policy, policy_name, observations, env)
            observations, rewards, done, info = env.step(actions)
            last_info = info

            if info['step'] % 20 == 0:
                tracked_id = env.tracked_agent_id
                tracked = env.agent_data[tracked_id]
                print(
                    f"  Step {info['step']:3d} | Tracked Agent {tracked_id} | "
                    f"pos=({tracked['position'][0]:.1f}, {tracked['position'][1]:.1f}) | "
                    f"speed={tracked['speed']:.1f} m/s | "
                    f"PDR={info['packet_delivery_ratio']:.3f} | "
                    f"collision={info['collision_rate']:.3f}"
                )
    finally:
        trace_file = env.export_trace(policy_name=policy_name)
        env._stop_sumo()

    outputs = generate_visual_report(trace_file, args.output_dir) if trace_file else {}

    print()
    print("=" * 72)
    print("  Demo Complete")
    print("=" * 72)
    if last_info:
        print(f"  Final throughput:     {last_info['throughput']:.0f}")
        print(f"  Final PDR:            {last_info['packet_delivery_ratio']:.4f}")
        print(f"  Final collision rate: {last_info['collision_rate']:.4f}")
    if trace_file:
        print(f"  Trace saved to:       {trace_file}")
    for label, path in outputs.items():
        print(f"  {label}: {path}")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()
