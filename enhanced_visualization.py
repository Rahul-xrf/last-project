"""
Enhanced movement visualization for SUMO replay traces.

Generates presentation-friendly outputs from a recorded trace:
  - overview dashboard
  - collision heatmap
  - 3D trajectory plot
  - animated replay GIF when Pillow is available
"""

import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


CHANNEL_COLORS = {
    0: '#2E86DE',
    1: '#E74C3C',
    2: '#27AE60',
    3: '#F1C40F',
    4: '#9B59B6',
}


def load_trace(trace_path):
    with open(trace_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _resolve_net_path(metadata):
    sumo_cfg = metadata.get('sumo_cfg')
    if not sumo_cfg or not os.path.exists(sumo_cfg):
        return None

    try:
        tree = ET.parse(sumo_cfg)
    except ET.ParseError:
        return None

    root = tree.getroot()
    net_node = root.find(".//input/net-file")
    if net_node is None:
        return None

    net_value = net_node.attrib.get('value')
    if not net_value:
        return None

    candidate = os.path.join(os.path.dirname(sumo_cfg), net_value)
    return os.path.abspath(candidate) if os.path.exists(candidate) else None


def _load_network_segments(metadata):
    net_path = _resolve_net_path(metadata)
    if not net_path:
        return []

    segments = []
    try:
        for _, elem in ET.iterparse(net_path, events=('end',)):
            if elem.tag != 'lane':
                continue
            if elem.attrib.get('function') == 'internal':
                elem.clear()
                continue
            shape = elem.attrib.get('shape')
            if not shape:
                elem.clear()
                continue
            coords = []
            for point in shape.split():
                x_str, y_str = point.split(',')
                coords.append((float(x_str), float(y_str)))
            if len(coords) >= 2:
                segments.append(coords)
            elem.clear()
    except ET.ParseError:
        return []

    return segments


def _prepare_series(trace):
    frames = trace['frames']
    vehicle_history = defaultdict(list)
    for frame in frames:
        step = frame['step']
        for vehicle in frame['vehicles']:
            vehicle_history[vehicle['agent_id']].append({
                'step': step,
                **vehicle
            })
    return vehicle_history


def _get_trace_bounds(frames):
    xs = []
    ys = []
    for frame in frames:
        for vehicle in frame['vehicles']:
            xs.append(vehicle['x'])
            ys.append(vehicle['y'])
    if not xs or not ys:
        return (0, 1, 0, 1)
    margin = 40
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def _draw_network(ax, segments):
    if not segments:
        return
    collection = LineCollection(segments, colors='#3A3A3A', linewidths=0.8, alpha=0.4)
    ax.add_collection(collection)


def plot_visual_summary(trace, save_dir):
    frames = trace['frames']
    metadata = trace['metadata']
    history = _prepare_series(trace)
    network_segments = _load_network_segments(metadata)
    xmin, xmax, ymin, ymax = _get_trace_bounds(frames)

    fig = plt.figure(figsize=(18, 10), facecolor='#F4F1EA')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1.2, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_channels = fig.add_subplot(gs[1, 1])

    _draw_network(ax_map, network_segments)
    ax_map.set_facecolor('#E4ECE2')
    ax_map.set_title('Vehicle Movement, Trails, and Focus Tracking', fontsize=16, fontweight='bold')

    tracked_agent = metadata.get('tracked_agent_id', 0)
    final_frame = frames[-1]

    for agent_id, points in history.items():
        coords = np.array([(p['x'], p['y']) for p in points], dtype=np.float32)
        if len(coords) < 2:
            continue
        line_color = CHANNEL_COLORS.get(agent_id % max(1, metadata['n_channels']), '#7F8C8D')
        alpha = 0.22 if agent_id != tracked_agent else 0.95
        width = 1.0 if agent_id != tracked_agent else 3.0
        ax_map.plot(coords[:, 0], coords[:, 1], color=line_color, linewidth=width, alpha=alpha)

    for vehicle in final_frame['vehicles']:
        color = CHANNEL_COLORS.get(vehicle['channel'], '#34495E')
        size = 80 if not vehicle['is_tracked'] else 220
        edge = '#FFFFFF' if not vehicle['is_tracked'] else '#00E5FF'
        ax_map.scatter(vehicle['x'], vehicle['y'], s=size, c=color, edgecolors=edge, linewidths=2.5, zorder=3)
        if vehicle['collision']:
            ax_map.scatter(vehicle['x'], vehicle['y'], s=340, facecolors='none', edgecolors='#FF3B30',
                           linewidths=2.5, zorder=4)

        if vehicle['is_tracked']:
            ax_map.annotate(
                f"Tracked Vehicle {vehicle['agent_id']}\nCh {vehicle['channel']} | {vehicle['speed']:.1f} m/s",
                xy=(vehicle['x'], vehicle['y']),
                xytext=(vehicle['x'] + 50, vehicle['y'] + 50),
                fontsize=10,
                color='#102A43',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#00B8D9', lw=2),
                arrowprops=dict(arrowstyle='-|>', lw=2, color='#00B8D9')
            )

    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.set_xlabel('X Position (m)')
    ax_map.set_ylabel('Y Position (m)')

    steps = [frame['step'] for frame in frames]
    pdr = [frame['metrics']['packet_delivery_ratio'] for frame in frames]
    collision = [frame['metrics']['collision_rate'] for frame in frames]
    throughput = [frame['metrics']['throughput'] for frame in frames]
    ax_metrics.plot(steps, pdr, color='#27AE60', linewidth=2.5, label='PDR')
    ax_metrics.plot(steps, collision, color='#E74C3C', linewidth=2.5, label='Collision Rate')
    ax_metrics.plot(steps, np.array(throughput) / max(1.0, max(throughput)),
                    color='#2E86DE', linewidth=2.5, label='Throughput (normalized)')
    ax_metrics.set_title('How the Episode Evolves Over Time', fontsize=14, fontweight='bold')
    ax_metrics.set_xlabel('Step')
    ax_metrics.set_ylabel('Value')
    ax_metrics.grid(alpha=0.25)
    ax_metrics.legend(frameon=False)

    channel_matrix = np.full((metadata['n_agents'], len(frames)), -1)
    for frame_index, frame in enumerate(frames):
        for vehicle in frame['vehicles']:
            channel_matrix[vehicle['agent_id'], frame_index] = -1 if vehicle['channel'] is None else vehicle['channel']

    cmap = mcolors.ListedColormap(['#D0D7DE'] + [CHANNEL_COLORS.get(i, '#7F8C8D') for i in range(metadata['n_channels'])])
    display_matrix = channel_matrix + 1
    ax_channels.imshow(display_matrix, aspect='auto', interpolation='nearest', cmap=cmap)
    ax_channels.set_title('Channel Choice Timeline by Vehicle', fontsize=14, fontweight='bold')
    ax_channels.set_xlabel('Step')
    ax_channels.set_ylabel('Agent ID')
    ax_channels.set_yticks(range(metadata['n_agents']))
    ax_channels.set_yticklabels([str(i) for i in range(metadata['n_agents'])])

    fig.suptitle(
        f"Enhanced SUMO Replay Dashboard: {metadata.get('policy_name', 'Policy Replay')}",
        fontsize=18,
        fontweight='bold',
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(save_dir, 'visual_summary.png')
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_collision_heatmap(trace, save_dir):
    frames = trace['frames']
    metadata = trace['metadata']
    network_segments = _load_network_segments(metadata)

    collisions = [(v['x'], v['y']) for frame in frames for v in frame['vehicles'] if v['collision']]
    xs = [pos[0] for pos in collisions]
    ys = [pos[1] for pos in collisions]
    xmin, xmax, ymin, ymax = _get_trace_bounds(frames)

    fig, ax = plt.subplots(figsize=(12, 9), facecolor='#F8F5F0')
    _draw_network(ax, network_segments)
    ax.set_facecolor('#EEF5EC')
    if collisions:
        ax.hexbin(xs, ys, gridsize=25, cmap='YlOrRd', mincnt=1, alpha=0.9)
    else:
        ax.text(0.5, 0.5, 'No collisions recorded in this trace', ha='center', va='center',
                fontsize=15, transform=ax.transAxes)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title('Collision Hotspots on the Road Network', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')

    output_path = os.path.join(save_dir, 'collision_heatmap.png')
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_3d_trajectories(trace, save_dir):
    history = _prepare_series(trace)
    metadata = trace['metadata']

    fig = plt.figure(figsize=(14, 10), facecolor='#F7F4EF')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Trajectory View: X, Y, and Time', fontsize=16, fontweight='bold', pad=20)

    tracked_agent = metadata.get('tracked_agent_id', 0)
    for agent_id, points in history.items():
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p['step'] for p in points]
        color = CHANNEL_COLORS.get(agent_id % max(1, metadata['n_channels']), '#7F8C8D')
        width = 1.2 if agent_id != tracked_agent else 3.5
        alpha = 0.3 if agent_id != tracked_agent else 0.95
        ax.plot(xs, ys, zs, color=color, linewidth=width, alpha=alpha)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Step')
    ax.view_init(elev=28, azim=-55)

    output_path = os.path.join(save_dir, 'trajectory_3d.png')
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_replay_animation(trace, save_dir, max_frames=140):
    frames = trace['frames']
    metadata = trace['metadata']
    network_segments = _load_network_segments(metadata)
    xmin, xmax, ymin, ymax = _get_trace_bounds(frames)

    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in indices]

    fig, (ax_map, ax_info) = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [3.2, 1]}, facecolor='#F5F1E8'
    )
    ax_info.axis('off')

    def update(frame_index):
        frame = frames[frame_index]
        ax_map.clear()
        ax_info.clear()
        ax_info.axis('off')

        _draw_network(ax_map, network_segments)
        ax_map.set_facecolor('#E6EFE3')
        ax_map.set_xlim(xmin, xmax)
        ax_map.set_ylim(ymin, ymax)
        ax_map.set_title('Replay Animation with Trails and Collision Alerts', fontsize=15, fontweight='bold')
        ax_map.set_xlabel('X Position (m)')
        ax_map.set_ylabel('Y Position (m)')

        trail_start = max(0, frame_index - 20)
        frame_slice = frames[trail_start:frame_index + 1]
        positions_by_agent = defaultdict(list)
        for trail_frame in frame_slice:
            for vehicle in trail_frame['vehicles']:
                positions_by_agent[vehicle['agent_id']].append((vehicle['x'], vehicle['y']))

        tracked_vehicle = None
        for agent_id, coords in positions_by_agent.items():
            coords_array = np.array(coords, dtype=np.float32)
            if len(coords_array) < 2:
                continue
            ax_map.plot(coords_array[:, 0], coords_array[:, 1],
                        color=CHANNEL_COLORS.get(agent_id % max(1, metadata['n_channels']), '#7F8C8D'),
                        linewidth=2.2 if agent_id == frame['tracked_agent_id'] else 1.0,
                        alpha=0.9 if agent_id == frame['tracked_agent_id'] else 0.2)

        for vehicle in frame['vehicles']:
            color = CHANNEL_COLORS.get(vehicle['channel'], '#34495E')
            ax_map.scatter(vehicle['x'], vehicle['y'],
                           s=220 if vehicle['is_tracked'] else 90,
                           c=color, edgecolors='#FFFFFF', linewidths=1.8, zorder=4)
            if vehicle['collision']:
                ax_map.scatter(vehicle['x'], vehicle['y'], s=380, facecolors='none',
                               edgecolors='#FF3B30', linewidths=2.8, zorder=5)
            if vehicle['is_tracked']:
                tracked_vehicle = vehicle

        if tracked_vehicle is not None:
            ax_map.annotate(
                f"Vehicle {tracked_vehicle['agent_id']}",
                xy=(tracked_vehicle['x'], tracked_vehicle['y']),
                xytext=(tracked_vehicle['x'] + 45, tracked_vehicle['y'] + 35),
                arrowprops=dict(arrowstyle='->', color='#00B8D9', lw=2.2),
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#00B8D9', lw=2),
                fontsize=10
            )
            info_lines = [
                'Focused Vehicle',
                f"ID: {tracked_vehicle['agent_id']}",
                f"Channel: {tracked_vehicle['channel']}",
                f"Speed: {tracked_vehicle['speed']:.1f} m/s",
                f"Payload Left: {tracked_vehicle['remaining_payload']}",
                f"Time Budget: {tracked_vehicle['time_budget']}",
                f"Event: {'Collision' if tracked_vehicle['collision'] else 'Success' if tracked_vehicle['success'] else 'Tracking'}",
            ]
        else:
            info_lines = ['Focused Vehicle', 'Unavailable']

        metrics = frame['metrics']
        info_lines += [
            '',
            'Episode Status',
            f"Step: {frame['step']}",
            f"Throughput: {metrics['throughput']:.0f}",
            f"PDR: {metrics['packet_delivery_ratio']:.3f}",
            f"Collision Rate: {metrics['collision_rate']:.3f}",
        ]
        ax_info.text(
            0.05, 0.95, '\n'.join(info_lines), va='top', fontsize=12, color='#102A43',
            bbox=dict(boxstyle='round,pad=0.6', fc='white', ec='#C5D1DC', lw=1.5)
        )

    animation = FuncAnimation(fig, update, frames=len(frames), interval=120, repeat=True)
    output_path = os.path.join(save_dir, 'movement_replay.gif')
    try:
        animation.save(output_path, writer=PillowWriter(fps=8))
        plt.close(fig)
        return output_path
    except Exception:
        plt.close(fig)
        return None


def generate_visual_report(trace_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    trace = load_trace(trace_path)

    outputs = {
        'visual_summary': plot_visual_summary(trace, save_dir),
        'collision_heatmap': plot_collision_heatmap(trace, save_dir),
        'trajectory_3d': plot_3d_trajectories(trace, save_dir),
    }
    animation_path = create_replay_animation(trace, save_dir)
    if animation_path is not None:
        outputs['movement_replay'] = animation_path

    return outputs
