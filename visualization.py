"""
Visualization Module for Traffic Signal RL
=============================================

Generates 6 professional-quality plots using matplotlib and seaborn:
    1. Learning Curves (reward + wait time over episodes)
    2. Algorithm Comparison (bar chart of all agents)
    3. Queue Dynamics (traffic flow over one episode)
    4. Action Distribution (keep vs switch per agent)
    5. Value Iteration Convergence (delta per iteration)
    6. Summary Dashboard (2×2 overview of key results)

All plots use a consistent dark theme with seaborn styling
for publication-quality output.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import smooth_curve


# ─── Global Style Configuration ───────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'figure.figsize': (12, 7),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
    'axes.edgecolor': '#3a3a5e',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.5,
})

# Color palette
COLORS = {
    'q_learning': '#00d4ff',    # Cyan
    'value_iter': '#ff6b6b',    # Coral
    'fixed_timer': '#ffd93d',   # Gold
    'random': '#95e1d3',        # Mint
    'accent': '#c084fc',        # Purple
    'raw_alpha': 0.15,          # Transparency for raw data
    'smooth': '#00d4ff',        # Smoothed line color
}


def plot_learning_curves(training_history, save_path='results/learning_curves.png'):
    """
    Plot Q-Learning training progress: reward and wait time over episodes.

    Creates a 2-row figure:
        Top: Total reward per episode (raw + smoothed)
        Bottom: Average wait time per episode (raw + smoothed)

    Args:
        training_history (dict): Training history from train_q_learning()
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle('Q-Learning Training Progress', fontsize=16, fontweight='bold', color='white')

    episodes = range(len(training_history['rewards']))

    # ─── Top: Reward per episode ───
    ax1 = axes[0]
    rewards = training_history['rewards']
    smoothed_rewards = smooth_curve(rewards, window=30)

    ax1.plot(episodes, rewards, alpha=COLORS['raw_alpha'], color=COLORS['q_learning'], linewidth=0.5)
    ax1.plot(range(29, 29 + len(smoothed_rewards)), smoothed_rewards,
             color=COLORS['q_learning'], linewidth=2.5, label='Smoothed (window=30)')
    ax1.set_ylabel('Total Reward per Episode')
    ax1.set_title('Reward Convergence')
    ax1.legend(loc='lower right')

    # Add convergence annotation
    if len(smoothed_rewards) > 0:
        final_reward = smoothed_rewards[-1]
        ax1.axhline(y=final_reward, color=COLORS['accent'], linestyle='--', alpha=0.5,
                     label=f'Final: {final_reward:.1f}')
        ax1.legend(loc='lower right')

    # ─── Bottom: Wait time per episode ───
    ax2 = axes[1]
    wait_times = training_history['avg_wait_times']
    smoothed_wait = smooth_curve(wait_times, window=30)

    ax2.plot(episodes, wait_times, alpha=COLORS['raw_alpha'], color=COLORS['value_iter'], linewidth=0.5)
    ax2.plot(range(29, 29 + len(smoothed_wait)), smoothed_wait,
             color=COLORS['value_iter'], linewidth=2.5, label='Smoothed (window=30)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Waiting Cars per Step')
    ax2.set_title('Wait Time Reduction')
    ax2.legend(loc='upper right')

    # Add epsilon on secondary y-axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(episodes, training_history['epsilons'],
                  color=COLORS['fixed_timer'], linewidth=1.5, linestyle=':', alpha=0.7, label='ε (Exploration)')
    ax2_twin.set_ylabel('Epsilon (ε)', color=COLORS['fixed_timer'])
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['fixed_timer'])
    ax2_twin.legend(loc='center right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_algorithm_comparison(comparison_results, save_path='results/algorithm_comparison.png'):
    """
    Plot a grouped bar chart comparing all agents on key metrics.

    Shows Average Wait Time, Average Reward, and Throughput for each agent.

    Args:
        comparison_results (dict): {agent_name: eval_metrics}
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold', color='white')

    agents = list(comparison_results.keys())
    colors = [COLORS['random'], COLORS['fixed_timer'], COLORS['q_learning'], COLORS['value_iter']]
    colors = colors[:len(agents)]

    # Metric 1: Average Wait Time (lower is better)
    wait_times = [comparison_results[a]['avg_wait_time'] for a in agents]
    bars1 = axes[0].bar(agents, wait_times, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title('Avg Wait Time\n(lower is better)', fontweight='bold')
    axes[0].set_ylabel('Avg Waiting Cars/Step')
    for bar, val in zip(bars1, wait_times):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10, color='white')

    # Metric 2: Average Reward (higher is better — less negative)
    avg_rewards = [comparison_results[a]['avg_reward'] for a in agents]
    bars2 = axes[1].bar(agents, avg_rewards, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_title('Avg Reward\n(higher is better)', fontweight='bold')
    axes[1].set_ylabel('Average Reward')
    for bar, val in zip(bars2, avg_rewards):
        y_pos = bar.get_height() - 0.5 if val < 0 else bar.get_height() + 0.1
        axes[1].text(bar.get_x() + bar.get_width() / 2., y_pos,
                     f'{val:.1f}', ha='center', va='top' if val < 0 else 'bottom',
                     fontweight='bold', fontsize=10, color='white')

    # Metric 3: Throughput (higher is better)
    throughputs = [comparison_results[a]['avg_throughput'] for a in agents]
    bars3 = axes[2].bar(agents, throughputs, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_title('Avg Throughput\n(higher is better)', fontweight='bold')
    axes[2].set_ylabel('Cars Served/Episode')
    for bar, val in zip(bars3, throughputs):
        axes[2].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                     f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10, color='white')

    # Rotate x-labels for readability
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_queue_dynamics(episode_details, save_path='results/queue_dynamics.png'):
    """
    Plot queue lengths and signal phases over one episode.

    Shows how a trained agent dynamically manages traffic flow,
    with vertical markers at each signal phase change.

    Args:
        episode_details (list): Per-step info dicts from evaluate_agent()
        save_path (str): Path to save the plot
    """
    if not episode_details:
        print("  [WARN] No episode details available for queue dynamics plot")
        return

    steps = range(len(episode_details))
    queues_n = [d['queues'][0] for d in episode_details]
    queues_s = [d['queues'][1] for d in episode_details]
    queues_e = [d['queues'][2] for d in episode_details]
    queues_w = [d['queues'][3] for d in episode_details]
    phases = [d['phase'] for d in episode_details]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Traffic Queue Dynamics (Single Episode)', fontsize=16, fontweight='bold', color='white')

    # ─── Top: Queue lengths ───
    ax1.plot(steps, queues_n, color='#ff6b6b', linewidth=2, label='North', marker='o', markersize=2)
    ax1.plot(steps, queues_s, color='#ffd93d', linewidth=2, label='South', marker='s', markersize=2)
    ax1.plot(steps, queues_e, color='#00d4ff', linewidth=2, label='East', marker='^', markersize=2)
    ax1.plot(steps, queues_w, color='#95e1d3', linewidth=2, label='West', marker='d', markersize=2)

    # Mark signal switches
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            ax1.axvline(x=i, color='#c084fc', alpha=0.3, linestyle='--', linewidth=1)

    ax1.set_ylabel('Queue Length (cars)')
    ax1.set_title('Queue Lengths per Direction')
    ax1.legend(loc='upper right', ncol=4)
    ax1.set_ylim(-0.5, max(max(queues_n + queues_s + queues_e + queues_w), 5) + 0.5)

    # ─── Bottom: Signal phase timeline ───
    phase_colors = ['#4caf50' if p == 0 else '#ff5722' for p in phases]
    ax2.bar(steps, [1] * len(steps), color=phase_colors, width=1.0, alpha=0.8)
    ax2.set_ylabel('Phase')
    ax2.set_xlabel('Time Step')
    ax2.set_title('Signal Phase (Green=NS | Orange=EW)')
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)

    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4caf50', label='NS Green'),
                       Patch(facecolor='#ff5722', label='EW Green')]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_action_distribution(comparison_results, save_path='results/action_distribution.png'):
    """
    Plot the action distribution (Keep vs Switch) for each agent.

    Shows how frequently each agent decides to keep or switch the signal,
    revealing different strategies learned by different algorithms.

    Args:
        comparison_results (dict): {agent_name: eval_metrics}
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, len(comparison_results), figsize=(4 * len(comparison_results), 5))
    fig.suptitle('Action Distribution per Agent', fontsize=16, fontweight='bold', color='white')

    if len(comparison_results) == 1:
        axes = [axes]

    agent_colors = {
        'Random': COLORS['random'],
        'Fixed Timer': COLORS['fixed_timer'],
        'Q-Learning': COLORS['q_learning'],
        'Value Iteration': COLORS['value_iter']
    }

    for idx, (agent_name, metrics) in enumerate(comparison_results.items()):
        ax = axes[idx]
        keep_pct = metrics['action_distribution']['keep']
        switch_pct = metrics['action_distribution']['switch']

        wedge_colors = [agent_colors.get(agent_name, COLORS['q_learning']), '#3a3a5e']
        wedges, texts, autotexts = ax.pie(
            [keep_pct, switch_pct],
            labels=['Keep', 'Switch'],
            autopct='%1.1f%%',
            colors=wedge_colors,
            startangle=90,
            textprops={'color': 'white', 'fontsize': 11},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title(agent_name, fontweight='bold', fontsize=13, color='white')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_vi_convergence(convergence_history, save_path='results/vi_convergence.png'):
    """
    Plot Value Iteration convergence: max delta per iteration.

    Shows exponential decay of value changes, demonstrating rapid convergence
    to the optimal value function.

    Args:
        convergence_history (list): Delta values per iteration from VI
        save_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Value Iteration Convergence', fontsize=16, fontweight='bold', color='white')

    iterations = range(1, len(convergence_history) + 1)

    ax.semilogy(iterations, convergence_history, color=COLORS['value_iter'],
                linewidth=2.5, marker='o', markersize=3, alpha=0.9)

    # Convergence threshold line
    ax.axhline(y=1e-6, color=COLORS['fixed_timer'], linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Threshold (θ = 1e-6)')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Value Change (Δ) — Log Scale')
    ax.set_title('Convergence of Value Function')
    ax.legend(loc='upper right')

    # Annotate convergence point
    ax.annotate(f'Converged at iteration {len(convergence_history)}',
                xy=(len(convergence_history), convergence_history[-1]),
                xytext=(len(convergence_history) * 0.5, convergence_history[0] * 0.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5),
                fontsize=11, color=COLORS['accent'], fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def create_summary_dashboard(comparison_results, training_history,
                              convergence_history, save_path='results/summary_dashboard.png'):
    """
    Create a 2×2 summary dashboard combining key visualizations.

    Layout:
        [Learning Curve]     [Algorithm Comparison]
        [Queue Dynamics]     [Convergence]

    Args:
        comparison_results (dict): {agent_name: eval_metrics}
        training_history (dict): Q-Learning training history
        convergence_history (list): VI convergence deltas
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Traffic Signal RL — Summary Dashboard',
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    # ─── Top-Left: Q-Learning Reward Curve ───
    ax = axes[0, 0]
    rewards = training_history['rewards']
    smoothed = smooth_curve(rewards, window=30)
    ax.plot(range(len(rewards)), rewards, alpha=0.1, color=COLORS['q_learning'])
    ax.plot(range(29, 29 + len(smoothed)), smoothed,
            color=COLORS['q_learning'], linewidth=2.5)
    ax.set_title('Q-Learning: Reward per Episode', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')

    # ─── Top-Right: Algorithm Comparison Bars ───
    ax = axes[0, 1]
    agents = list(comparison_results.keys())
    wait_times = [comparison_results[a]['avg_wait_time'] for a in agents]
    colors = [COLORS.get(a.lower().replace(' ', '_').replace('-', '_'), COLORS['q_learning'])
              for a in agents]
    # Use a fixed color list for consistency
    bar_colors = [COLORS['random'], COLORS['fixed_timer'],
                  COLORS['q_learning'], COLORS['value_iter']][:len(agents)]
    bars = ax.bar(agents, wait_times, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_title('Avg Wait Time Comparison', fontweight='bold')
    ax.set_ylabel('Avg Waiting Cars/Step')
    for bar, val in zip(bars, wait_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', color='white', fontsize=10)
    ax.tick_params(axis='x', rotation=15)

    # ─── Bottom-Left: Wait Time Over Training ───
    ax = axes[1, 0]
    wait_data = training_history['avg_wait_times']
    smoothed_wait = smooth_curve(wait_data, window=30)
    ax.plot(range(len(wait_data)), wait_data, alpha=0.1, color=COLORS['value_iter'])
    ax.plot(range(29, 29 + len(smoothed_wait)), smoothed_wait,
            color=COLORS['value_iter'], linewidth=2.5)
    ax.set_title('Q-Learning: Wait Time Reduction', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Waiting Cars/Step')

    # ─── Bottom-Right: VI Convergence ───
    ax = axes[1, 1]
    iters = range(1, len(convergence_history) + 1)
    ax.semilogy(iters, convergence_history, color=COLORS['value_iter'],
                linewidth=2.5, marker='o', markersize=2)
    ax.axhline(y=1e-6, color=COLORS['fixed_timer'], linestyle='--', alpha=0.5)
    ax.set_title('Value Iteration Convergence', fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Δ (log scale)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: {save_path}")
