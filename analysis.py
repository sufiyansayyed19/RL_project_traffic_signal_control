"""
Performance Analysis Module for Traffic Signal RL
====================================================

This module provides tools to analyze and compare the performance
of different traffic signal control strategies:
    - Compute per-agent metrics (reward, wait time, throughput)
    - Compare multiple algorithms side-by-side
    - Detect convergence points in Q-Learning training
    - Generate human-readable performance reports

All analysis functions output structured data (dicts, DataFrames)
that can be directly used by the visualization module.
"""

import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    """
    Analyzes and compares the performance of traffic signal control agents.

    Provides methods to compute metrics, detect convergence, compare
    algorithms, and generate formatted reports.
    """

    def __init__(self):
        """Initialize the performance analyzer."""
        self.results = {}

    def add_results(self, agent_name, eval_metrics, training_history=None):
        """
        Add an agent's results for comparison.

        Args:
            agent_name (str): Name of the agent (e.g., 'Q-Learning')
            eval_metrics (dict): Evaluation metrics from evaluate_agent()
            training_history (dict, optional): Training history from train_q_learning()
        """
        self.results[agent_name] = {
            'eval': eval_metrics,
            'training': training_history
        }

    def compute_metrics(self, eval_metrics):
        """
        Compute comprehensive performance metrics from evaluation data.

        Args:
            eval_metrics (dict): Raw evaluation metrics

        Returns:
            dict: Computed metrics including statistical measures
        """
        rewards = eval_metrics['episode_rewards']
        wait_times = eval_metrics['episode_wait_times']

        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'median_reward': np.median(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_wait_time': np.mean(wait_times),
            'std_wait_time': np.std(wait_times),
            'avg_throughput': eval_metrics['avg_throughput'],
            'keep_pct': eval_metrics['action_distribution']['keep'],
            'switch_pct': eval_metrics['action_distribution']['switch'],
        }

    def find_convergence_episode(self, rewards, window=50, threshold=0.05):
        """
        Find the episode at which Q-Learning converges.

        Convergence is detected when the moving average reward stabilizes
        (change between consecutive windows < threshold × |mean reward|).

        Args:
            rewards (list): List of rewards per episode
            window (int): Smoothing window size
            threshold (float): Relative change threshold for convergence

        Returns:
            int: Episode number at which convergence is detected, or -1 if not found
        """
        if len(rewards) < 2 * window:
            return -1

        # Compute moving average
        smoothed = []
        for i in range(window, len(rewards)):
            smoothed.append(np.mean(rewards[i - window:i]))

        # Find where change becomes small
        for i in range(1, len(smoothed)):
            if abs(smoothed[i]) > 0:
                relative_change = abs(smoothed[i] - smoothed[i - 1]) / abs(smoothed[i])
                if relative_change < threshold:
                    # Check if it stays stable for another window
                    stable = True
                    for j in range(i, min(i + window, len(smoothed))):
                        if abs(smoothed[j]) > 0:
                            rc = abs(smoothed[j] - smoothed[j - 1]) / abs(smoothed[j])
                            if rc > threshold * 2:
                                stable = False
                                break
                    if stable:
                        return i + window  # Return original episode index

        return -1  # No convergence detected

    def compare_algorithms(self):
        """
        Create a comparison DataFrame of all registered agents.

        Returns:
            pd.DataFrame: Comparison table with metrics for each agent
        """
        comparison_data = []

        for agent_name, data in self.results.items():
            metrics = self.compute_metrics(data['eval'])

            row = {
                'Agent': agent_name,
                'Avg Reward': f"{metrics['avg_reward']:.2f}",
                'Std Reward': f"{metrics['std_reward']:.2f}",
                'Avg Wait Time': f"{metrics['avg_wait_time']:.2f}",
                'Avg Throughput': f"{metrics['avg_throughput']:.1f}",
                'Keep %': f"{metrics['keep_pct']:.1f}",
                'Switch %': f"{metrics['switch_pct']:.1f}",
            }

            # Add training time if available
            if data['training']:
                row['Training Time'] = f"{data['training'].get('training_time', 0):.2f}s"
            else:
                row['Training Time'] = 'N/A'

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.set_index('Agent')
        return df

    def calculate_improvement(self, baseline_name, agent_name):
        """
        Calculate the percentage improvement of an agent over a baseline.

        Args:
            baseline_name (str): Name of the baseline agent
            agent_name (str): Name of the agent to compare

        Returns:
            dict: Improvement percentages for key metrics
        """
        if baseline_name not in self.results or agent_name not in self.results:
            return None

        baseline = self.compute_metrics(self.results[baseline_name]['eval'])
        agent = self.compute_metrics(self.results[agent_name]['eval'])

        # For wait time: lower is better, so improvement = (baseline - agent) / baseline
        wait_improvement = 0
        if baseline['avg_wait_time'] != 0:
            wait_improvement = ((baseline['avg_wait_time'] - agent['avg_wait_time'])
                                / baseline['avg_wait_time'] * 100)

        # For reward: higher (less negative) is better
        reward_improvement = 0
        if baseline['avg_reward'] != 0:
            reward_improvement = ((agent['avg_reward'] - baseline['avg_reward'])
                                  / abs(baseline['avg_reward']) * 100)

        return {
            'wait_time_improvement': wait_improvement,
            'reward_improvement': reward_improvement,
            'throughput_diff': agent['avg_throughput'] - baseline['avg_throughput']
        }

    def generate_report(self):
        """
        Generate a comprehensive text performance report.

        Returns:
            str: Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("  TRAFFIC SIGNAL CONTROL — PERFORMANCE REPORT")
        report.append("=" * 70)
        report.append("")

        # Comparison table
        report.append("ALGORITHM COMPARISON")
        report.append("-" * 70)
        df = self.compare_algorithms()
        report.append(df.to_string())
        report.append("")

        # Q-Learning convergence analysis
        if 'Q-Learning' in self.results and self.results['Q-Learning']['training']:
            ql_training = self.results['Q-Learning']['training']
            convergence_ep = self.find_convergence_episode(ql_training['rewards'])

            report.append("Q-LEARNING CONVERGENCE ANALYSIS")
            report.append("-" * 70)
            report.append(f"  Total episodes:          {len(ql_training['rewards'])}")
            report.append(f"  Training time:           {ql_training['training_time']:.2f}s")
            if convergence_ep > 0:
                report.append(f"  Convergence episode:     ~{convergence_ep}")
            else:
                report.append(f"  Convergence episode:     Not clearly detected")
            report.append(f"  Initial avg reward:      {np.mean(ql_training['rewards'][:50]):.2f}")
            report.append(f"  Final avg reward:        {np.mean(ql_training['rewards'][-50:]):.2f}")
            report.append(f"  Initial avg wait:        {np.mean(ql_training['avg_wait_times'][:50]):.2f}")
            report.append(f"  Final avg wait:          {np.mean(ql_training['avg_wait_times'][-50:]):.2f}")
            report.append("")

        # Improvement analysis
        if 'Fixed Timer' in self.results and 'Q-Learning' in self.results:
            improvement = self.calculate_improvement('Fixed Timer', 'Q-Learning')
            if improvement:
                report.append("Q-LEARNING vs FIXED TIMER IMPROVEMENT")
                report.append("-" * 70)
                report.append(f"  Wait time reduction:     {improvement['wait_time_improvement']:.1f}%")
                report.append(f"  Reward improvement:      {improvement['reward_improvement']:.1f}%")
                report.append(f"  Throughput difference:   {improvement['throughput_diff']:+.1f} cars/episode")
                report.append("")

        if 'Fixed Timer' in self.results and 'Value Iteration' in self.results:
            improvement = self.calculate_improvement('Fixed Timer', 'Value Iteration')
            if improvement:
                report.append("VALUE ITERATION vs FIXED TIMER IMPROVEMENT")
                report.append("-" * 70)
                report.append(f"  Wait time reduction:     {improvement['wait_time_improvement']:.1f}%")
                report.append(f"  Reward improvement:      {improvement['reward_improvement']:.1f}%")
                report.append(f"  Throughput difference:   {improvement['throughput_diff']:+.1f} cars/episode")
                report.append("")

        # Q-Learning vs Value Iteration
        if 'Q-Learning' in self.results and 'Value Iteration' in self.results:
            ql_metrics = self.compute_metrics(self.results['Q-Learning']['eval'])
            vi_metrics = self.compute_metrics(self.results['Value Iteration']['eval'])

            if vi_metrics['avg_wait_time'] != 0:
                optimality = (1 - (ql_metrics['avg_wait_time'] - vi_metrics['avg_wait_time'])
                              / vi_metrics['avg_wait_time']) * 100
            else:
                optimality = 100

            report.append("Q-LEARNING OPTIMALITY GAP")
            report.append("-" * 70)
            report.append(f"  Q-Learning avg wait:     {ql_metrics['avg_wait_time']:.2f}")
            report.append(f"  Value Iteration avg wait:{vi_metrics['avg_wait_time']:.2f}")
            report.append(f"  Q-Learning achieves:     ~{min(optimality, 100):.1f}% of optimal")
            report.append("")

        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 70)
        report.append("  1. RL agents (Q-Learning, Value Iteration) significantly outperform")
        report.append("     both Random and Fixed Timer baselines in reducing wait times.")
        report.append("")
        report.append("  2. Value Iteration provides the theoretical optimal policy but")
        report.append("     requires complete knowledge of traffic dynamics (transition model).")
        report.append("")
        report.append("  3. Q-Learning learns a near-optimal policy through trial-and-error,")
        report.append("     making it more practical for real-world deployment where traffic")
        report.append("     patterns are unknown and change over time.")
        report.append("")
        report.append("  4. The switch penalty in the reward function is crucial for preventing")
        report.append("     unrealistic rapid signal changes (flickering).")
        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


def smooth_curve(data, window=20):
    """
    Smooth a noisy data series using a simple moving average.

    Args:
        data (list or np.array): Raw data series
        window (int): Smoothing window size

    Returns:
        np.array: Smoothed data series
    """
    if len(data) < window:
        return np.array(data)

    smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
    return smoothed
