"""
Training Module for Traffic Signal RL
=======================================

This module provides training and evaluation functions for all agents:
    - train_q_learning(): Trains the Q-Learning agent over episodes
    - evaluate_agent(): Evaluates any agent's performance (no learning)
    - run_value_iteration(): Runs Value Iteration to compute optimal policy

All functions return structured history/metrics dictionaries for analysis.
"""

import numpy as np
import time


def train_q_learning(env, agent, episodes=1000, max_steps=100, verbose=True):
    """
    Train the Q-Learning agent by interacting with the environment.

    For each episode:
        1. Reset environment
        2. For each step: select action, take step, learn from experience
        3. Decay exploration rate
        4. Record metrics

    Args:
        env: TrafficIntersection environment
        agent: QLearningAgent instance
        episodes (int): Number of training episodes
        max_steps (int): Maximum steps per episode
        verbose (bool): Whether to print progress

    Returns:
        dict: Training history containing:
            - rewards: List of total rewards per episode
            - avg_wait_times: List of average waiting times per episode
            - epsilons: List of epsilon values per episode
            - episode_lengths: List of steps per episode
            - action_counts: List of [keep_count, switch_count] per episode
            - training_time: Total training time in seconds
    """
    history = {
        'rewards': [],
        'avg_wait_times': [],
        'epsilons': [],
        'episode_lengths': [],
        'action_counts': [],
        'training_time': 0
    }

    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Q-LEARNING TRAINING")
        print(f"  Episodes: {episodes} | Steps/episode: {max_steps}")
        print(f"  lr={agent.lr} | gamma={agent.gamma} | eps={agent.epsilon:.2f}->{agent.epsilon_min}")
        print(f"{'='*60}")

    for episode in range(episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)

        total_reward = 0
        total_waiting = 0
        keep_count = 0
        switch_count = 0

        for step in range(max_steps):
            # Select action using epsilon-greedy
            action = agent.select_action(state_idx)

            # Take action in environment
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            # Learn from this experience
            agent.learn(state_idx, action, reward, next_state_idx)

            # Track metrics
            total_reward += reward
            total_waiting += info['total_waiting']
            if action == 0:
                keep_count += 1
            else:
                switch_count += 1

            # Move to next state
            state_idx = next_state_idx

            if done:
                break

        # Decay exploration rate after each episode
        agent.decay_epsilon()

        # Record episode metrics
        history['rewards'].append(total_reward)
        history['avg_wait_times'].append(total_waiting / max_steps)
        history['epsilons'].append(agent.epsilon)
        history['episode_lengths'].append(step + 1)
        history['action_counts'].append([keep_count, switch_count])

        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(history['rewards'][-100:])
            avg_wait = np.mean(history['avg_wait_times'][-100:])
            print(f"  Episode {episode + 1:>5}/{episodes} | "
                  f"Avg Reward: {avg_reward:>8.2f} | "
                  f"Avg Wait: {avg_wait:>5.2f} | "
                  f"eps: {agent.epsilon:.4f}")

    elapsed = time.time() - start_time
    history['training_time'] = elapsed

    if verbose:
        print(f"{'-'*60}")
        print(f"  Training complete in {elapsed:.2f} seconds")
        print(f"  Final eps: {agent.epsilon:.4f}")
        print(f"  Final Avg Reward (last 100): {np.mean(history['rewards'][-100:]):.2f}")
        print(f"  Final Avg Wait (last 100): {np.mean(history['avg_wait_times'][-100:]):.2f}")
        print(f"{'='*60}\n")

    return history


def evaluate_agent(env, agent, episodes=100, max_steps=100, agent_name="Agent", verbose=True):
    """
    Evaluate an agent's performance without learning (pure exploitation).

    Runs the agent's current policy for multiple episodes and collects
    performance metrics. No Q-table updates or exploration.

    Args:
        env: TrafficIntersection environment
        agent: Any agent with a select_action(state_index) method
        episodes (int): Number of evaluation episodes
        max_steps (int): Maximum steps per episode
        agent_name (str): Name for display purposes
        verbose (bool): Whether to print results

    Returns:
        dict: Evaluation metrics including:
            - avg_reward: Average total reward per episode
            - std_reward: Standard deviation of rewards
            - avg_wait_time: Average waiting time per step
            - avg_throughput: Average cars served per episode
            - action_distribution: {keep: %, switch: %}
            - episode_rewards: All episode rewards
            - episode_wait_times: All episode wait times
            - episode_details: Detailed per-step data for one episode
    """
    episode_rewards = []
    episode_wait_times = []
    episode_throughputs = []
    total_keep = 0
    total_switch = 0
    episode_details = None  # Store details of last episode for visualization

    # Reset fixed timer if applicable
    if hasattr(agent, 'reset'):
        agent.reset()

    for episode in range(episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)

        total_reward = 0
        total_waiting = 0
        step_details = []

        if hasattr(agent, 'reset'):
            agent.reset()

        for step in range(max_steps):
            action = agent.select_action(state_idx)
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            total_reward += reward
            total_waiting += info['total_waiting']
            if action == 0:
                total_keep += 1
            else:
                total_switch += 1

            # Store step details for last episode
            if episode == episodes - 1:
                step_details.append(info)

            state_idx = next_state_idx
            if done:
                break

        episode_rewards.append(total_reward)
        episode_wait_times.append(total_waiting / max_steps)
        episode_throughputs.append(env.total_cars_served)

        if episode == episodes - 1:
            episode_details = step_details

    total_actions = total_keep + total_switch
    metrics = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_wait_time': np.mean(episode_wait_times),
        'std_wait_time': np.std(episode_wait_times),
        'avg_throughput': np.mean(episode_throughputs),
        'action_distribution': {
            'keep': total_keep / total_actions * 100 if total_actions > 0 else 50,
            'switch': total_switch / total_actions * 100 if total_actions > 0 else 50
        },
        'episode_rewards': episode_rewards,
        'episode_wait_times': episode_wait_times,
        'episode_details': episode_details
    }

    if verbose:
        print(f"\n  [{agent_name}] Evaluation Results ({episodes} episodes):")
        print(f"    Avg Reward:     {metrics['avg_reward']:>8.2f} +/- {metrics['std_reward']:.2f}")
        print(f"    Avg Wait Time:  {metrics['avg_wait_time']:>8.2f} +/- {metrics['std_wait_time']:.2f}")
        print(f"    Avg Throughput: {metrics['avg_throughput']:>8.1f} cars/episode")
        print(f"    Actions: Keep {metrics['action_distribution']['keep']:.1f}% | "
              f"Switch {metrics['action_distribution']['switch']:.1f}%")

    return metrics


def run_value_iteration(env, vi_agent, verbose=True):
    """
    Run Value Iteration to compute the optimal policy.

    This is a wrapper that calls the agent's value_iteration() method
    and adds timing information.

    Args:
        env: TrafficIntersection environment (used by vi_agent internally)
        vi_agent: ValueIterationAgent instance
        verbose (bool): Whether to print progress

    Returns:
        dict: Convergence information with timing
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  VALUE ITERATION")
        print(f"  States: {env.n_states} | gamma={vi_agent.gamma} | theta={vi_agent.theta}")
        print(f"{'='*60}")

    start_time = time.time()
    convergence_info = vi_agent.value_iteration()
    elapsed = time.time() - start_time

    convergence_info['training_time'] = elapsed

    if verbose:
        print(f"{'-'*60}")
        print(f"  Completed in {elapsed:.2f} seconds")
        stats = vi_agent.get_stats()
        print(f"  Keep actions: {stats['keep_actions']} | Switch actions: {stats['switch_actions']}")
        print(f"  Mean value: {stats['mean_value']:.4f}")
        print(f"{'='*60}\n")

    return convergence_info
