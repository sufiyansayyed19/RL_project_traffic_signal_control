"""
Main Orchestrator — Traffic Signal Control using Reinforcement Learning
========================================================================

This script runs the complete pipeline:
    1. Create the traffic intersection environment
    2. Train Q-Learning agent (model-free, learns by interaction)
    3. Run Value Iteration (model-based, computes optimal policy)
    4. Evaluate baseline agents (Random, Fixed Timer)
    5. Evaluate RL agents
    6. Analyze and compare all approaches
    7. Generate professional visualizations (6 plots)
    8. Save performance report

Usage:
    python main.py

Output:
    results/learning_curves.png         — Q-Learning training progress
    results/algorithm_comparison.png    — Bar chart of all agents
    results/queue_dynamics.png          — Traffic flow over one episode
    results/action_distribution.png     — Keep vs Switch per agent
    results/vi_convergence.png          — Value Iteration convergence
    results/summary_dashboard.png       — 2×2 overview dashboard
    results/performance_report.txt      — Text report with metrics
"""

import os
import numpy as np

from environment import TrafficIntersection
from agents import QLearningAgent, ValueIterationAgent, FixedTimerAgent, RandomAgent
from training import train_q_learning, evaluate_agent, run_value_iteration
from analysis import PerformanceAnalyzer
from visualization import (
    plot_learning_curves,
    plot_algorithm_comparison,
    plot_queue_dynamics,
    plot_action_distribution,
    plot_vi_convergence,
    create_summary_dashboard
)


def main():
    """
    Run the complete Traffic Signal RL pipeline.
    """
    # ─── Configuration ────────────────────────────────────────────────
    SEED = 42
    Q_LEARNING_EPISODES = 2000
    MAX_STEPS = 100
    EVAL_EPISODES = 100
    RESULTS_DIR = 'results'

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("+" + "=" * 60 + "+")
    print("|  TRAFFIC SIGNAL CONTROL USING REINFORCEMENT LEARNING     |")
    print("+" + "=" * 60 + "+")
    print()

    # ─── Step 1: Create Environment ───────────────────────────────────
    print(">> Step 1: Creating traffic intersection environment...")
    env_config = {
        'max_queue': 5,
        'arrival_probs': {
            'north': 0.4,
            'south': 0.35,
            'east': 0.3,
            'west': 0.25
        },
        'departure_rate': 2,
        'switch_penalty': 0.5,
        'max_steps': MAX_STEPS,
        'seed': SEED
    }
    env = TrafficIntersection(env_config)
    print(f"  Environment created: {env.n_states} states, {env.n_actions} actions")
    print(f"  Config: {env.get_config()}")
    print()

    # ─── Step 2: Train Q-Learning Agent ───────────────────────────────
    print(">> Step 2: Training Q-Learning agent...")
    np.random.seed(SEED)
    ql_agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        lr=0.15,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.997
    )
    ql_history = train_q_learning(env, ql_agent, episodes=Q_LEARNING_EPISODES,
                                   max_steps=MAX_STEPS, verbose=True)

    # ─── Step 3: Run Value Iteration ──────────────────────────────────
    print(">> Step 3: Running Value Iteration...")
    # Create a fresh environment for VI (needs deterministic transition model)
    vi_env = TrafficIntersection(env_config)
    vi_agent = ValueIterationAgent(vi_env, gamma=0.95, theta=1e-6, max_iterations=500)
    vi_convergence = run_value_iteration(vi_env, vi_agent, verbose=True)

    # ─── Step 4: Evaluate All Agents ──────────────────────────────────
    print(">> Step 4: Evaluating all agents...")
    print("-" * 60)

    # Create evaluation environment (same config, different seed for fairness)
    eval_config = env_config.copy()
    eval_config['seed'] = SEED + 100
    eval_env = TrafficIntersection(eval_config)

    # Evaluate Random baseline
    random_agent = RandomAgent()
    np.random.seed(SEED + 200)
    random_metrics = evaluate_agent(eval_env, random_agent, episodes=EVAL_EPISODES,
                                     max_steps=MAX_STEPS, agent_name="Random")

    # Evaluate Fixed Timer baseline
    fixed_agent = FixedTimerAgent(switch_interval=5)
    np.random.seed(SEED + 300)
    fixed_metrics = evaluate_agent(eval_env, fixed_agent, episodes=EVAL_EPISODES,
                                    max_steps=MAX_STEPS, agent_name="Fixed Timer")

    # Evaluate Q-Learning (exploitation only)
    old_epsilon = ql_agent.epsilon
    ql_agent.epsilon = 0  # Pure exploitation during evaluation
    np.random.seed(SEED + 400)
    ql_metrics = evaluate_agent(eval_env, ql_agent, episodes=EVAL_EPISODES,
                                 max_steps=MAX_STEPS, agent_name="Q-Learning")
    ql_agent.epsilon = old_epsilon  # Restore

    # Evaluate Value Iteration
    np.random.seed(SEED + 500)
    vi_metrics = evaluate_agent(eval_env, vi_agent, episodes=EVAL_EPISODES,
                                 max_steps=MAX_STEPS, agent_name="Value Iteration")

    print("-" * 60)

    # ─── Step 5: Analysis ─────────────────────────────────────────────
    print("\n>> Step 5: Analyzing results...")
    analyzer = PerformanceAnalyzer()
    analyzer.add_results('Random', random_metrics)
    analyzer.add_results('Fixed Timer', fixed_metrics)
    analyzer.add_results('Q-Learning', ql_metrics, ql_history)
    analyzer.add_results('Value Iteration', vi_metrics,
                          {'training_time': vi_convergence.get('training_time', 0)})

    # Print comparison table
    print("\n" + "=" * 60)
    print("  COMPARISON TABLE")
    print("=" * 60)
    comparison_df = analyzer.compare_algorithms()
    print(comparison_df.to_string())

    # Generate full report
    report = analyzer.generate_report()
    report_path = os.path.join(RESULTS_DIR, 'performance_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  [OK] Report saved: {report_path}")

    # ─── Step 6: Visualization ────────────────────────────────────────
    print("\n>> Step 6: Generating visualizations...")
    print("-" * 60)

    # Collect all evaluation results for comparison plots
    all_eval_results = {
        'Random': random_metrics,
        'Fixed Timer': fixed_metrics,
        'Q-Learning': ql_metrics,
        'Value Iteration': vi_metrics
    }

    # Plot 1: Learning Curves
    plot_learning_curves(ql_history,
                          save_path=os.path.join(RESULTS_DIR, 'learning_curves.png'))

    # Plot 2: Algorithm Comparison
    plot_algorithm_comparison(all_eval_results,
                               save_path=os.path.join(RESULTS_DIR, 'algorithm_comparison.png'))

    # Plot 3: Queue Dynamics (using Q-Learning's episode details)
    if ql_metrics.get('episode_details'):
        plot_queue_dynamics(ql_metrics['episode_details'],
                            save_path=os.path.join(RESULTS_DIR, 'queue_dynamics.png'))

    # Plot 4: Action Distribution
    plot_action_distribution(all_eval_results,
                              save_path=os.path.join(RESULTS_DIR, 'action_distribution.png'))

    # Plot 5: Value Iteration Convergence
    if vi_convergence.get('history'):
        plot_vi_convergence(vi_convergence['history'],
                            save_path=os.path.join(RESULTS_DIR, 'vi_convergence.png'))

    # Plot 6: Summary Dashboard
    create_summary_dashboard(
        all_eval_results,
        ql_history,
        vi_convergence.get('history', []),
        save_path=os.path.join(RESULTS_DIR, 'summary_dashboard.png')
    )

    print("-" * 60)

    # ─── Final Summary ────────────────────────────────────────────────
    print("\n+" + "=" * 60 + "+")
    print("|  PIPELINE COMPLETE                                       |")
    print("+" + "=" * 60 + "+")
    print("|  Generated outputs in results/:                          |")
    print("|    - learning_curves.png                                 |")
    print("|    - algorithm_comparison.png                            |")
    print("|    - queue_dynamics.png                                  |")
    print("|    - action_distribution.png                             |")
    print("|    - vi_convergence.png                                  |")
    print("|    - summary_dashboard.png                               |")
    print("|    - performance_report.txt                              |")
    print("+" + "=" * 60 + "+")


if __name__ == '__main__':
    main()
