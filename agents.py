"""
Reinforcement Learning Agents for Traffic Signal Control
=========================================================

This module implements four agents:
    1. QLearningAgent     — Model-free TD learning (learns by interacting)
    2. ValueIterationAgent — Model-based DP (uses transition model)
    3. FixedTimerAgent     — Baseline: switches every N steps
    4. RandomAgent         — Baseline: random actions

The RL agents (Q-Learning, Value Iteration) are compared against the
baselines to demonstrate that learned policies outperform naive strategies.
"""

import numpy as np


class QLearningAgent:
    """
    Q-Learning Agent (Off-policy Temporal Difference Learning).

    Learns an action-value function Q(s, a) using the update rule:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

    Uses ε-greedy exploration strategy with decaying ε.

    Attributes:
        n_states (int): Total number of states in the environment
        n_actions (int): Number of available actions
        lr (float): Learning rate α
        gamma (float): Discount factor γ
        epsilon (float): Current exploration rate ε
        epsilon_min (float): Minimum exploration rate
        epsilon_decay (float): Multiplicative decay per episode
        q_table (np.ndarray): Q-value table of shape (n_states, n_actions)
    """

    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize the Q-Learning agent.

        Args:
            n_states (int): Number of states in the environment
            n_actions (int): Number of actions available
            lr (float): Learning rate (step size for Q-updates)
            gamma (float): Discount factor for future rewards
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate (floor)
            epsilon_decay (float): Decay multiplier applied per episode
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table with small random values (breaks symmetry)
        self.q_table = np.random.uniform(low=-0.01, high=0.01,
                                         size=(n_states, n_actions))

        # Track learning statistics
        self.td_errors = []

    def select_action(self, state_index):
        """
        Select an action using ε-greedy policy.

        With probability ε: choose a random action (exploration)
        With probability 1-ε: choose the action with highest Q-value (exploitation)

        Args:
            state_index (int): Current state index

        Returns:
            int: Selected action (0 = Keep, 1 = Switch)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.q_table[state_index])  # Exploit

    def learn(self, state_idx, action, reward, next_state_idx):
        """
        Update Q-value using the TD learning rule.

        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            state_idx (int): Current state index
            action (int): Action taken
            reward (float): Reward received
            next_state_idx (int): Next state index
        """
        # Current Q-value
        current_q = self.q_table[state_idx, action]

        # TD target: r + γ max_a' Q(s', a')
        best_next_q = np.max(self.q_table[next_state_idx])
        td_target = reward + self.gamma * best_next_q

        # TD error: target - current
        td_error = td_target - current_q

        # Update Q-value
        self.q_table[state_idx, action] += self.lr * td_error

        # Track TD error for analysis
        self.td_errors.append(abs(td_error))

    def decay_epsilon(self):
        """
        Decay the exploration rate after each episode.
        Ensures ε never goes below ε_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table(self):
        """
        Get a copy of the current Q-table.

        Returns:
            np.ndarray: Copy of Q-table (n_states × n_actions)
        """
        return self.q_table.copy()

    def get_policy(self):
        """
        Extract the greedy policy from the Q-table.

        Returns:
            np.ndarray: Policy array (best action for each state)
        """
        return np.argmax(self.q_table, axis=1)

    def get_stats(self):
        """
        Get training statistics.

        Returns:
            dict: Statistics including mean TD error and Q-value stats
        """
        return {
            'mean_td_error': np.mean(self.td_errors[-1000:]) if self.td_errors else 0,
            'mean_q_value': np.mean(self.q_table),
            'max_q_value': np.max(self.q_table),
            'min_q_value': np.min(self.q_table),
            'epsilon': self.epsilon,
            'total_updates': len(self.td_errors)
        }


class ValueIterationAgent:
    """
    Value Iteration Agent (Model-Based Dynamic Programming).

    Computes the optimal value function V*(s) using the Bellman optimality equation:
        V(s) ← max_a Σ_s' T(s'|s,a) [R(s,a,s') + γ V(s')]

    Then extracts the optimal policy:
        π*(s) = argmax_a Σ_s' T(s'|s,a) [R(s,a,s') + γ V(s')]

    Requires: Full knowledge of the environment's transition dynamics T(s'|s,a)

    Attributes:
        env: TrafficIntersection environment (for transition model)
        gamma (float): Discount factor
        theta (float): Convergence threshold
        max_iterations (int): Maximum number of iterations
        values (np.ndarray): State value function V(s)
        policy (np.ndarray): Optimal policy π(s)
    """

    def __init__(self, env, gamma=0.95, theta=1e-6, max_iterations=500):
        """
        Initialize the Value Iteration agent.

        Args:
            env: TrafficIntersection environment instance
            gamma (float): Discount factor for future rewards
            theta (float): Convergence threshold (stop when max Δ < θ)
            max_iterations (int): Maximum allowed iterations
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.n_states = env.n_states
        self.n_actions = env.n_actions

        # Value function and policy
        self.values = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)

        # Convergence history for plotting
        self.convergence_history = []

    def value_iteration(self):
        """
        Run the Value Iteration algorithm.

        Iteratively updates V(s) until convergence (max change < θ)
        or maximum iterations reached.

        Returns:
            dict: Convergence information
                - iterations (int): Number of iterations performed
                - converged (bool): Whether convergence was achieved
                - final_delta (float): Final maximum value change
                - history (list): Delta values per iteration
        """
        print("Running Value Iteration...")
        print(f"  States: {self.n_states}, gamma={self.gamma}, theta={self.theta}")

        for iteration in range(self.max_iterations):
            delta = 0

            # Update V(s) for each state
            for state in self.env.get_all_states():
                state_idx = self.env.state_to_index(state)
                old_value = self.values[state_idx]

                # Compute V(s) = max_a Σ_s' T(s'|s,a) [R + γ V(s')]
                action_values = []
                for action in range(self.n_actions):
                    transitions = self.env.get_transition_prob(state, action)
                    q_value = 0
                    for prob, next_state, reward in transitions:
                        next_idx = self.env.state_to_index(next_state)
                        q_value += prob * (reward + self.gamma * self.values[next_idx])
                    action_values.append(q_value)

                # Update V(s) to the best action value
                self.values[state_idx] = max(action_values)

                # Track maximum change
                delta = max(delta, abs(old_value - self.values[state_idx]))

            self.convergence_history.append(delta)

            # Print progress every 20 iterations
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}: delta = {delta:.8f}")

            # Check convergence
            if delta < self.theta:
                print(f"  Converged at iteration {iteration + 1} (delta={delta:.10f})")
                self._extract_policy()
                return {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_delta': delta,
                    'history': self.convergence_history
                }

        print(f"  Max iterations reached ({self.max_iterations}), delta={delta:.8f}")
        self._extract_policy()
        return {
            'iterations': self.max_iterations,
            'converged': False,
            'final_delta': delta,
            'history': self.convergence_history
        }

    def _extract_policy(self):
        """
        Extract the optimal policy from the computed value function.

        π*(s) = argmax_a Σ_s' T(s'|s,a) [R(s,a,s') + γ V(s')]
        """
        for state in self.env.get_all_states():
            state_idx = self.env.state_to_index(state)

            action_values = []
            for action in range(self.n_actions):
                transitions = self.env.get_transition_prob(state, action)
                q_value = 0
                for prob, next_state, reward in transitions:
                    next_idx = self.env.state_to_index(next_state)
                    q_value += prob * (reward + self.gamma * self.values[next_idx])
                action_values.append(q_value)

            self.policy[state_idx] = np.argmax(action_values)

    def select_action(self, state_index):
        """
        Select action using the computed optimal policy.

        Args:
            state_index (int): Current state index

        Returns:
            int: Action from optimal policy
        """
        return self.policy[state_index]

    def get_policy(self):
        """
        Get the computed optimal policy.

        Returns:
            np.ndarray: Policy array (best action for each state)
        """
        return self.policy.copy()

    def get_stats(self):
        """
        Get Value Iteration statistics.

        Returns:
            dict: Statistics about the converged value function
        """
        return {
            'mean_value': np.mean(self.values),
            'max_value': np.max(self.values),
            'min_value': np.min(self.values),
            'iterations': len(self.convergence_history),
            'final_delta': self.convergence_history[-1] if self.convergence_history else None,
            'keep_actions': np.sum(self.policy == 0),
            'switch_actions': np.sum(self.policy == 1),
        }


class FixedTimerAgent:
    """
    Fixed Timer Baseline Agent.

    Switches the traffic signal at a fixed interval, regardless of traffic conditions.
    This mimics traditional traffic lights that operate on timed cycles.

    This is the standard approach in most real-world traffic signals,
    making it a meaningful baseline to compare RL agents against.
    """

    def __init__(self, switch_interval=5):
        """
        Initialize the fixed timer agent.

        Args:
            switch_interval (int): Number of steps between each switch
        """
        self.switch_interval = switch_interval
        self.step_count = 0

    def select_action(self, state_index=None):
        """
        Select action based on fixed timer.

        Args:
            state_index: Ignored (included for API compatibility)

        Returns:
            int: 1 (Switch) every switch_interval steps, else 0 (Keep)
        """
        self.step_count += 1
        if self.step_count % self.switch_interval == 0:
            return 1  # Switch
        return 0  # Keep

    def reset(self):
        """Reset the step counter for a new episode."""
        self.step_count = 0

    def get_stats(self):
        """Return baseline statistics."""
        return {
            'type': 'Fixed Timer',
            'switch_interval': self.switch_interval
        }


class RandomAgent:
    """
    Random Baseline Agent.

    Selects actions uniformly at random. Serves as a lower bound
    to verify that RL agents learn meaningful policies.
    """

    def __init__(self, n_actions=2):
        """
        Initialize the random agent.

        Args:
            n_actions (int): Number of available actions
        """
        self.n_actions = n_actions

    def select_action(self, state_index=None):
        """
        Select a random action.

        Args:
            state_index: Ignored (included for API compatibility)

        Returns:
            int: Random action
        """
        return np.random.randint(self.n_actions)

    def reset(self):
        """No-op for API compatibility."""
        pass

    def get_stats(self):
        """Return baseline statistics."""
        return {
            'type': 'Random',
            'n_actions': self.n_actions
        }
