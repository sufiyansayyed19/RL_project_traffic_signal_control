"""
Traffic Intersection Environment for Reinforcement Learning
============================================================

This module implements a simulated 4-way traffic intersection as an RL environment.
The agent controls the traffic signal to minimize total vehicle waiting time.

MDP Formulation:
    States:  (queue_N, queue_S, queue_E, queue_W, current_phase)
             Each queue ∈ {0, 1, ..., max_queue}, phase ∈ {0, 1}
             Total states = (max_queue+1)^4 × 2

    Actions: {0: Keep current phase, 1: Switch phase}

    Rewards: -(total_waiting_cars) - switch_penalty × (switched)

    Transitions: Stochastic (random car arrivals each step)

Signal Phases:
    Phase 0: North-South GREEN, East-West RED
    Phase 1: East-West GREEN, North-South RED
"""

import numpy as np
from itertools import product


class TrafficIntersection:
    """
    Simulates a single 4-way traffic intersection with signal control.

    The environment models:
    - Four approach directions (N, S, E, W) each with a vehicle queue
    - Two signal phases (NS-green or EW-green)
    - Stochastic vehicle arrivals (Poisson-like)
    - Deterministic departures when signal is green
    - A switch penalty to prevent unrealistic rapid flickering

    Attributes:
        max_queue (int): Maximum number of cars per direction (queue cap).
        arrival_probs (dict): Probability of a new car arriving per direction per step.
        departure_rate (int): Number of cars that can depart per green step.
        switch_penalty (float): Reward penalty for switching the signal phase.
        max_steps (int): Maximum time steps per episode.
    """

    # Direction indices for readability
    NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
    DIRECTION_NAMES = ['North', 'South', 'East', 'West']

    # Signal phases
    PHASE_NS_GREEN = 0  # North-South has green light
    PHASE_EW_GREEN = 1  # East-West has green light

    # Actions
    ACTION_KEEP = 0
    ACTION_SWITCH = 1
    ACTION_NAMES = ['Keep', 'Switch']

    def __init__(self, config=None):
        """
        Initialize the traffic intersection environment.

        Args:
            config (dict, optional): Configuration parameters. Keys:
                - max_queue (int): Max cars per queue (default: 5)
                - arrival_probs (dict): Arrival probability per direction
                - departure_rate (int): Cars leaving per green step (default: 2)
                - switch_penalty (float): Penalty for switching (default: 2.0)
                - max_steps (int): Steps per episode (default: 100)
                - seed (int): Random seed for reproducibility
        """
        config = config or {}

        # Environment parameters
        self.max_queue = config.get('max_queue', 5)
        self.departure_rate = config.get('departure_rate', 2)
        self.switch_penalty = config.get('switch_penalty', 2.0)
        self.max_steps = config.get('max_steps', 100)

        # Arrival probabilities — slightly asymmetric for realism
        # (e.g., North-South is a busier road)
        default_arrivals = {
            'north': 0.4,
            'south': 0.35,
            'east': 0.3,
            'west': 0.25
        }
        self.arrival_probs = config.get('arrival_probs', default_arrivals)

        # Derived constants
        self.n_actions = 2  # Keep or Switch
        self.n_queue_levels = self.max_queue + 1  # 0 to max_queue
        self.n_states = (self.n_queue_levels ** 4) * 2  # queues × phases

        # State variables (set in reset())
        self.queues = None        # [queue_N, queue_S, queue_E, queue_W]
        self.current_phase = None  # 0 = NS-green, 1 = EW-green
        self.current_step = None
        self.total_cars_served = None

        # Random number generator
        seed = config.get('seed', None)
        self.rng = np.random.RandomState(seed)

        # Pre-compute green directions for each phase
        self._green_dirs = {
            self.PHASE_NS_GREEN: [self.NORTH, self.SOUTH],
            self.PHASE_EW_GREEN: [self.EAST, self.WEST]
        }

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            tuple: Initial state as (queue_N, queue_S, queue_E, queue_W, phase)
        """
        # Start with some cars already waiting (realistic initial condition)
        self.queues = [
            self.rng.randint(0, 3),  # North: 0-2 cars
            self.rng.randint(0, 3),  # South: 0-2 cars
            self.rng.randint(0, 3),  # East: 0-2 cars
            self.rng.randint(0, 3),  # West: 0-2 cars
        ]
        self.current_phase = self.rng.randint(0, 2)  # Random starting phase
        self.current_step = 0
        self.total_cars_served = 0

        return self.get_state()

    def step(self, action):
        """
        Execute one time step in the environment.

        Process order:
            1. Apply action (keep or switch phase)
            2. Process departures (green direction cars leave)
            3. Generate new arrivals (cars arrive at all directions)
            4. Calculate reward
            5. Check if episode is done

        Args:
            action (int): 0 = Keep current phase, 1 = Switch phase

        Returns:
            tuple: (next_state, reward, done, info)
                - next_state (tuple): New state after this step
                - reward (float): Reward signal
                - done (bool): Whether episode has ended
                - info (dict): Additional information for analysis
        """
        assert action in [0, 1], f"Invalid action: {action}. Must be 0 (Keep) or 1 (Switch)."

        self.current_step += 1
        switched = False

        # 1. Apply action
        if action == self.ACTION_SWITCH:
            self.current_phase = 1 - self.current_phase  # Toggle: 0↔1
            switched = True

        # 2. Process departures (green lanes lose cars)
        cars_departed = self._process_departures()

        # 3. Generate new arrivals (all lanes can receive cars)
        cars_arrived = self._generate_arrivals()

        # 4. Calculate reward
        reward = self._calculate_reward(switched)

        # 5. Check if episode is done
        done = self.current_step >= self.max_steps

        # Info dict for analysis
        info = {
            'step': self.current_step,
            'queues': list(self.queues),
            'phase': self.current_phase,
            'switched': switched,
            'cars_departed': cars_departed,
            'cars_arrived': cars_arrived,
            'total_waiting': sum(self.queues),
            'total_served': self.total_cars_served
        }

        return self.get_state(), reward, done, info

    def get_state(self):
        """
        Get the current state as a tuple.

        Returns:
            tuple: (queue_N, queue_S, queue_E, queue_W, phase)
        """
        return tuple(self.queues) + (self.current_phase,)

    def state_to_index(self, state):
        """
        Convert a state tuple to a unique integer index for Q-table.

        The encoding uses a mixed-radix system:
            index = q_N * (L^3 * 2) + q_S * (L^2 * 2) + q_E * (L * 2) + q_W * 2 + phase

        where L = max_queue + 1 (number of queue levels)

        Args:
            state (tuple): State as (queue_N, queue_S, queue_E, queue_W, phase)

        Returns:
            int: Unique index in [0, n_states)
        """
        q_n, q_s, q_e, q_w, phase = state
        L = self.n_queue_levels
        index = q_n * (L ** 3 * 2) + q_s * (L ** 2 * 2) + q_e * (L * 2) + q_w * 2 + phase
        return index

    def index_to_state(self, index):
        """
        Convert a state index back to a state tuple.

        Args:
            index (int): State index in [0, n_states)

        Returns:
            tuple: State as (queue_N, queue_S, queue_E, queue_W, phase)
        """
        L = self.n_queue_levels
        phase = index % 2
        index //= 2
        q_w = index % L
        index //= L
        q_e = index % L
        index //= L
        q_s = index % L
        index //= L
        q_n = index % L
        return (q_n, q_s, q_e, q_w, phase)

    def get_all_states(self):
        """
        Generate all possible states in the environment.

        Yields:
            tuple: Each possible state (queue_N, queue_S, queue_E, queue_W, phase)
        """
        queue_range = range(self.n_queue_levels)
        for q_n, q_s, q_e, q_w in product(queue_range, repeat=4):
            for phase in [0, 1]:
                yield (q_n, q_s, q_e, q_w, phase)

    def get_transition_prob(self, state, action):
        """
        Compute transition probabilities T(s'|s, a) for Value Iteration.

        This builds the explicit transition model by considering all
        possible arrival combinations and their probabilities.

        Args:
            state (tuple): Current state
            action (int): Action to take

        Returns:
            list of tuples: [(probability, next_state, reward), ...]
        """
        q_n, q_s, q_e, q_w, phase = state
        queues = [q_n, q_s, q_e, q_w]
        switched = False

        # Apply action
        if action == self.ACTION_SWITCH:
            phase = 1 - phase
            switched = True

        # Process departures (deterministic)
        green_dirs = self._green_dirs[phase]
        for d in green_dirs:
            queues[d] = max(0, queues[d] - self.departure_rate)

        # Generate all possible arrival combinations
        # Each direction either gets a car (+1) or doesn't (0)
        arrival_probs_list = [
            self.arrival_probs['north'],
            self.arrival_probs['south'],
            self.arrival_probs['east'],
            self.arrival_probs['west']
        ]

        transitions = []
        # Iterate over all 2^4 = 16 arrival combinations
        for arrivals in product([0, 1], repeat=4):
            # Calculate probability of this specific arrival combination
            prob = 1.0
            for d in range(4):
                if arrivals[d] == 1:
                    prob *= arrival_probs_list[d]
                else:
                    prob *= (1.0 - arrival_probs_list[d])

            if prob < 1e-10:
                continue  # Skip negligible probabilities

            # Compute next state queues with arrivals
            next_queues = list(queues)  # Start from post-departure queues
            for d in range(4):
                next_queues[d] = min(self.max_queue, next_queues[d] + arrivals[d])

            next_state = tuple(next_queues) + (phase,)

            # Calculate reward for this transition
            total_waiting = sum(next_queues)
            reward = -total_waiting
            if switched:
                reward -= self.switch_penalty

            transitions.append((prob, next_state, reward))

        return transitions

    def _process_departures(self):
        """
        Process car departures from green-light directions.

        Returns:
            int: Total number of cars that departed
        """
        green_dirs = self._green_dirs[self.current_phase]
        total_departed = 0

        for d in green_dirs:
            departed = min(self.queues[d], self.departure_rate)
            self.queues[d] -= departed
            total_departed += departed

        self.total_cars_served += total_departed
        return total_departed

    def _generate_arrivals(self):
        """
        Generate new car arrivals at all directions (stochastic).

        Each direction has an independent probability of receiving a new car.

        Returns:
            int: Total number of cars that arrived
        """
        total_arrived = 0
        prob_keys = ['north', 'south', 'east', 'west']

        for d in range(4):
            if self.rng.random() < self.arrival_probs[prob_keys[d]]:
                self.queues[d] = min(self.max_queue, self.queues[d] + 1)
                total_arrived += 1

        return total_arrived

    def _calculate_reward(self, switched):
        """
        Calculate the reward for the current state.

        Reward = -(total waiting cars) - switch_penalty × (switched)

        Args:
            switched (bool): Whether the signal was switched this step

        Returns:
            float: Reward value (always negative or zero)
        """
        total_waiting = sum(self.queues)
        reward = -total_waiting

        if switched:
            reward -= self.switch_penalty

        return reward

    def render(self):
        """
        Print a text visualization of the current intersection state.
        Useful for debugging and understanding agent behavior.
        """
        phase_str = "NS-GREEN" if self.current_phase == 0 else "EW-GREEN"
        q = self.queues

        print(f"\n{'='*40}")
        print(f"  Step: {self.current_step}  |  Phase: {phase_str}")
        print(f"{'='*40}")
        print(f"            N: {'#' * q[0]}{' ' * (self.max_queue - q[0])} ({q[0]})")
        print(f"               {'v' if self.current_phase == 0 else '|'}")
        print(f"  W: {'#' * q[3]}{' ' * (self.max_queue - q[3])} ({q[3]}) "
              f"[TL] "
              f"({q[2]}) {'#' * q[2]}{' ' * (self.max_queue - q[2])} :E")
        print(f"               {'|' if self.current_phase == 0 else '^'}")
        print(f"            S: {'#' * q[1]}{' ' * (self.max_queue - q[1])} ({q[1]})")
        print(f"{'-'*40}")
        print(f"  Total waiting: {sum(q)} | Served: {self.total_cars_served}")
        print(f"{'='*40}\n")

    def get_config(self):
        """
        Return the environment configuration (for reproducibility).

        Returns:
            dict: Configuration parameters
        """
        return {
            'max_queue': self.max_queue,
            'arrival_probs': self.arrival_probs,
            'departure_rate': self.departure_rate,
            'switch_penalty': self.switch_penalty,
            'max_steps': self.max_steps,
            'n_states': self.n_states,
            'n_actions': self.n_actions
        }
