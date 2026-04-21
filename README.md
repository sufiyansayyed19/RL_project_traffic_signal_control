# Traffic Signal Control using Reinforcement Learning

## Overview
This project implements an intelligent traffic signal control system using Reinforcement Learning. An RL agent learns to optimally switch traffic signal phases at a 4-way intersection to **minimize total vehicle waiting time**.

Two RL algorithms are compared:
- **Q-Learning** (Model-Free, Temporal Difference)
- **Value Iteration** (Model-Based, Dynamic Programming)

Both are benchmarked against two baselines:
- **Random Agent** (lower bound)
- **Fixed Timer** (traditional approach)

## MDP Formulation
| Component | Definition |
|-----------|-----------|
| **States** | `(queue_N, queue_S, queue_E, queue_W, phase)` — 2,592 total states |
| **Actions** | `{Keep, Switch}` — 2 actions |
| **Reward** | `-(total_waiting_cars) - switch_penalty` |
| **Transitions** | Stochastic (random car arrivals) |
| **Discount (γ)** | 0.95 |

## Project Structure
```
RL_project/
├── environment.py       # Traffic intersection simulation (MDP)
├── agents.py            # Q-Learning, Value Iteration, Baselines
├── training.py          # Training loops and evaluation
├── analysis.py          # Performance metrics and comparison
├── visualization.py     # 6 professional plots
├── main.py              # Pipeline orchestrator
├── dashboard.html       # Visual dashboard representation of the results
├── requirements.txt     # Dependencies
├── README.md            # This file
├── docs/                # Detailed project documentation
│   ├── 1_Problem_Statement.md
│   ├── 2_Proposed_Solution.md
│   ├── 3_Algorithms_and_Equations.md
│   └── 4_Step_by_Step_Implementation.md
└── results/             # Output directory (auto-generated)
    ├── learning_curves.png
    ├── algorithm_comparison.png
    ├── queue_dynamics.png
    ├── action_distribution.png
    ├── vi_convergence.png
    ├── summary_dashboard.png
    └── performance_report.txt
```

## Setup & Run

### 1. Create virtual environment
```bash
python -m venv venv
```

### 2. Activate virtual environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python main.py
```

All standard outputs will be saved in the `results/` folder.

### 5. View Dashboard
You can simply double-click `dashboard.html` in your file explorer to see a full, stylized web dashboard of the results. No server needed!

### 6. Read Detailed Documentation
For an academic explanation, please read the markdown files located in the `docs/` folder, which explain the math, equations, and logic behind the implementations mapping to the syllabus.

## Algorithms

### Q-Learning
- **Type**: Model-free, off-policy TD learning
- **Update**: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`
- **Exploration**: ε-greedy with decay (1.0 → 0.01)
- **Practical**: Can learn online without knowing traffic dynamics

### Value Iteration
- **Type**: Model-based Dynamic Programming
- **Update**: `V(s) ← max_a Σ T(s'|s,a)[R + γV(s')]`
- **Requires**: Complete transition model
- **Guarantees**: Optimal policy (theoretical best)

## Results
After running `main.py`, check the `results/` folder for:
- 6 visualization plots showing training progress and comparisons
- A text performance report with detailed metrics

## Dependencies
- Python 3.8+
- NumPy
- Matplotlib
- Pandas
- Seaborn

## Syllabus Alignment
| Module | Topic | Coverage |
|--------|-------|----------|
| Module 1 | RL Concepts | Agent-environment loop, states, actions, rewards |
| Module 3 | MDP Formulation | State/action/reward/transition design |
| Module 4 | Dynamic Programming | Value Iteration implementation |
| Module 5 | Temporal Difference | Q-Learning implementation |
| Module 6 | Applications | Real-world traffic signal control |
