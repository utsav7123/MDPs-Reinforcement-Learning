# MDPs-Reinforcement-Learning

## Overview

This repository contains an implementation of foundational algorithms in **Markov Decision Processes (MDP)** and **Reinforcement Learning (RL)**, using the classic **Gridworld** environment. The project is structured following the UC Berkeley CS188 "Pacman AI" assignments and supports both algorithmic experimentation and automated grading.

Core algorithms include:
- Value Iteration
- Prioritized Sweeping Value Iteration
- Q-Learning
- Approximate Q-Learning

The project is modular, easily extendable, and is designed for educational use.

---

## Directory Structure
```
.
├── analysis.py # Answers to conceptual questions (parameters, reasoning)
├── autograder.py # Automated test/grade harness
├── environment.py # Core environment base class
├── game.py # General game mechanics (state, agents)
├── gridworld.py # Gridworld environment, agent movement logic
├── graphicsGridworldDisplay.py # Graphical display for Gridworld
├── graphicsUtils.py # GUI utilities
├── grading.py # Helper functions for grade reporting
├── keyboardAgents.py # Agents controllable via keyboard
├── layout.py # Gridworld layout parsing and management
├── learningAgents.py # Abstract base classes for RL agents
├── mdp.py # Abstract MDP class
├── projectParams.py # Configurable global parameters
├── qlearningAgents.py # Q-Learning and Approximate Q-Learning agents
├── reinforcementTestClasses.py # RL autograder test logic
├── testClasses.py # Test harness classes
├── testParser.py # Test file parsing
├── textDisplay.py # Text-based state visualizations
├── textGridworldDisplay.py # ASCII/textual gridworld rendering
├── util.py # Utilities (Counter, PriorityQueue, etc.)
├── valueIterationAgents.py # Value Iteration & Prioritized Sweeping agents
├── MDP_RL.pdf # Assignment handout/instructions
└── ...
```

## Features

### Algorithms Implemented

- **Value Iteration**: Computes the optimal policy by iteratively improving value estimates for all states.
- **Prioritized Sweeping Value Iteration**: Accelerates convergence using a priority queue based on state value differences.
- **Q-Learning**: Model-free RL agent, learning state-action values via direct interaction with the environment.
- **Approximate Q-Learning**: Q-Learning using feature-based function approximation for large or continuous state spaces.

### Visualization

- **Graphical Display**: Interactive graphical interface using Tkinter (see `graphicsGridworldDisplay.py`).
- **Text Display**: Console-based ASCII/textual interface for running in headless environments.

### Automated Grading

- **Autograder**: Supports running test cases and evaluating correctness (`autograder.py`, `reinforcementTestClasses.py`).
- **Analysis Questions**: Parameter tuning and conceptual reasoning go in `analysis.py`.

---

## How to Run

### Prerequisites

- **Python 3.6+** (not compatible with Python 2)
- No third-party packages required (uses standard library + Tkinter for GUI)

### Usage

#### Run Value Iteration Agent:

```bash
python gridworld.py -a value
```
![Screenshot](images.png)

Run Q-Learning Agent:
```bash
python gridworld.py -a q
```

# File/Module Descriptions

   - gridworld.py: Main entry point for running experiments; handles environment and agent setup.

   - valueIterationAgents.py: Contains```ValueIterationAgent``` and ```PrioritizedSweepingValueIterationAgent```.

   - qlearningAgents.py: Implements ```QLearningAgent``` and ```ApproximateQAgent```.

   - learningAgents.py: Abstract base classes for value-based and RL agents.

   - mdp.py: Defines the MDP interface used by all agents.

   - util.py: Frequently used data structures: ```Counter``` (dict with default 0), ```PriorityQueue```, ```Experiences``` for offline RL testing, etc.

   - graphicsGridworldDisplay.py/textGridworldDisplay.py: For visualizing agent actions and value functions.

   - testClasses.py, reinforcementTestClasses.py: Classes for automated test case management and validation.

   - analysis.py: Fill in responses for analytical/conceptual assignment questions.

   - autograder.py: Runs all required tests, produces feedback and score.

# Example: Running Value Iteration
```bash
python gridworld.py -a value -k 100 -d 0.9 --noise 0.2
```
- ```-a``` value: Use Value Iteration agent

- ```-k 100```: Run for 100 iterations

- ```-d 0.9```: Set discount factor

- ```--noise 0.2```: Transition noise (stochasticity)
# License
Educational Use Only (UC Berkeley CS188 Pacman AI).
# Credits

Original framework and assignment developed by
John DeNero, Dan Klein, and the UC Berkeley CS188 Team.


