# Water Jug Problem AI Solver

## Overview

This project implements an AI-powered solver for the classic Water Jug Problem using four different search algorithms:
1. **Breadth-First Search (BFS)** - Uninformed search that guarantees optimal solution
2. **Depth-First Search (DFS)** - Uninformed search that finds a solution quickly but not necessarily optimal
3. **A* Search** - Informed search using heuristics for efficient pathfinding
4. **Bidirectional Search** - Simultaneous search from start and goal states for improved efficiency

The Water Jug Problem involves finding a sequence of operations to measure out a specific amount of water using jugs of different capacities. Operations include filling jugs from a source, draining jugs to a sink, and pouring water between jugs.

## Problem Definition

Given:
- Multiple water jugs with specific capacities
- A target amount of water to measure
- Allowed operations: Fill, Drain, Pour

Goal: Find a sequence of operations to get exactly the target amount in any jug.

## Algorithms

### 1. Breadth-First Search (BFS)
**Type:** Uninformed Search  
**Data Structure:** Queue (FIFO)  
**Strategy:** Explores all states level by level

**Characteristics:**
- Guarantees finding the shortest path (optimal solution)
- Complete: Will find a solution if one exists
- Time Complexity: O(b^d) where b is branching factor and d is depth
- Space Complexity: O(b^d)

**Implementation Details:**
- Uses a deque for efficient queue operations
- Explores states in the order they are discovered
- Maintains explored dictionary to avoid revisiting states

### 2. Depth-First Search (DFS)
**Type:** Uninformed Search  
**Data Structure:** Stack (LIFO)  
**Strategy:** Explores deeply before backtracking

**Characteristics:**
- May find a solution quickly but not necessarily optimal
- Complete in finite state spaces
- Time Complexity: O(b^m) where m is maximum depth
- Space Complexity: O(bm)

**Implementation Details:**
- Uses a list as a stack
- Explores one branch completely before backtracking
- Can find solutions with fewer explored states but longer paths

### 3. A* Search
**Type:** Informed Search  
**Data Structure:** Priority Queue (Min-Heap)  
**Strategy:** Uses heuristic function to guide search

**Characteristics:**
- Finds optimal solution when using admissible heuristic
- More efficient than uninformed searches
- Time Complexity: O(b^d) but typically much better in practice
- Space Complexity: O(b^d)

**Heuristic Function:**
The implementation uses a custom heuristic that calculates:
- Minimum distance from any bucket's current water level to the target
- Considers the possibility of combining water from multiple buckets
- Admissible (never overestimates the actual cost)

**Implementation Details:**
- Uses heapq for priority queue operations
- f(n) = g(n) + h(n) where:
  - g(n) = actual cost from start to current node
  - h(n) = heuristic estimate from current node to goal
- Maintains g-scores dictionary for optimal path tracking

### 4. Bidirectional Search
**Type:** Uninformed Search (Optimized)  
**Data Structure:** Two Queues (Forward and Backward)  
**Strategy:** Searches simultaneously from start and goal states

**Characteristics:**
- Reduces search space by meeting in the middle
- Can find solutions faster than unidirectional searches
- Time Complexity: O(b^(d/2)) - significantly better than BFS
- Space Complexity: O(b^(d/2))

**Implementation Details:**
- **Forward Search:** Starts from initial state, generates children
- **Backward Search:** Starts from all possible goal states, generates parents
- **Meeting Point:** When a state is found in both searches, path is complete
- **Reverse Operations:**
  - Fill ↔ Drain (reversible operations)
  - Pour operations require careful state calculation
- **Path Reconstruction:** Combines forward path and reversed backward path

**Advantages:**
- Dramatically reduces search space for deep solutions
- Particularly effective when there are multiple goal states
- Can find solutions with fewer total node expansions

**When to Use:**
- Best for problems with well-defined goal states
- Effective when the branching factor is high
- Ideal when the solution depth is large

## Code Structure

### Classes

#### `Bucket`
Represents a water jug with:
- `size`: Maximum capacity
- `filled`: Current water amount
- Methods: `fill()`, `drain()`, `pour(other_bucket)`

#### `Node`
Represents a state in the search tree with:
- `bucket_array`: Current state of all buckets
- `parent_node`: Reference to parent for path reconstruction
- `actions`: List of actions taken to reach this state
- `cost`: Path cost (for A* search)

#### `Player`
Main solver class with:
- `set_algorithm(algorithm)`: Choose search algorithm ("bfs", "dfs", "astar", or "bidirectional")
- `run(problem)`: Execute the selected algorithm
- Algorithm-specific methods for each search type
- Helper methods for state generation and goal checking
- Special methods for bidirectional search:
  - `generate_goal_states()`: Creates all possible goal configurations
  - `generate_parents()`: Generates parent states (reverse operations)
  - `connect_paths()`: Merges forward and backward paths

## Usage

### Basic Example

```python
from water_jug_ai import Player

# Define the problem
problem = {
    "size": [8, 5, 3],      # Jug capacities
    "filled": [0, 0, 0],    # Initial amounts
    "source": True,         # Can fill from source
    "sink": True,           # Can drain to sink
    "target": 4             # Target amount
}

# Create solver and select algorithm
player = Player()
player.set_algorithm("bidirectional")  # Options: "bfs", "dfs", "astar", "bidirectional"

# Solve the problem
path, search_tree = player.run(problem)
```

### Running All Algorithms

```python
algorithms = ["bfs", "dfs", "astar", "bidirectional"]

for algo in algorithms:
    player = Player()
    player.set_algorithm(algo)
    path, tree = player.run(problem)
    print(f"{algo.upper()}: Found solution in {len(path)-1} steps")
    print(f"Nodes explored: {len(tree)}")
```

### Command Line Execution

```bash
python3 water-jug-ai.py
```

This will run all four algorithms on the default problem and display:
- Solution path with actions
- Number of states explored
- Search tree visualization (for small problems)
- Performance comparison
- For bidirectional search: meeting point and states explored in each direction

## Performance Comparison

For the classic 3-jug problem (8, 5, 3 liters) with target 4:

| Algorithm | States Explored | Solution Length | Optimality | Notes |
|-----------|----------------|-----------------|------------|-------|
| BFS       | ~96            | 6 steps         | Optimal    | Exhaustive level-by-level search |
| DFS       | ~28            | 7 steps         | Non-optimal| Quick but may find longer path |
| A*        | ~57            | 6 steps         | Optimal    | Heuristic-guided efficiency |
| Bidirectional | ~40 forward + ~51 backward | 6-7 steps | Near-optimal | Meets in the middle |

**Key Observations:**
- BFS guarantees the shortest solution but explores many states
- DFS finds a solution with fewer explorations but the path may be longer
- A* balances efficiency and optimality using heuristic guidance
- Bidirectional search can reduce the search space significantly, especially for deeper solutions

## Features

### State Space Generation
- Generates all valid actions from any state
- Prevents invalid operations (e.g., filling a full jug)
- Efficient duplicate state detection
- Bidirectional search generates both forward children and backward parents

### Path Reconstruction
- Tracks parent references for complete path reconstruction
- Records specific actions taken at each step
- Provides human-readable action descriptions
- Bidirectional search merges forward and backward paths at meeting point

### Visualization
- Displays solution path with state transitions
- Shows search tree structure (for small problems) in JSON format
- Prints explored states dictionary
- Provides performance metrics
- For bidirectional search: shows meeting point and directional statistics

## Customization

### Adding New Problems

```python
custom_problem = {
    "size": [10, 7, 4, 3],     # 4 jugs of different sizes
    "filled": [0, 0, 0, 0],    # All start empty
    "source": True,
    "sink": True,
    "target": 6                 # Find 6 liters
}
```

### Modifying the Heuristic

To use a different heuristic for A*, modify the `calculate_heuristic` method in the `Player` class:

```python
def calculate_heuristic(self, bucket_array, target):
    # Your custom heuristic logic here
    return estimated_cost
```

## Requirements

- Python 3.6+
- Standard library only (no external dependencies)
  - `copy`: Deep copying for state generation
  - `json`: JSON formatting for output
  - `heapq`: Priority queue for A* search
  - `collections.deque`: Efficient queue for BFS and Bidirectional Search

## Installation

```bash
git clone https://github.com/yourusername/Water-Jug-Problem-AI.git
cd Water-Jug-Problem-AI
python3 water-jug-ai.py
```

## Theory and Applications

### Real-World Applications
- Resource allocation problems
- Container loading optimization
- Chemical mixing procedures
- Network flow problems
- Path planning in robotics

### Educational Value
- Demonstrates fundamental AI search algorithms
- Illustrates trade-offs between different search strategies
- Shows the importance of heuristics in informed search
- Demonstrates bidirectional search optimization
- Provides hands-on experience with state space exploration

## Algorithm Selection Guide

Choose the right algorithm based on your needs:

| Use Case | Recommended Algorithm | Reason |
|----------|----------------------|---------|
| Need optimal solution | BFS or A* | Guarantees shortest path |
| Large state space | A* or Bidirectional | More efficient exploration |
| Quick solution (any) | DFS | Finds a solution fast |
| Deep solutions | Bidirectional | Reduces exponential growth |
| Educational purposes | All | Compare different approaches |

## Future Enhancements

Potential improvements to consider:
- [x] Bidirectional search implementation
- [ ] Interactive GUI for visualization
- [ ] Support for constraints (e.g., limited operations)
- [ ] Parallel search algorithms
- [ ] Machine learning for heuristic optimization
- [ ] Support for fractional amounts
- [ ] Cost-based operations (different costs for different actions)
- [ ] Iterative deepening search
- [ ] Memory-bounded search variants

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Developed by Zyetech - God Emperor F. Nik

## Acknowledgments

- Classic AI problem from computer science literature
- Inspired by Russell & Norvig's "Artificial Intelligence: A Modern Approach"
- Search algorithm implementations based on standard AI textbook approaches
- Bidirectional search inspired by Pohl's work on bi-directional heuristic search