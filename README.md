# Water-Jug-Problem-AI

# Software Description
The software is an AI player designed to solve the Water Jug problem using both: (1) uninformed (Breadth-First Search, BFS) and (2) informed (heuristic-based) search algorithms. Users provide two inputs: (1) the number of water jugs and (2) their respective water levels, which define the problem parameters. The software then produces a solution, showcasing the path to the goal state and the search tree.


# Uninformed Search (BFS):
Step 1: Starting with the given array of buckets, expand the root node and generate all possible action outcomes as child nodes. These actions include:

• Filling a bucket
• Draining a bucket
• Pouring from one bucket into another

When a parent node is expanded, if the new state has not been explored or is not already in the frontier, add the node to the frontier.

Step 2: The total number of possible actions depends on the problem specifics. For example, with three buckets of capacities 8, 5, and 3 (all initially empty), we have:

• 3 options from the "Fill a bucket" action
• 3 options from the "Drain a bucket" action
• For the "Pour from one bucket to another" action, there are (n - 1) * n possibilities, where n is the total number of buckets. In this case, (3 - 1) * 3 = 6 possibilities.

Step 3: After expanding all children from the root node, remove the root node from the frontier. Continue in a queue fashion (first-in-first-out) with the next node in the frontier.

Step 4: Repeat steps 1-3 until the frontier is empty (indicating no solution) or a solution is found.


# Informed Search (Heuristic-based):
Step 1: Given the array of buckets, consider all possible pairs of buckets.

Step 2: Begin a loop that continues until the goal state is found.

Step 3: For each iteration, if a bucket is empty, fill it.

Step 4: If a bucket is not empty, consider pouring from one bucket to another.

Step 5: After pouring into another bucket, check if the second bucket is full. If so, drain it.

Step 6: Calculate a heuristic that quantifies the absolute difference between the current water level in each bucket and the desired goal state. Based on this heuristic, prioritize actions that minimize this difference.

Step 7: Repeat steps 3-6 until the frontier is empty (indicating no solution) or a solution is found.
