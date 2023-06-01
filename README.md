# Water-Jug-Problem-AI

# Software Description
The software is designed to solve the Water Jug problem through an AI player using both informed and uninformed search algorithms. It takes two inputs from the user: the number of water jugs and their respective water levels, which help define the problem. The software then provides a solution by displaying the path to the goal state and the search tree.


# Uninformed Search (BFS):
Step 1: Given the array of buckets specified in the problem, expand the root node and generate children for all possible actions. The available actions are: The available actions are:
• Fill a bucket
• Drain a bucket
• Pour from a bucket to another bucket

When expanding a parent node, if the state has not been explored or is not already in the frontier, add the node to the frontier.

Step 2: The total number of possible actions depends on the specifics of the problem. For instance, consider three buckets with capacities of 8, 5, and 3 where initially all buckets are empty. The possible actions are:

• 3 (THREE) options from the "Fill a bucket" action
• 3 (THREE) options from the "Drain a bucket" action
• "Pour from a bucket to another bucket" action has a total of (n - 1) * n possibilities, where n is the total number of buckets. For this case, there are (3 - 1) * 3 = 6 possibilities.

Step 3: After expanding all children from the root node, remove the root node from the frontier. Then, in a queue fashion (first-in-first-out), proceed with the next node in the frontier.

Step 4: Repeat steps 1-3 until the frontier is empty (indicating no solution) or a solution is found.


# Informed Search (Heuristic-based):
Step 1: Given the array of buckets, consider all possible pairs of buckets.

Step 2: While the goal state has not been found, iterate through the loop.

Step 3: For each iteration, if a bucket is empty, fill it.

Step 4: If a bucket is not empty, consider pouring from one bucket to another.

Step 5: After pouring into another bucket, check if that bucket is full. If it is, drain the bucket.

Step 6: Calculate a heuristic that considers the absolute difference between the current water level in each bucket and the desired goal state. Use this heuristic to decide which action to take next.

Step 7: Repeat steps 3-6 until the frontier is empty (indicating no solution) or a solution is found.
