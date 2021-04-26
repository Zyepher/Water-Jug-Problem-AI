# Water-Jug-Problem-AI

# Software Description
Built an AI player to solve the puzzles. Implemented informed search (BFS) and uninformed search algorithms. The software takes two inputs (Σ water jugs &amp; water level for each jug) from the user to create the puzzle and output the solution by displaying the path and the search tree.


# Problem Description
To develop an AI player equipped with two different search algorithms which are informed search and uninformed search for water bucket puzzles. The program will take input that contains the information of the problem and will output the path and the search tree to the problem.


# Uninformed search (Bfs):
Step 1 - With the given array of buckets specified in the problem, expand the root node and return children for all possible actions. Everytime a parent node is expanded, if the state is not yet explored, append the node to the frontier.

The available actions are:
1. Fill a bucket
2. Drain a bucket
3. Pour from a bucket to another bucket

Step 2 - The total of possible actions depends on the specified problem. For example, three buckets with capacity of 8, 5, 3 where initially all buckets are empty, there are:
  • 3 possibilities from “Fill a bucket” action.
  • 3 possibilities from “Drain a bucket” action.
  • “Pour from a bucket to another bucket” action has the total of possibility of (n - 1) * n, where n is the total number of buckets. 
    In this example, there are (3 - 1) * 3 = 6 possibilities. The possibilities are:
      • A -> B
      • A -> C
      • B -> A
      • B -> C
      • C -> A
      • C -> B

Step 3 - After all the child is expanded from the root node, remove the root node from the frontier and proceed with the next iteration (Queue, first-in-first-out) with the first child node in the frontier.

Step 4 - Repeat until the frontier is empty (no solution) or until a solution is found.


# Informed search:
Step 1 -  With the given array of buckets specified in the problem, algorithmically choose one best combination of two buckets. From the two buckets, let's call the bucket with bigger capacity (size) as upper and the bucket with lower capacity (size) as lower. 
Step 2 -  Then, while loop with the condition of the goal state not found
Step 3 -  For every iteration, if the lower bucket is empty, fill the lower bucket.
Step 4 -  If the lower bucket is not empty, pour from lower bucket to upper bucket.
Step 5 -  After pouring to the upper bucket, check if the upper bucket is full. If it is full, drain the upper bucket.
Step 6 -  Repeat until the frontier is empty (no solution) or until a solution is found.
