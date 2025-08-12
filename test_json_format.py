#!/usr/bin/env python3

# Import the classes directly from the file
import sys
sys.path.insert(0, '/Users/Zyepher/Developer/Water-Jug-Problem-AI')

# Now run a smaller problem to see the full JSON output
exec(open('/Users/Zyepher/Developer/Water-Jug-Problem-AI/water-jug-ai.py').read())

# Test with a smaller problem for cleaner output
small_problem = {
    "size": [3, 2],
    "filled": [0, 0],
    "source": True,
    "sink": True,
    "target": 1
}

print("\n" + "=" * 60)
print("TESTING JSON FORMATTED OUTPUT WITH SMALLER PROBLEM")
print("=" * 60)
print(f"Problem: Buckets of size {small_problem['size']}, target = {small_problem['target']}")
print("=" * 60)

player = Player()
player.set_algorithm("dfs")
path, tree = player.run(small_problem)
print(f"\nNodes explored: {len(tree)}")
if path:
    print(f"Solution found in {len(path) - 1} steps")