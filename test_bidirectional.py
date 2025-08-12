#!/usr/bin/env python3

# Test bidirectional search with smaller problem
exec(open('/Users/Zyepher/Developer/Water-Jug-Problem-AI/water-jug-ai.py').read())

# Small problem for cleaner output
small_problem = {
    "size": [5, 3],
    "filled": [0, 0],
    "source": True,
    "sink": True,
    "target": 4
}

print("\n" + "=" * 60)
print("TESTING BIDIRECTIONAL SEARCH")
print("=" * 60)
print(f"Problem: Buckets of size {small_problem['size']}, target = {small_problem['target']}")
print("=" * 60)

player = Player()
player.set_algorithm("bidirectional")
path, tree = player.run(small_problem)

if path:
    print(f"\nSolution found in {len(path) - 1} steps")
    print(f"Total nodes in search tree: {len(tree)}")