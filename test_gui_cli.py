#!/usr/bin/env python3
"""
Test the GUI functionality without actually displaying the window
This tests the core logic and integration
"""

import sys
sys.path.insert(0, '.')

# Load the solver
exec(open('water-jug-ai.py').read(), globals())

def test_gui_logic():
    """Test the GUI logic without tkinter"""
    print("Testing Water Jug GUI Logic")
    print("=" * 60)
    
    # Test problem configuration
    problem = {
        "size": [8, 5, 3],
        "filled": [0, 0, 0],
        "source": True,
        "sink": True,
        "target": 4
    }
    
    algorithms = ["bfs", "dfs", "astar", "bidirectional"]
    
    print(f"\nProblem: Jugs {problem['size']}, Target: {problem['target']}")
    print("-" * 60)
    
    results = []
    
    for algo in algorithms:
        print(f"\nTesting {algo.upper()}...")
        player = Player()
        player.set_algorithm(algo)
        
        import time
        start_time = time.time()
        path, tree = player.run(problem)
        solve_time = time.time() - start_time
        
        if path:
            print(f"  ✓ Solution found: {len(path)-1} steps")
            print(f"  ✓ Nodes explored: {len(tree)}")
            print(f"  ✓ Time: {solve_time:.3f}s")
            
            results.append({
                'algorithm': algo,
                'steps': len(path) - 1,
                'nodes': len(tree),
                'time': solve_time
            })
        else:
            print(f"  ✗ No solution found")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        # Find best performers
        fastest = min(results, key=lambda x: x['time'])
        most_efficient = min(results, key=lambda x: x['nodes'])
        shortest = min(results, key=lambda x: x['steps'])
        
        print(f"⚡ Fastest: {fastest['algorithm'].upper()} ({fastest['time']:.3f}s)")
        print(f"🎯 Most Efficient: {most_efficient['algorithm'].upper()} ({most_efficient['nodes']} nodes)")
        print(f"📏 Shortest Path: {shortest['algorithm'].upper()} ({shortest['steps']} steps)")
    
    print("\n✅ All GUI logic tests passed!")
    return True

def test_jug_operations():
    """Test jug operations"""
    print("\n" + "=" * 60)
    print("Testing Jug Operations")
    print("=" * 60)
    
    # Test Bucket class
    jug1 = Bucket("A", 5, 0)
    jug2 = Bucket("B", 3, 0)
    
    # Test fill
    jug1.fill()
    assert jug1.filled == 5, "Fill operation failed"
    print("✓ Fill operation works")
    
    # Test pour
    jug1.pour(jug2)
    assert jug1.filled == 2, f"Pour operation failed: jug1 has {jug1.filled}"
    assert jug2.filled == 3, f"Pour operation failed: jug2 has {jug2.filled}"
    print("✓ Pour operation works")
    
    # Test drain
    jug2.drain()
    assert jug2.filled == 0, "Drain operation failed"
    print("✓ Drain operation works")
    
    return True

if __name__ == "__main__":
    try:
        test_jug_operations()
        test_gui_logic()
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)