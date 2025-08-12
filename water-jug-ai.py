import copy
import json
import heapq
from collections import deque


class Bucket:
    def __init__(self, name, size, filled):
        self.name = name
        self.size = size
        self.filled = filled

    def __repr__(self):
        return "{}".format(self.filled)

    def __str__(self):
        return "{}".format(self.filled)

    def drain(self):
        self.filled = 0

    def fill(self):
        self.filled = self.size

    def pour(self, other_bucket):
        unit_to_full = other_bucket.size - other_bucket.filled
        
        if self.filled == 0:
            pass
        elif unit_to_full == 0:
            pass
        else:
            if unit_to_full >= self.filled:
                other_bucket.filled = other_bucket.filled + self.filled
                self.filled = 0
            elif unit_to_full < self.filled:
                other_bucket.filled = other_bucket.size
                self.filled = self.filled - unit_to_full


class Node:
    def __init__(self, id, bucket_array, expansion_sequence, children, actions, removed, parent_node, cost=0):
        self.id = id
        self.bucket_array = bucket_array
        self.expansion_sequence = expansion_sequence
        self.children = children
        self.actions = actions
        self.removed = removed
        self.parent_node = parent_node
        self.cost = cost

    def __repr__(self):
        return "{}".format(self.bucket_array)

    def __str__(self):
        return "{}".format(self.bucket_array)
    
    def __lt__(self, other):
        return self.cost < other.cost

    def perform_action_and_return_buckets(self, bucket_array, selected_index, action, target_index):
        if target_index is None:
            if action == "Fill":
                bucket_array[selected_index].fill()
                return f"Fill bucket {selected_index}"
            elif action == "Drain":
                bucket_array[selected_index].drain()
                return f"Drain bucket {selected_index}"
        else:
            if action == "Pour":
                bucket_array[selected_index].pour(other_bucket=bucket_array[target_index])
                return f"Pour from bucket {selected_index} to bucket {target_index}"
        return ""


class Player:
    name = "Zyebot"
    group = "Zyetech"
    icon = "mdi-cloud"
    members = [
        ["God Emperor F. Nik", "17085309"]
    ]

    def __init__(self):
        self.algorithm = "bfs"  # Default algorithm
        self.problem = None  # Store problem for access in print_result

    def set_algorithm(self, algorithm):
        """Set the search algorithm to use: 'bfs', 'dfs', 'astar', or 'bidirectional'"""
        if algorithm in ["bfs", "dfs", "astar", "bidirectional"]:
            self.algorithm = algorithm
        else:
            raise ValueError("Algorithm must be 'bfs', 'dfs', 'astar', or 'bidirectional'")

    def generate_result(self, node, search_tree, explored_dictionary):
        node_path = [node]
        
        expansion_sequence = 1
        expansion_sequence_array = []
        
        # Finding path from Goal node to Root node
        while True:
            expansion_sequence_array.append(expansion_sequence)
            expansion_sequence += 1
            
            if node.parent_node is not None:
                node_path.append(node.parent_node)
                node = node.parent_node
            else:
                path = []
                actions = []
                # Reverse the path to get path from Root node to Goal node
                node_path.reverse()
                
                for i, node in enumerate(node_path):
                    path.append([b.filled for b in node.bucket_array])
                    if i > 0 and len(node.actions) > 0:
                        actions.append(node.actions[-1])
                
                i = 0
                while i < len(expansion_sequence_array):
                    for d in search_tree:
                        if d.get('id', 0) == node_path[i].id:
                            d.update((k, expansion_sequence_array[i]) for k, v in d.items() if v == -1)
                    i += 1
                
                self.print_result(found=True, explored_dictionary=explored_dictionary,
                                  search_tree=search_tree, path=path, actions=actions)
                
                return path, search_tree

    def check_found_goal_state(self, bucket_array, target):
        for bucket in bucket_array:
            if bucket.filled == target:
                return True
        return False

    def print_result(self, found, explored_dictionary, search_tree, path, actions=None):
        if found:
            print(f"\nAlgorithm: {self.algorithm.upper()}")
            print("\nExplored dictionary ({} states explored):".format(len(explored_dictionary)))
            if len(explored_dictionary) <= 20:
                # Convert dictionary keys to list for JSON serialization
                explored_list = list(explored_dictionary.keys())
                print(json.dumps(explored_list, indent=2))
            else:
                print("(Too many states to display - showing first 10)")
                explored_list = list(explored_dictionary.keys())[:10]
                print(json.dumps(explored_list, indent=2))
            
            print("\nSearch tree ({} total nodes):".format(len(search_tree)))
            if len(search_tree) <= 30:
                # Format search tree for better readability
                formatted_tree = []
                for node in search_tree:
                    formatted_node = {
                        "id": node["id"],
                        "state": node["state"],
                        "parent": node.get("parent"),
                        "expansion_seq": node.get("expansionsequence", -1),
                        "removed": node.get("removed", False),
                        "children": node.get("children", []),
                        "actions": node.get("actions", [])
                    }
                    if "cost" in node:
                        formatted_node["cost"] = node["cost"]
                    formatted_tree.append(formatted_node)
                print(json.dumps(formatted_tree, indent=2))
            else:
                print("(Too many nodes to display - showing first 15)")
                formatted_tree = []
                for node in search_tree[:15]:
                    formatted_node = {
                        "id": node["id"],
                        "state": node["state"],
                        "parent": node.get("parent"),
                        "expansion_seq": node.get("expansionsequence", -1),
                        "children": node.get("children", [])[:3] + (["..."] if len(node.get("children", [])) > 3 else [])
                    }
                    formatted_tree.append(formatted_node)
                print(json.dumps(formatted_tree, indent=2))
            
            print("\nPath ({} steps):".format(len(path) - 1))
            for i, state in enumerate(path):
                if i == 0:
                    print(f"  Initial: {state}")
                elif actions and i <= len(actions):
                    print(f"  Step {i}: {actions[i-1]} -> {state}")
                else:
                    print(f"  Step {i}: {state}")
            
            if self.problem:
                print("\nTarget: {} in any of the buckets".format(self.problem["target"]))
            print("\nSolution: Found")
        else:
            print(f"\nAlgorithm: {self.algorithm.upper()}")
            print("\nExplored dictionary ({} states explored):".format(len(explored_dictionary)))
            explored_list = list(explored_dictionary.keys())
            print(json.dumps(explored_list, indent=2))
            print("\nSearch tree ({} total nodes):".format(len(search_tree)))
            formatted_tree = []
            for node in search_tree:
                formatted_node = {
                    "id": node["id"],
                    "state": node["state"],
                    "parent": node.get("parent"),
                    "expansion_seq": node.get("expansionsequence", -1),
                    "children": node.get("children", [])
                }
                formatted_tree.append(formatted_node)
            print(json.dumps(formatted_tree, indent=2))
            print("\nSolution: Not found")

    def calculate_heuristic(self, bucket_array, target):
        """Calculate heuristic for A* search"""
        # Heuristic: minimum distance to target among all buckets
        min_distance = float('inf')
        for bucket in bucket_array:
            distance = abs(bucket.filled - target)
            if distance < min_distance:
                min_distance = distance
        
        # Also consider if we can reach target by combining buckets
        total = sum(b.filled for b in bucket_array)
        if total >= target:
            # Penalize states where we have too much water spread across buckets
            min_distance = min(min_distance, len(bucket_array) - 1)
        
        return min_distance

    def generate_children(self, parent_node, problem, node_id_counter, explored_dictionary):
        """Generate all possible child nodes from a parent node"""
        children = []
        
        # Action: Fill each bucket
        if problem["source"]:
            for i in range(len(parent_node.bucket_array)):
                if parent_node.bucket_array[i].filled < parent_node.bucket_array[i].size:
                    node_id_counter[0] += 1
                    child_node = copy.deepcopy(parent_node)
                    new_actions = list(child_node.actions)
                    action_desc = child_node.perform_action_and_return_buckets(
                        bucket_array=child_node.bucket_array, selected_index=i, 
                        action="Fill", target_index=None)
                    new_actions.append(action_desc)
                    
                    child_str = str([b.filled for b in child_node.bucket_array])
                    
                    if child_str not in explored_dictionary:
                        node = Node(id=node_id_counter[0], bucket_array=child_node.bucket_array,
                                  expansion_sequence=-1, children=[], actions=new_actions,
                                  removed=False, parent_node=parent_node,
                                  cost=parent_node.cost + 1)
                        children.append(node)
                        parent_node.children.append(node_id_counter[0])
        
        # Action: Drain each bucket
        if problem["sink"]:
            for i in range(len(parent_node.bucket_array)):
                if parent_node.bucket_array[i].filled > 0:
                    node_id_counter[0] += 1
                    child_node = copy.deepcopy(parent_node)
                    new_actions = list(child_node.actions)
                    action_desc = child_node.perform_action_and_return_buckets(
                        bucket_array=child_node.bucket_array, selected_index=i,
                        action="Drain", target_index=None)
                    new_actions.append(action_desc)
                    
                    child_str = str([b.filled for b in child_node.bucket_array])
                    
                    if child_str not in explored_dictionary:
                        node = Node(id=node_id_counter[0], bucket_array=child_node.bucket_array,
                                  expansion_sequence=-1, children=[], actions=new_actions,
                                  removed=False, parent_node=parent_node,
                                  cost=parent_node.cost + 1)
                        children.append(node)
                        parent_node.children.append(node_id_counter[0])
        
        # Action: Pour from one bucket to another
        for i in range(len(parent_node.bucket_array)):
            for j in range(len(parent_node.bucket_array)):
                if i != j and parent_node.bucket_array[i].filled > 0 and \
                   parent_node.bucket_array[j].filled < parent_node.bucket_array[j].size:
                    node_id_counter[0] += 1
                    child_node = copy.deepcopy(parent_node)
                    new_actions = list(child_node.actions)
                    action_desc = child_node.perform_action_and_return_buckets(
                        bucket_array=child_node.bucket_array, selected_index=i,
                        action="Pour", target_index=j)
                    new_actions.append(action_desc)
                    
                    child_str = str([b.filled for b in child_node.bucket_array])
                    
                    if child_str not in explored_dictionary:
                        node = Node(id=node_id_counter[0], bucket_array=child_node.bucket_array,
                                  expansion_sequence=-1, children=[], actions=new_actions,
                                  removed=False, parent_node=parent_node,
                                  cost=parent_node.cost + 1)
                        children.append(node)
                        parent_node.children.append(node_id_counter[0])
        
        return children

    def uninformed_search_bfs(self, problem, initial_node, search_tree):
        """Breadth-First Search implementation"""
        frontier = deque([initial_node])
        explored_dictionary = {str([b.filled for b in initial_node.bucket_array]): True}
        node_id_counter = [1]
        
        while frontier:
            current_node = frontier.popleft()
            
            # Check if goal state is reached
            if self.check_found_goal_state(current_node.bucket_array, problem["target"]):
                return self.generate_result(current_node, search_tree, explored_dictionary)
            
            # Generate children
            children = self.generate_children(current_node, problem, node_id_counter, explored_dictionary)
            
            for child in children:
                child_str = str([b.filled for b in child.bucket_array])
                explored_dictionary[child_str] = True
                frontier.append(child)
                
                search_tree.append({
                    "id": child.id,
                    "state": [b.filled for b in child.bucket_array],
                    "expansionsequence": child.expansion_sequence,
                    "children": child.children,
                    "actions": child.actions,
                    "removed": child.removed,
                    "parent": child.parent_node.id if child.parent_node else None
                })
        
        self.print_result(found=False, explored_dictionary=explored_dictionary,
                        search_tree=search_tree, path=[], actions=[])
        return [], search_tree

    def uninformed_search_dfs(self, problem, initial_node, search_tree):
        """Depth-First Search implementation"""
        frontier = [initial_node]  # Using list as stack (LIFO)
        explored_dictionary = {str([b.filled for b in initial_node.bucket_array]): True}
        node_id_counter = [1]
        
        while frontier:
            current_node = frontier.pop()  # Pop from end for LIFO behavior
            
            # Check if goal state is reached
            if self.check_found_goal_state(current_node.bucket_array, problem["target"]):
                return self.generate_result(current_node, search_tree, explored_dictionary)
            
            # Generate children
            children = self.generate_children(current_node, problem, node_id_counter, explored_dictionary)
            
            for child in children:
                child_str = str([b.filled for b in child.bucket_array])
                explored_dictionary[child_str] = True
                frontier.append(child)  # Append to end for LIFO behavior
                
                search_tree.append({
                    "id": child.id,
                    "state": [b.filled for b in child.bucket_array],
                    "expansionsequence": child.expansion_sequence,
                    "children": child.children,
                    "actions": child.actions,
                    "removed": child.removed,
                    "parent": child.parent_node.id if child.parent_node else None
                })
        
        self.print_result(found=False, explored_dictionary=explored_dictionary,
                        search_tree=search_tree, path=[], actions=[])
        return [], search_tree

    def astar_search(self, problem, initial_node, search_tree):
        """A* Search implementation with heuristic"""
        # Priority queue: (f_score, node)
        initial_node.cost = 0
        h_score = self.calculate_heuristic(initial_node.bucket_array, problem["target"])
        frontier = [(h_score, initial_node)]
        heapq.heapify(frontier)
        
        explored_dictionary = {}
        g_scores = {str([b.filled for b in initial_node.bucket_array]): 0}
        node_id_counter = [1]
        
        while frontier:
            _, current_node = heapq.heappop(frontier)
            
            current_str = str([b.filled for b in current_node.bucket_array])
            
            if current_str in explored_dictionary:
                continue
            
            explored_dictionary[current_str] = True
            
            # Check if goal state is reached
            if self.check_found_goal_state(current_node.bucket_array, problem["target"]):
                return self.generate_result(current_node, search_tree, explored_dictionary)
            
            # Generate children
            children = self.generate_children(current_node, problem, node_id_counter, {})
            
            for child in children:
                child_str = str([b.filled for b in child.bucket_array])
                
                if child_str not in explored_dictionary:
                    tentative_g = current_node.cost + 1
                    
                    if child_str not in g_scores or tentative_g < g_scores[child_str]:
                        g_scores[child_str] = tentative_g
                        child.cost = tentative_g
                        h = self.calculate_heuristic(child.bucket_array, problem["target"])
                        f = tentative_g + h
                        heapq.heappush(frontier, (f, child))
                        
                        search_tree.append({
                            "id": child.id,
                            "state": [b.filled for b in child.bucket_array],
                            "expansionsequence": child.expansion_sequence,
                            "children": child.children,
                            "actions": child.actions,
                            "removed": child.removed,
                            "parent": child.parent_node.id if child.parent_node else None,
                            "cost": f
                        })
        
        self.print_result(found=False, explored_dictionary=explored_dictionary,
                        search_tree=search_tree, path=[], actions=[])
        return [], search_tree

    def generate_goal_states(self, problem):
        """Generate all possible goal states (any bucket with target amount)"""
        goal_states = []
        for i in range(len(problem["size"])):
            bucket_array = []
            for j in range(len(problem["size"])):
                bucket = Bucket(name="{}".format(chr(ord('@') + (j + 1))),
                              size=problem["size"][j], filled=0)
                bucket_array.append(bucket)
            bucket_array[i].filled = problem["target"]
            goal_states.append(bucket_array)
        return goal_states

    def generate_parents(self, node, problem, node_id_counter, explored_dictionary):
        """Generate all possible parent nodes (reverse operations)"""
        parents = []
        
        # Reverse of Fill: Drain
        if problem["sink"]:
            for i in range(len(node.bucket_array)):
                if node.bucket_array[i].filled == node.bucket_array[i].size:
                    node_id_counter[0] += 1
                    parent_node = copy.deepcopy(node)
                    new_actions = list(parent_node.actions)
                    parent_node.bucket_array[i].drain()
                    action_desc = f"Drain bucket {i} (reverse of Fill)"
                    new_actions.append(action_desc)
                    
                    parent_str = str([b.filled for b in parent_node.bucket_array])
                    
                    if parent_str not in explored_dictionary:
                        new_node = Node(id=node_id_counter[0], bucket_array=parent_node.bucket_array,
                                      expansion_sequence=-1, children=[], actions=new_actions,
                                      removed=False, parent_node=node,
                                      cost=node.cost + 1)
                        parents.append(new_node)
        
        # Reverse of Drain: Fill
        if problem["source"]:
            for i in range(len(node.bucket_array)):
                if node.bucket_array[i].filled == 0:
                    node_id_counter[0] += 1
                    parent_node = copy.deepcopy(node)
                    new_actions = list(parent_node.actions)
                    parent_node.bucket_array[i].fill()
                    action_desc = f"Fill bucket {i} (reverse of Drain)"
                    new_actions.append(action_desc)
                    
                    parent_str = str([b.filled for b in parent_node.bucket_array])
                    
                    if parent_str not in explored_dictionary:
                        new_node = Node(id=node_id_counter[0], bucket_array=parent_node.bucket_array,
                                      expansion_sequence=-1, children=[], actions=new_actions,
                                      removed=False, parent_node=node,
                                      cost=node.cost + 1)
                        parents.append(new_node)
        
        # Reverse of Pour: Pour back (all possible reverse pours)
        for i in range(len(node.bucket_array)):
            for j in range(len(node.bucket_array)):
                if i != j:
                    # Try reversing a pour from j to i
                    node_id_counter[0] += 1
                    parent_node = copy.deepcopy(node)
                    new_actions = list(parent_node.actions)
                    
                    # Calculate how much could have been poured
                    original_parent = copy.deepcopy(parent_node)
                    parent_node.bucket_array[j].pour(parent_node.bucket_array[i])
                    
                    # Check if this creates a valid parent state
                    parent_str = str([b.filled for b in parent_node.bucket_array])
                    if parent_str not in explored_dictionary and parent_node.bucket_array != original_parent.bucket_array:
                        action_desc = f"Pour from bucket {j} to bucket {i} (reverse)"
                        new_actions.append(action_desc)
                        new_node = Node(id=node_id_counter[0], bucket_array=parent_node.bucket_array,
                                      expansion_sequence=-1, children=[], actions=new_actions,
                                      removed=False, parent_node=node,
                                      cost=node.cost + 1)
                        parents.append(new_node)
        
        return parents

    def bidirectional_search(self, problem, initial_node, search_tree):
        """Bidirectional Search implementation"""
        # Forward search from initial state
        forward_frontier = deque([initial_node])
        forward_explored = {str([b.filled for b in initial_node.bucket_array]): initial_node}
        
        # Backward search from goal states
        backward_frontier = deque()
        backward_explored = {}
        
        node_id_counter = [1]
        
        # Create goal nodes for backward search
        goal_states = self.generate_goal_states(problem)
        for goal_state in goal_states:
            node_id_counter[0] += 1
            goal_node = Node(id=node_id_counter[0], bucket_array=goal_state,
                           expansion_sequence=-1, children=[], actions=[],
                           removed=False, parent_node=None, cost=0)
            backward_frontier.append(goal_node)
            state_str = str([b.filled for b in goal_state])
            backward_explored[state_str] = goal_node
            
            search_tree.append({
                "id": goal_node.id,
                "state": [b.filled for b in goal_node.bucket_array],
                "expansionsequence": goal_node.expansion_sequence,
                "children": goal_node.children,
                "actions": ["GOAL STATE"],
                "removed": goal_node.removed,
                "parent": None,
                "direction": "backward"
            })
        
        # Alternating forward and backward search
        while forward_frontier and backward_frontier:
            # Forward step
            if forward_frontier:
                current_forward = forward_frontier.popleft()
                
                # Generate children for forward search
                children = self.generate_children(current_forward, problem, node_id_counter, {})
                
                for child in children:
                    child_str = str([b.filled for b in child.bucket_array])
                    
                    # Check if we've met the backward search
                    if child_str in backward_explored:
                        # Found connection! Build complete path
                        backward_node = backward_explored[child_str]
                        path, search_tree = self.connect_paths(child, backward_node, search_tree, 
                                                              forward_explored, backward_explored)
                        return path, search_tree
                    
                    if child_str not in forward_explored:
                        forward_explored[child_str] = child
                        forward_frontier.append(child)
                        
                        search_tree.append({
                            "id": child.id,
                            "state": [b.filled for b in child.bucket_array],
                            "expansionsequence": child.expansion_sequence,
                            "children": child.children,
                            "actions": child.actions,
                            "removed": child.removed,
                            "parent": child.parent_node.id if child.parent_node else None,
                            "direction": "forward"
                        })
            
            # Backward step
            if backward_frontier:
                current_backward = backward_frontier.popleft()
                
                # Generate parents for backward search
                parents = self.generate_parents(current_backward, problem, node_id_counter, {})
                
                for parent in parents:
                    parent_str = str([b.filled for b in parent.bucket_array])
                    
                    # Check if we've met the forward search
                    if parent_str in forward_explored:
                        # Found connection! Build complete path
                        forward_node = forward_explored[parent_str]
                        path, search_tree = self.connect_paths(forward_node, parent, search_tree,
                                                              forward_explored, backward_explored)
                        return path, search_tree
                    
                    if parent_str not in backward_explored:
                        backward_explored[parent_str] = parent
                        backward_frontier.append(parent)
                        
                        search_tree.append({
                            "id": parent.id,
                            "state": [b.filled for b in parent.bucket_array],
                            "expansionsequence": parent.expansion_sequence,
                            "children": parent.children,
                            "actions": parent.actions,
                            "removed": parent.removed,
                            "parent": parent.parent_node.id if parent.parent_node else None,
                            "direction": "backward"
                        })
        
        # No solution found
        all_explored = {**forward_explored, **backward_explored}
        self.print_result(found=False, explored_dictionary=all_explored,
                        search_tree=search_tree, path=[], actions=[])
        return [], search_tree

    def connect_paths(self, forward_node, backward_node, search_tree, forward_explored, backward_explored):
        """Connect forward and backward paths when they meet"""
        # Build forward path from start to meeting point
        forward_path = []
        forward_actions = []
        current = forward_node
        
        while current is not None:
            forward_path.append([b.filled for b in current.bucket_array])
            if current.actions and len(current.actions) > 0:
                forward_actions.append(current.actions[-1])
            current = current.parent_node
        
        forward_path.reverse()
        forward_actions.reverse()
        
        # Build backward path from meeting point to goal
        backward_path = []
        backward_actions = []
        current = backward_node
        
        # Skip the meeting point node
        if current.parent_node:
            current = current.parent_node
        
        while current is not None and len(current.actions) > 0 and current.actions[0] != "GOAL STATE":
            backward_path.append([b.filled for b in current.bucket_array])
            if current.actions:
                # Invert the backward action to get forward action
                action = current.actions[-1] if len(current.actions) > 0 else ""
                # Simple reversal - this is approximation
                if "Fill" in action and "reverse" in action:
                    backward_actions.append(action.split(" (")[0])
                elif "Drain" in action and "reverse" in action:
                    backward_actions.append(action.split(" (")[0])
                else:
                    backward_actions.append(action.replace(" (reverse)", ""))
            current = current.parent_node
        
        # Add final goal state if we have one
        if current and current.actions and current.actions[0] == "GOAL STATE":
            backward_path.append([b.filled for b in current.bucket_array])
        
        complete_path = forward_path + backward_path
        complete_actions = forward_actions[1:] if len(forward_actions) > 0 else []  # Skip first empty action
        complete_actions.extend(backward_actions)
        
        # Prepare result
        all_explored = {}
        for key, node in forward_explored.items():
            all_explored[key + " (F)"] = True
        for key, node in backward_explored.items():
            all_explored[key + " (B)"] = True
        
        print(f"\nAlgorithm: BIDIRECTIONAL")
        print(f"\nMeeting point: {forward_path[-1]}")
        print(f"Forward search explored: {len(forward_explored)} states")
        print(f"Backward search explored: {len(backward_explored)} states")
        print(f"Total explored: {len(all_explored)} states")
        
        self.print_result(found=True, explored_dictionary=all_explored,
                        search_tree=search_tree, path=complete_path, actions=complete_actions)
        
        return complete_path, search_tree

    def run(self, problem):
        """Main entry point to run the selected algorithm"""
        self.problem = problem  # Store for access in print_result
        bucket_array = []
        search_tree = []
        node_id = 1
        
        # Initialize buckets
        for i in range(len(problem["size"])):
            bucket = Bucket(name="{}".format(chr(ord('@') + (i + 1))),
                          size=problem["size"][i], filled=problem["filled"][i])
            bucket_array.append(bucket)
        
        # Create initial node
        initial_node = Node(id=node_id, bucket_array=bucket_array, expansion_sequence=1,
                          children=[], actions=[], removed=False, parent_node=None, cost=0)
        
        search_tree.append({
            "id": initial_node.id,
            "state": [b.filled for b in initial_node.bucket_array],
            "expansionsequence": initial_node.expansion_sequence,
            "children": initial_node.children,
            "actions": initial_node.actions,
            "removed": initial_node.removed,
            "parent": None
        })
        
        # Check if initial state is goal
        if self.check_found_goal_state(initial_node.bucket_array, problem["target"]):
            path = [[b.filled for b in initial_node.bucket_array]]
            self.print_result(found=True, explored_dictionary={},
                            search_tree=search_tree, path=path, actions=[])
            return path, search_tree
        
        # Run selected algorithm
        if self.algorithm == "bfs":
            return self.uninformed_search_bfs(problem, initial_node, search_tree)
        elif self.algorithm == "dfs":
            return self.uninformed_search_dfs(problem, initial_node, search_tree)
        elif self.algorithm == "astar":
            return self.astar_search(problem, initial_node, search_tree)
        elif self.algorithm == "bidirectional":
            return self.bidirectional_search(problem, initial_node, search_tree)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")


if __name__ == '__main__':
    problem = {
        "size": [8, 5, 3],
        "filled": [0, 0, 0],
        "source": True,
        "sink": True,
        "target": 4
    }
    
    print("=" * 60)
    print("WATER JUG PROBLEM SOLVER")
    print("=" * 60)
    print(f"Problem: Buckets of size {problem['size']}, target = {problem['target']}")
    print("=" * 60)
    
    # Test all four algorithms
    algorithms = ["bfs", "dfs", "astar", "bidirectional"]
    
    for algo in algorithms:
        print(f"\n{'=' * 60}")
        print(f"Running {algo.upper()} Algorithm")
        print("=" * 60)
        player = Player()
        player.set_algorithm(algo)
        path, tree = player.run(problem)
        print(f"\nNodes explored: {len(tree)}")
        if path:
            print(f"Solution found in {len(path) - 1} steps")