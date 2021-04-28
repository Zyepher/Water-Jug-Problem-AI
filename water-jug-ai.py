import copy
import pprint


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
    def __init__(self, id, bucket_array, expansion_sequence, children, actions, removed, parent_node):
        self.id = id
        self.bucket_array = bucket_array
        self.expansion_sequence = expansion_sequence
        self.children = children
        self.actions = actions
        self.removed = removed
        self.parent_node = parent_node

    def __repr__(self):
        return "{}".format(self.bucket_array)

    def __str__(self):
        return "{}".format(self.bucket_array)

    def perform_action_and_return_buckets(self, bucket_array, selected_index, action, target_index):
        action_performed = ""

        if target_index is None:
            if action == "Fill":
                action_performed = bucket_array[selected_index].fill()

            elif action == "Drain":
                action_performed = bucket_array[selected_index].drain()
        else:
            # Action is pouring
            if action == "Pour":
                action_performed = bucket_array[selected_index].pour(other_bucket=bucket_array[target_index])

        return bucket_array


class Player:
    name = "Zyebot"
    group = "Zyetech"
    icon = "mdi-cloud"
    members = [
        ["God Emperor F. Nik", "17085309"]
    ]

    informed = True

    def __init__(self):
        pass

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
                # Reverse the path to get path from Root node to Goal node
                node_path.reverse()

                for node in node_path:
                    path.append(node.bucket_array)

                i = 0
                while i < len(expansion_sequence_array):
                    for d in search_tree:
                        if d.get('id', 0) == node_path[i].id:
                            d.update((k, expansion_sequence_array[i]) for k, v in d.items() if v == -1)
                    i += 1

                self.print_result(found=True, explored_dictionary=explored_dictionary,
                                  search_tree=search_tree, path=path)

                return path, search_tree

    def check_found_goal_state(self, bucket_array, target):
        for bucket in bucket_array:
            if bucket.filled == target:
                return True
        return False

    def print_result(self, found, explored_dictionary, search_tree, path):
        if found:
            print("\nExplored dictionary ({} states explored):".format(len(explored_dictionary)))
            pprint.pprint(explored_dictionary)
            print("\nSearch tree ({} total nodes):".format(len(search_tree)))
            pprint.pprint(search_tree)
            print("\nPath ({} sequence of actions):\n{}".format(len(path) - 1, path))
            print("\nTarget: {} in any of the buckets".format(problem["target"]))
            print("\nSolution: Found")
        else:
            print("\nExplored dictionary ({} states explored):".format(len(explored_dictionary)))
            pprint.pprint(explored_dictionary)
            print("\nSearch tree ({} total nodes):".format(len(search_tree)))
            pprint.pprint(search_tree)
            print("\nSolution: Not found")

    def run(self, problem):
        # this function should return the path and the search_tree
        bucket_array = []
        frontier = []
        path = []
        explored_dictionary = {}
        search_tree = []
        node_id = 1

        i = 0
        while i < len(problem["size"]):
            bucket = Bucket(name="{}".format(chr(ord('@') + (i + 1))),
                            size=problem["size"][i], filled=problem["filled"][i])
            bucket_array.append(bucket)
            i += 1

        # Instantiate the root of the node
        parent_node = Node(id=node_id, bucket_array=bucket_array, expansion_sequence=1, children=[],
                           actions=[], removed=False, parent_node=None)

        search_tree.append(
            {
                "id": parent_node.id,
                "state": parent_node.bucket_array,
                "expansionsequence": parent_node.expansion_sequence,
                "children": parent_node.children,
                "actions": parent_node.actions,
                "removed": parent_node.removed,
                "parent": parent_node.parent_node
            })

        frontier.append(parent_node)

        # Goal test in case initial state is the goal
        found_goal = self.check_found_goal_state(parent_node.bucket_array, problem["target"])
        if found_goal:
            path = [parent_node.bucket_array]
            self.print_result(found=True, explored_dictionary=explored_dictionary,
                              search_tree=search_tree, path=path)
            return path, search_tree

        if self.informed:
            path, search_tree = self.informed_search(problem=problem, frontier=frontier, path=path,
                                                     explored_dictionary=explored_dictionary, search_tree=search_tree,
                                                     node_id=node_id, found_goal=found_goal)
        else:
            path, search_tree = self.uninformed_search_bfs(problem=problem, frontier=frontier, path=path,
                                                           explored_dictionary=explored_dictionary,
                                                           search_tree=search_tree, node_id=node_id,
                                                           found_goal=found_goal)
        return path, search_tree


    def uninformed_search_bfs(self, problem, frontier, path, explored_dictionary, search_tree, node_id, found_goal):
        while not found_goal:
            # Action is fill
            if problem["source"]:
                bucket_index = 0
                while bucket_index < len(frontier[0].bucket_array):
                    node_id += 1
                    child_node = copy.deepcopy(frontier[0])

                    child_node.bucket_array = child_node.perform_action_and_return_buckets(
                        bucket_array=child_node.bucket_array, selected_index=bucket_index, action="Fill",
                        target_index=None)

                    frontier[0].children.append(node_id)

                    node = Node(id=node_id, bucket_array=child_node.bucket_array, expansion_sequence=-1, children=[],
                                actions=child_node.actions, removed=None, parent_node=frontier[0])

                    childStr = '{}'.format(child_node)

                    if childStr in explored_dictionary:
                        node.removed = True
                    else:
                        explored_dictionary.update({'{}'.format(child_node): True})
                        node.removed = False
                        found_goal = self.check_found_goal_state(child_node.bucket_array, problem["target"])
                        frontier.append(node)
                    search_tree.append(
                        {
                            "id": node.id,
                            "state": node.bucket_array,
                            "expansionsequence": node.expansion_sequence,
                            "children": node.children,
                            "actions": node.actions,
                            "removed": node.removed,
                            "parent": node.parent_node.id
                        })

                    if found_goal:
                        path, search_tree = self.generate_result(node=node, search_tree=search_tree,
                                                                 explored_dictionary=explored_dictionary)
                        return path, search_tree

                    bucket_index += 1

            # Action is pour from a bucket to another bucket
            # NOTE:-
            # total pouring possibilities = (n - 1) * n, where n = total number of buckets

            bucket_index = 0
            while bucket_index < ((len(frontier[0].bucket_array)-1)*len(frontier[0].bucket_array)):
                target_index = 0
                while (bucket_index + 1) < len(frontier[0].bucket_array):
                    node_id += 1
                    child_node = copy.deepcopy(frontier[0])

                    if target_index == bucket_index:
                        child_node.bucket_array = child_node.perform_action_and_return_buckets(
                            bucket_array=child_node.bucket_array, selected_index=bucket_index, action="Pour",
                            target_index=target_index+1)
                    else:
                        child_node.bucket_array = child_node.perform_action_and_return_buckets(
                            bucket_array=child_node.bucket_array, selected_index=bucket_index, action="Pour",
                            target_index=target_index)

                    frontier[0].children.append(node_id)

                    node = Node(id=node_id, bucket_array=child_node.bucket_array, expansion_sequence=-1,
                                children=[], actions=child_node.actions, removed=None, parent_node=frontier[0])

                    childStr = '{}'.format(child_node)

                    if childStr in explored_dictionary:
                        node.removed = True
                    else:
                        explored_dictionary.update({'{}'.format(child_node): True})
                        node.removed = False
                        found_goal = self.check_found_goal_state(child_node.bucket_array, problem["target"])
                        frontier.append(node)

                    search_tree.append(
                        {
                            "id": node.id,
                            "state": node.bucket_array,
                            "expansionsequence": node.expansion_sequence,
                            "children": node.children,
                            "actions": node.actions,
                            "removed": node.removed,
                            "parent": node.parent_node.id
                        })

                    if found_goal:
                        path, search_tree = self.generate_result(node=node, search_tree=search_tree,
                                                                 explored_dictionary=explored_dictionary)
                        return path, search_tree

                    bucket_index += 1

                bucket_index += 1

            # Action is drain
            if problem["sink"]:
                bucket_index = 0
                while bucket_index < len(frontier[0].bucket_array):
                    node_id += 1
                    child_node = copy.deepcopy(frontier[0])

                    child_node.bucket_array = child_node.perform_action_and_return_buckets(
                        bucket_array=child_node.bucket_array, selected_index=bucket_index, action="Drain",
                        target_index=None)

                    frontier[0].children.append(node_id)

                    node = Node(id=node_id, bucket_array=child_node.bucket_array, expansion_sequence=-1, children=[],
                                actions=child_node.actions, removed=None, parent_node=frontier[0])

                    childStr = '{}'.format(child_node)

                    if childStr in explored_dictionary:
                        node.removed = True
                    else:
                        explored_dictionary.update({'{}'.format(child_node): True})
                        node.removed = False
                        found_goal = self.check_found_goal_state(child_node.bucket_array, problem["target"])
                        frontier.append(node)

                    search_tree.append(
                        {
                            "id": node.id,
                            "state": node.bucket_array,
                            "expansionsequence": node.expansion_sequence,
                            "children": node.children,
                            "actions": node.actions,
                            "removed": node.removed,
                            "parent": node.parent_node.id
                        })

                    if found_goal:
                        path, search_tree = self.generate_result(node=node, search_tree=search_tree,
                                                                 explored_dictionary=explored_dictionary)
                        return path, search_tree

                    bucket_index += 1

            del frontier[0]

            if not frontier and not path:
                self.print_result(found=False, explored_dictionary=explored_dictionary,
                                  search_tree=search_tree, path=path)
                return path, search_tree


    def informed_search(self, problem, frontier, path, explored_dictionary, search_tree, node_id, found_goal):

        # Find index of mid and index of upper
        my_dict = {}

        for bucket in frontier[0].bucket_array:
            if bucket.size >= problem["target"]:
                frontier[0].bucket_array.index(bucket)
                my_dict.update({int(frontier[0].bucket_array.index(bucket)): bucket.size})
            else:
                if len(my_dict) < 2:
                    frontier[0].bucket_array.index(bucket)
                    my_dict.update({int(frontier[0].bucket_array.index(bucket)): bucket.size})

        lower_index = my_dict.popitem()
        upper_index = my_dict.popitem()

        # Assuming m and n is determined, assign them
        lower_index = m
        upper_index = n

        while goal_state_not_found:
             if bucket_array[lower_index].filled == 0:
                bucket_array[lower_index].fill()
            else:
                bucket_array[lower_index].pour(upper)
                if bucket_array[upper_index].filled == bucket_array[upper_index].size:
                    bucket_array[upper_index].drain()


        while not found_goal:
            # Action:- Filling a bucket
            if frontier[0].bucket_array[lower_index[0]].filled == 0:

                node_id += 1
                child_node = copy.deepcopy(frontier[0])

                child_node.bucket_array = child_node.perform_action_and_return_buckets(
                    bucket_array=child_node.bucket_array, selected_index=lower_index[0], action="Fill",
                    target_index=None)

                frontier[0].children.append(node_id)

                node = Node(id=node_id, bucket_array=child_node.bucket_array, expansion_sequence=-1, children=[],
                            actions=child_node.actions, removed=None, parent_node=frontier[0])

                childStr = '{}'.format(child_node)

                if childStr in explored_dictionary:
                    node.removed = True
                else:
                    node.removed = False
                    explored_dictionary.update({'{}'.format(child_node): True})
                    found_goal = self.check_found_goal_state(child_node.bucket_array, problem["target"])

                search_tree.append(
                    {
                        "id": node.id,
                        "state": node.bucket_array,
                        "expansionsequence": node.expansion_sequence,
                        "children": node.children,
                        "actions": node.actions,
                        "removed": node.removed,
                        "parent": node.parent_node.id
                    })

                if found_goal:
                    path, search_tree = self.generate_result(node=node, search_tree=search_tree,
                                                             explored_dictionary=explored_dictionary)
                    return path, search_tree
                else:
                    frontier.append(node)

            # Action:- Pouring from a bucket to another bucket
            else:
                node_id += 1
                child_node = copy.deepcopy(frontier[0])

                child_node.bucket_array = child_node.perform_action_and_return_buckets(
                    bucket_array=child_node.bucket_array, selected_index=lower_index[0], action="Pour",
                    target_index=upper_index[0])

                frontier[0].children.append(node_id)

                node = Node(id=node_id, bucket_array=child_node.bucket_array, expansion_sequence=-1, children=[],
                            actions=child_node.actions, removed=None, parent_node=frontier[0])

                childStr = '{}'.format(child_node)

                if childStr in explored_dictionary:
                    node.removed = True
                else:
                    explored_dictionary.update({'{}'.format(child_node): True})
                    node.removed = False
                    found_goal = self.check_found_goal_state(child_node.bucket_array, problem["target"])

                search_tree.append(
                    {
                        "id": node.id,
                        "state": node.bucket_array,
                        "expansionsequence": node.expansion_sequence,
                        "children": node.children,
                        "actions": node.actions,
                        "removed": node.removed,
                        "parent": node.parent_node.id
                    })

                if found_goal:
                    path, search_tree = self.generate_result(node=node, search_tree=search_tree,
                                                             explored_dictionary=explored_dictionary)
                    return path, search_tree
                else:
                    frontier.append(node)

            # Action:- Draining a bucket
            if frontier[0].bucket_array[upper_index[0]].filled == frontier[0].bucket_array[upper_index[0]].size:
                node_id += 1
                child_node = copy.deepcopy(frontier[0])

                child_node.bucket_array = child_node.perform_action_and_return_buckets(
                    bucket_array=child_node.bucket_array, selected_index=upper_index[0], action="Drain",
                    target_index=None)

                frontier[0].children.append(node_id)

                node = Node(id=node_id, bucket_array=child_node.bucket_array, expansion_sequence=-1, children=[],
                            actions=child_node.actions, removed=None, parent_node=frontier[0])

                childStr = '{}'.format(child_node)

                if childStr in explored_dictionary:
                    node.removed = True
                else:
                    explored_dictionary.update({'{}'.format(child_node): True})
                    node.removed = False
                    found_goal = self.check_found_goal_state(child_node.bucket_array, problem["target"])

                search_tree.append(
                    {
                        "id": node.id,
                        "state": node.bucket_array,
                        "expansionsequence": node.expansion_sequence,
                        "children": node.children,
                        "actions": node.actions,
                        "removed": node.removed,
                        "parent": node.parent_node.id
                    })

                if found_goal:
                    path, search_tree = self.generate_result(node=node, search_tree=search_tree,
                                                             explored_dictionary=explored_dictionary)
                    return path, search_tree
                else:
                    frontier.append(node)

            del frontier[0]

            if not frontier and not path:
                self.print_result(found=False, explored_dictionary=explored_dictionary,
                                  search_tree=search_tree, path=path)
                return search_tree


if __name__ == '__main__':

    problem = {
        "size": [8, 5, 3],
        "filled": [0, 0, 0],
        "source": True,
        "sink": True,
        "target": 4
    }

    player = Player()
    player.run(problem)