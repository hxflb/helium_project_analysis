import heapq
import time


class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g  # Path cost
        self.h = h  # Heuristic cost
        self.f = g + h  # Total cost (f = g + h)

    def __lt__(self, other):
        # For the priority queue to order nodes by their f value
        return self.f < other.f

    def __repr__(self):
        return f"Node(state={self.state}, f={self.f}, g={self.g}, h={self.h})"


def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(16):  # There are 16 elements in total (4x4 puzzle)
        value = state[i]
        if value != 0:
            goal_x, goal_y = divmod(goal_state.index(value), 4)  # Get the goal position
            x, y = divmod(i, 4)  # Get the current position
            distance += abs(x - goal_x) + abs(y - goal_y)
    return distance


def generate_neighbors(state):
    # Find the position of the blank space (0)
    zero_pos = state.index(0)
    x, y = divmod(zero_pos, 4)

    # Define possible moves for the blank space (up, down, left, right)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []

    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < 4 and 0 <= new_y < 4:
            # Valid move, generate new state
            new_state = state[:]
            new_zero_pos = new_x * 4 + new_y
            new_state[zero_pos], new_state[new_zero_pos] = new_state[new_zero_pos], new_state[zero_pos]
            neighbors.append(new_state)

    return neighbors


def a_star(start_state, goal_state):
    open_list = []
    closed_list = set()
    start_node = Node(start_state, None, g=0, h=manhattan_distance(start_state, goal_state))
    heapq.heappush(open_list, start_node)

    expanded_nodes = 0
    generated_nodes = 1

    while open_list:
        current_node = heapq.heappop(open_list)

        # If we reach the goal, reconstruct the path
        if current_node.state == goal_state:
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            path.reverse()
            return path, expanded_nodes, generated_nodes

        # Add current node to closed list
        closed_list.add(tuple(current_node.state))

        # Generate neighbors
        for neighbor_state in generate_neighbors(current_node.state):
            if tuple(neighbor_state) in closed_list:
                continue

            neighbor_node = Node(
                neighbor_state,
                parent=current_node,
                g=current_node.g + 1,
                h=manhattan_distance(neighbor_state, goal_state)
            )

            if not any(neighbor_node.state == open_node.state and neighbor_node.f >= open_node.f for open_node in
                       open_list):
                heapq.heappush(open_list, neighbor_node)
                generated_nodes += 1

        expanded_nodes += 1

    return None, expanded_nodes, generated_nodes  # If no solution is found


def print_solution(path, expanded_nodes, generated_nodes):
    if path:
        print("找到解路径!")
        for step in path:
            for i in range(0, 16, 4):
                # Ensure number alignment
                print(" ".join(f"{num:2d}" for num in step[i:i + 4]))
            print()
        print(f"扩展节点数: {expanded_nodes}")
        print(f"生成节点数: {generated_nodes}")
    else:
        print("未找到解。")


# Example usage
if __name__ == "__main__":
    # Initial and goal state for 15-puzzle
    start_state = [8, 2, 3, 4, 5, 6, 1, 7, 9, 10, 11, 12, 13, 14, 0, 15]
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]

    start_time = time.time()
    solution, expanded_nodes, generated_nodes = a_star(start_state, goal_state)
    end_time = time.time()

    print_solution(solution, expanded_nodes, generated_nodes)
    print(f"运行时间: {end_time - start_time} s")
