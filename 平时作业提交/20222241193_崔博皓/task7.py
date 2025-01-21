import heapq
import time
import numpy as np

# 定义节点类
class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        """
        初始化节点类，表示一个拼图状态。
        Parameters:
        state (list): 当前拼图状态（长度为16，包含0）。
        parent (Node): 当前节点的父节点，用于路径回溯。
        g (int): 从起始状态到当前节点的路径成本。
        h (int): 当前状态到目标状态的启发式估计值。
        """
        self.state = state  # 当前的状态
        self.parent = parent  # 父节点
        self.g = g  # 从起点到当前节点的路径成本
        self.h = h  # 启发式估计
        self.f = g + h  # 总成本 f = g + h
        self.blank_pos = state.index(0)  # 空白块的位置（0所在的位置）

    def __lt__(self, other):
        """
        小于运算符，用于堆排序。
        根据节点的总成本 f 值进行比较，使得堆按照 f 值排序。
        """
        return self.f < other.f


# 启发式函数：曼哈顿距离
def manhattan_distance(state, goal_state):
    """
    计算当前状态到目标状态的曼哈顿距离（启发式函数）。
    曼哈顿距离是所有非空格块的目标位置和当前状态的距离之和。

    Parameters:
    state (list): 当前拼图状态。
    goal_state (list): 目标拼图状态。

    Returns: 曼哈顿距离。
    """
    distance = 0
    for i in range(len(state)):
        if state[i] != 0:  # 忽略空白块（0）
            goal_pos = goal_state.index(state[i])  # 找到当前块在目标状态中的位置
            # 计算当前块与目标位置的行列距离
            distance += abs(i // 4 - goal_pos // 4) + abs(i % 4 - goal_pos % 4)
    return distance


# 启发式函数：错位数
def misplaced_tiles(state, goal_state):
    """
    计算当前状态到目标状态的错位数（启发式函数）。
    错位数是指当前状态与目标状态中不匹配的块的数量。

    Parameters:
    state (list): 当前拼图状态。
    goal_state (list): 目标拼图状态。

    Returns:
    int: 错位数。
    """
    return sum(1 for i in range(len(state)) if state[i] != goal_state[i] and state[i] != 0)


# 生成邻居节点
def generate_neighbors(state):
    """
    生成当前状态的所有可能邻居状态。

    通过移动空白块（0）到相邻位置，生成新的状态。

    Parameters:
    state (list): 当前拼图状态。

    Returns:
    list: 当前状态的所有邻居状态列表。
    """
    neighbors = []
    size = 4  # 4x4的拼图，尺寸为4x4
    blank_pos = state.index(0)  # 找到空白块的位置
    row, col = divmod(blank_pos, size)  # 计算空白块的行列位置

    # 定义可能的移动方向（上下左右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < size and 0 <= new_col < size:
            new_blank_pos = new_row * size + new_col  # 新的空白块位置
            new_state = state[:]  # 复制当前状态
            new_state[blank_pos], new_state[new_blank_pos] = new_state[new_blank_pos], new_state[blank_pos]  # 交换空白块和相邻块
            neighbors.append(new_state)  # 添加新的状态到邻居列表

    return neighbors


# A*算法实现
def a_star(start_state, goal_state, heuristic=manhattan_distance):
    """
    A*算法实现，用于求解15数码问题的最短路径。
    Parameters:
    start_state (list): 起始状态（拼图的初始状态）。
    goal_state (list): 目标状态（拼图的最终状态）。
    heuristic (function): 启发式函数，默认使用曼哈顿距离。

    Returns:
    tuple: 包含路径、访问状态列表、扩展节点数、生成节点数和总运行时间的信息。

    扩展过程
    从开放列表中选择一个具有最低 f(n) 值的节点，并将其从开放列表移到关闭列表。
    对于该节点的所有邻居节点，执行以下操作：
    如果邻居节点在关闭列表中，跳过它。
    计算邻居节点的 g 和 h 值。
    如果邻居节点不在开放列表中，添加到开放列表中，并设置其父节点为当前节点。
    如果邻居节点已经在开放列表中，检查是否通过当前节点到达该邻居节点的路径更短。如果是，则更新该邻居节点的 g 值，并更新父节点为当前节点。

    如果开放列表为空，说明没有路径可达目标，算法失败。
    如果目标节点被移到关闭列表中，说明已经找到最短路径，可以结束算法并重建路径。
    重建路径
    从目标节点回溯到起始节点，沿着每个节点的父节点指针，直到回到起点，得到完整的路径。
    """
    open_list = []  # 开放列表，用于存储待扩展的节点
    closed_list = set()  # 关闭列表，用于存储已扩展的节点
    goal_node = Node(goal_state, g=0, h=heuristic(goal_state, start_state))  # 从目标状态开始
    heapq.heappush(open_list, goal_node)  # 将目标节点加入开放列表

    visited_states = []  # 访问过的状态列表
    expanded_nodes = 0  # 扩展节点数
    generated_nodes = 0  # 生成节点数

    start_time = time.time()  # 记录开始时间

    while open_list:
        # 从开放列表中弹出具有最小f值的节点
        current_node = heapq.heappop(open_list)
        expanded_nodes += 1  # 扩展节点数增加

        # 如果当前节点是起始状态，回溯路径
        if current_node.state == start_state:
            path = []
            while current_node:
                path.append(current_node.state)
                visited_states.append(current_node.state)  # 添加访问的状态到列表
                current_node = current_node.parent
            end_time = time.time()  # 记录结束时间
            total_time = end_time - start_time  # 计算运行时间
            return path[::-1], visited_states, expanded_nodes, generated_nodes, total_time  # 返回路径和相关信息

        closed_list.add(tuple(current_node.state))  # 将当前节点加入关闭列表

        # 生成当前节点的所有邻居节点
        for next_state in generate_neighbors(current_node.state):
            if tuple(next_state) in closed_list:
                continue  # 如果该邻居节点已经扩展过，跳过

            g = current_node.g + 1  # 每次移动的代价是1
            h = heuristic(next_state, start_state)  # 计算启发式值
            neighbor_node = Node(next_state, parent=current_node, g=g, h=h)

            # 如果该邻居节点已经在开放列表中，检查是否需要更新
            if not any(neighbor_node.state == n.state and neighbor_node.f >= n.f for n in open_list):
                heapq.heappush(open_list, neighbor_node)  # 将该邻居节点加入开放列表
                generated_nodes += 1  # 每生成一个节点，生成节点数加1

    return None, visited_states, expanded_nodes, generated_nodes, 0  # 无解


# 步骤列表可视化
def a_star_visualize(start_state, goal_state, heuristic=manhattan_distance):
    """
    使用A*算法解决15数码问题并展示求解过程。

    参数:
    start_state (list): 起始状态（拼图的初始状态）。
    goal_state (list): 目标状态（拼图的最终状态）。
    heuristic (function): 启发式函数，默认使用曼哈顿距离。

    输出:
    打印解决方案，扩展节点数，生成节点数，运行时间，以及每一步状态。
    """
    visited_states = []
    path, visited_states, expanded_nodes, generated_nodes, total_time = a_star(start_state, goal_state, heuristic)

    if not path:
        print("No solution found!")
        return

    # 打印关键信息
    print("Solution found!")
    print(f"Expanded nodes: {expanded_nodes}")
    print(f"Generated nodes: {generated_nodes}")
    print(f"Total runtime: {total_time:.4f} seconds")

    # 打印每步的状态
    for i, state in enumerate(visited_states):
        print(f"Step {i}:")
        print(np.array(state).reshape(4, 4))  # 每一步的状态展示为4x4的拼图


# 主程序
if __name__ == "__main__":
    # 设定初始状态和目标状态
    start_state = [8, 2, 3, 4, 5, 6, 1, 7, 9, 10, 11, 12, 13, 14, 0, 15]  # 修改为16个元素
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]  # 修改为16个元素

    # 执行A*算法并展示可视化
    print("Starting A* algorithm visualization...")
    a_star_visualize(start_state, goal_state)
