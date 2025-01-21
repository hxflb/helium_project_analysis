import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# 定义节点类
class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        """
        初始化节点类，表示拼图问题中的一个状态。

        Parameters:
        state (list): 当前拼图状态（长度为16，包含0）。
        parent (Node): 当前节点的父节点，用于路径回溯。
        g (int): 从起始状态到当前节点的路径成本。
        h (int): 启发式估计值（通常是曼哈顿距离或错位数）。
        """
        self.state = state  # 当前拼图状态
        self.parent = parent  # 父节点，用于路径回溯
        self.g = g  # 从起始状态到当前节点的路径成本
        self.h = h  # 启发式估计值
        self.f = g + h  # 总成本，f = g + h
        self.blank_pos = state.index(0)  # 空白块的位置

    def __lt__(self, other):
        """
        小于运算符，用于堆排序。

        比较两个节点的总代价f，用于堆中节点的优先级排序。
        """
        return self.f < other.f


# 启发式函数：曼哈顿距离 + 错位数
def combined_heuristic(state, goal_state):
    """
    计算当前状态到目标状态的启发式估计值（结合曼哈顿距离和错位数）。

    曼哈顿距离表示每个块从当前位置到目标位置的水平和垂直距离的和。
    错位数表示当前状态和目标状态中不一致的块的数量。

    Parameters:
    state (list): 当前拼图状态。
    goal_state (list): 目标拼图状态。

    Returns:
    int: 启发式估计值（曼哈顿距离 + 错位数的和）。
    """
    manhattan_dist = 0
    misplaced_tiles = 0
    for i in range(len(state)):
        if state[i] != 0:  # 忽略空白块（0）
            goal_pos = goal_state.index(state[i])  # 找到目标状态中该块的位置
            manhattan_dist += abs(i // 4 - goal_pos // 4) + abs(i % 4 - goal_pos % 4)  # 计算曼哈顿距离
            if state[i] != goal_state[i]:  # 判断该块是否错位
                misplaced_tiles += 1
    return manhattan_dist + misplaced_tiles  # 返回启发式估计值


# 生成邻居节点
def generate_neighbors(state):
    """
    生成当前状态的所有可能邻居状态。通过移动空白块（0）到相邻的位置。

    Parameters:
    state (list): 当前拼图状态。

    Returns:
    list: 当前状态的所有邻居状态列表。
    """
    neighbors = []
    size = 4  # 4x4的拼图，尺寸为4x4
    blank_pos = state.index(0)  # 找到空白块的位置
    row, col = divmod(blank_pos, size)  # 获取空白块的行列坐标

    # 定义可能的移动方向（上下左右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc  # 计算新位置的行列坐标
        if 0 <= new_row < size and 0 <= new_col < size:  # 如果新位置合法
            new_blank_pos = new_row * size + new_col  # 计算新空白块的位置
            new_state = state[:]  # 复制当前状态
            new_state[blank_pos], new_state[new_blank_pos] = new_state[new_blank_pos], new_state[blank_pos]  # 交换空白块和相邻块
            neighbors.append(new_state)  # 将新状态添加到邻居列表

    return neighbors


# A*算法实现，优化包括剪枝和记忆化搜索
def a_star(start_state, goal_state, heuristic=combined_heuristic):
    """
    A*算法的实现，用于求解15数码问题的最短路径。

    参数：
    start_state (list): 起始状态（拼图的初始状态）。
    goal_state (list): 目标状态（拼图的目标状态）。
    heuristic (function): 启发式函数，默认使用结合的曼哈顿距离和错位数。

    返回：
    tuple: 包含路径（list）、访问状态列表、扩展节点数、生成节点数和总运行时间的信息。
    """
    open_list = []  # 开放列表，存储待扩展的节点
    closed_list = set()  # 关闭列表，存储已扩展的节点
    g_costs = {}  # 记忆化搜索：存储每个状态的最短路径成本
    goal_node = Node(goal_state, g=0, h=heuristic(goal_state, start_state))  # 创建目标节点
    heapq.heappush(open_list, goal_node)  # 将目标节点加入开放列表

    visited_states = []  # 访问过的状态列表
    expanded_nodes = 0  # 扩展的节点数
    generated_nodes = 0  # 生成的节点数

    start_time = time.time()  # 记录开始时间

    while open_list:
        current_node = heapq.heappop(open_list)  # 从开放列表中取出f值最小的节点
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
            if tuple(next_state) not in g_costs or g_costs[tuple(next_state)] > g:
                g_costs[tuple(next_state)] = g  # 更新最短路径成本
                heapq.heappush(open_list, neighbor_node)  # 将该邻居节点加入开放列表
                generated_nodes += 1  # 每生成一个节点，生成节点数加1

    return None, visited_states, expanded_nodes, generated_nodes, 0  # 无解

# 步骤列表可视化（带动画）
def a_star_visualize(start_state, goal_state, heuristic=combined_heuristic):
    """
    使用A*算法解决15数码问题并展示求解过程的动画。

    参数：
    start_state (list): 起始状态（拼图的初始状态）。
    goal_state (list): 目标状态（拼图的目标状态）。
    heuristic (function): 启发式函数，默认使用结合的曼哈顿距离和错位数。

    输出：
    打印解决方案，扩展节点数，生成节点数，运行时间，以及每一步状态的动画。
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
        print(np.array(state).reshape(4, 4))  # 打印每一步的状态，格式化为4x4的拼图

    # 创建绘图区域
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_yticks(np.arange(0, 4, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='both')

    # 更新绘图函数，用于动画
    def update(frame):
        """
        动画更新函数，显示每一步的状态。
        """
        ax.clear()
        ax.set_xticks(np.arange(0, 4, 1))
        ax.set_yticks(np.arange(0, 4, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        state = visited_states[frame]
        state = np.array(state).reshape(4, 4)
        ax.matshow(state, cmap='Blues')  # 使用蓝色渐变显示拼图状态

        for i in range(4):
            for j in range(4):
                if state[i, j] != 0:  # 如果不是空白块（0），则显示数字
                    ax.text(j, i, str(state[i, j]), ha='center', va='center', fontsize=20, color='white')

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(visited_states), repeat=False, interval=500)

    # 显示动画
    plt.show()


# 主程序
if __name__ == "__main__":
    # 设定初始状态和目标状态
    start_state = [8, 2, 3, 4, 5, 6, 1, 7, 9, 10, 11, 12, 13, 14, 0, 15]  # 修改为16个元素
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]  # 修改为16个元素

    # 执行A*算法并展示可视化
    print("Starting A* algorithm visualization...")
    a_star_visualize(start_state, goal_state)
