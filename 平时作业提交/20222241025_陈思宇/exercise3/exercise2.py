import heapq
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
class Node:
    def __init__(self, state, parent=None, move=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.g = g  # 实际代价
        self.h = h  # 启发式代价
        self.f = g + h  # 总代价

    def __lt__(self, other):
        return self.f < other.f  # 优先队列使用 f 值

# 曼哈顿距离启发式函数
def manhattan_distance(state):
    distance = 0
    for i in range(16):
        if state[i] == 0:
            continue
        target_x, target_y = divmod(state[i] - 1, 4)
        current_x, current_y = divmod(i, 4)
        distance += abs(current_x - target_x) + abs(current_y - target_y)
    return distance



def a_star(initial_state, goal_state):
    start_time = time.time()
    start_node = Node(initial_state, g=0, h=manhattan_distance(initial_state))
    frontier = []
    heapq.heappush(frontier, start_node)
    explored = set()
    g_values = {}  # 哈希表记录每个状态的 g 值
    g_values[tuple(initial_state)] = 0  # 初始状态的 g 值为 0
    expanded_nodes = 0  # 记录扩展的节点数
    generated_nodes = 1  # 记录生成的节点数，初始状态是已生成的

    while frontier:
        current_node = heapq.heappop(frontier)
        expanded_nodes += 1

        if current_node.state == goal_state:
            end_time = time.time()
            print(f"解找到！步数: {current_node.g}")
            print(f"扩展节点数: {expanded_nodes}")
            print(f"生成节点数: {generated_nodes}")
            print(f"运行时间: {end_time - start_time:.6f}秒")
            return current_node

        explored.add(tuple(current_node.state))

        # 查找空格位置
        zero_pos = current_node.state.index(0)
        zero_x, zero_y = zero_pos // 4, zero_pos % 4

        for dx, dy in directions:
            new_x, new_y = zero_x + dx, zero_y + dy
            if 0 <= new_x < 4 and 0 <= new_y < 4:
                new_pos = new_x * 4 + new_y
                new_state = current_node.state[:]
                # 交换空格和目标位置的数字
                new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]

                if tuple(new_state) not in explored:
                    new_g = current_node.g + 1
                    if tuple(new_state) not in g_values or new_g < g_values[tuple(new_state)]:
                        g_values[tuple(new_state)] = new_g
                        new_node = Node(new_state, parent=current_node, move=(zero_pos, new_pos), g=new_g, h=manhattan_distance(new_state))
                        heapq.heappush(frontier, new_node)
                        generated_nodes += 1
    print("未找到解！")
    return None
# 运行 A* 算法
initial_state = [8, 2, 3, 4, 5, 6, 1, 0, 9, 10, 11, 12, 13, 14, 7, 15]  # 示例初始状态

def visualize_solution():
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    def draw_board(state):
        board = np.array(state).reshape(4, 4)
        ax.imshow(board, cmap="Blues", vmin=0, vmax=15)
        for (i, j), val in np.ndenumerate(board):
            ax.text(j, i, str(val), ha='center', va='center', color='white', fontsize=12)

    def update(i):
        ax.clear()
        draw_board(solution_path[i])

    solution_path = []
    current_node = a_star(initial_state,goal_state)
    while current_node:
        solution_path.append(current_node.state)
        current_node = current_node.parent
    solution_path.reverse()

    ani = FuncAnimation(fig, update, frames=len(solution_path), repeat=False)
    plt.show()
visualize_solution()