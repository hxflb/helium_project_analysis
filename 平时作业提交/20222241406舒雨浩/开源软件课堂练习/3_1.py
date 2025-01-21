import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state# 当前状态
        self.parent = parent#父节点
        self.g = g#从起点到当前节点的成本
        self.h = h # 启发式估计从当前节点到目标节点的成本
        self.f = g + h # 总成本

    def __lt__(self, other):
        #重载小于操作，用于优先队列中按f值排序
        return self.f < other.f

#曼哈顿距离启发式函数，计算从当前状态到目标状态的最小成本估计
def manhattan_distance(state, goal):
    distance = 0
    for i in range(4):
        for j in range(4):
            value = state[i][j]
            if value != 0:
                # 找到数字在goal中的位置
                target_index = goal.index(value) if value in goal else len(goal) - 1
                goal_x, goal_y = divmod(target_index, 4)
                state_x, state_y = i, j  # 获取state中数字的位置
                distance += abs(state_x - goal_x) + abs(state_y - goal_y)
    return distance

#获取当前状态的所有合法移动（邻居状态）
def get_neighbors(state):
    neighbors = []
    zero_pos = [(i, j) for i in range(4) for j in range(4) if state[i][j] == 0][0]
    x, y = zero_pos
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 4 and 0 <= ny < 4:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(new_state)

    return neighbors



def a_star(start, goal):
    open_list = []
    closed_set = set()
    start_node = Node(start, None, 0, manhattan_distance(start, goal))
    heapq.heappush(open_list, start_node)
    start_time = time.time()

    expanded_nodes = 0
    generated_nodes = 1

    while open_list:
        current_node = heapq.heappop(open_list)
        expanded_nodes += 1

        if current_node.state == goal:
            path = reconstruct_path(current_node)
            print(f"Expanded nodes: {expanded_nodes}")
            print(f"Generated nodes: {generated_nodes}")
            print(f"Runtime: {time.time() - start_time} seconds")
            return path

        closed_set.add(tuple(map(tuple, current_node.state)))

        for neighbor in get_neighbors(current_node.state):
            if tuple(map(tuple, neighbor)) in closed_set:
                continue

            g = current_node.g + 1
            h = manhattan_distance(neighbor, goal)
            neighbor_node = Node(neighbor, current_node, g, h)
            generated_nodes += 1
            if (neighbor_node not in open_list and
                    tuple(map(tuple, neighbor)) not in closed_set):
                heapq.heappush(open_list, neighbor_node)

    return None

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

    # while open_list:

    #     current_node = heapq.heappop(open_list)
    #     if current_node.state == goal:
    #         return reconstruct_path(current_node)
    #
    #     closed_set.add(tuple(map(tuple, current_node.state)))
    #     for neighbor in get_neighbors(current_node.state):
    #         if tuple(map(tuple, neighbor)) in closed_set:
    #             continue
    #
    #         g = current_node.g + 1
    #         h = manhattan_distance(neighbor, goal)
    #         neighbor_node = Node(neighbor, current_node, g, h)
    #         if neighbor_node not in open_list and tuple(map(tuple, neighbor)) not in closed_set:
    #             heapq.heappush(open_list, neighbor_node)
    #
    # return None



# 可视化函数
def visualize(path, start, goal):
    fig, ax = plt.subplots()
    ims = []  # 用于存储每一帧的图像

    for state in path:
        im = ax.imshow(state, cmap='Greys', interpolation='none')
        ims.append([im])

    def init():
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        return ims[0]

    def update(frame):
        ax.clear()
        ax.imshow(path[frame], cmap='Greys', interpolation='none')
        return ims[frame]

    ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init, blit=False, interval=100)
    plt.show()

def initialize_plot():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    return fig, ax


def update_plot(ax, state, path):
    ax.clear()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    for i in range(4):
        for j in range(4):
            value = state[i][j]
            if value == 0:
                ax.text(j + 0.5, 4 - i, ' ', va='center', ha='center', fontsize=12)
            else:
                ax.text(j + 0.5, 4 - i, str(value), va='center', ha='center', fontsize=12)
    # 突出显示路径中的当前状态
    if path:
        current_state = path[-1]  # 获取当前步骤的状态
        zero_pos = [(i, j) for i in range(4) for j in range(4) if current_state[i][j] == 0][0]
        ax.scatter(zero_pos[1] + 0.5, 4 - zero_pos[0], color='red', zorder=5, s=100)

def visualize_path(path):
    fig, ax = initialize_plot()
    for step in path:
        update_plot(ax, step, path)
        plt.pause(0.5)  # Pause to show the update

def animate_path(path):
    fig, ax = plt.subplots()
    ax.set_title('A* Algorithm Animation')
    states = [step for step in path]  # 将路径转换为状态列表

    def animate(i):
        update_plot(ax, states[i], path)  # 确保只传递当前状态

    ani = animation.FuncAnimation(fig, animate, frames=len(states), interval=200, blit=False)
    plt.show()

if __name__ == "__main__":
    start = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 0, 11, 12],  # 只有数字10不在正确的位置上
        [13, 14, 15, 10]
        # [1, 2, 3, 0],  # 假设0在最后一个位置
        # [5, 6, 7, 8],
        # [9, 10, 11, 12],
        # [13, 14, 15, 4]
    ]

    goal = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
    ]

    path = a_star(start, goal)
    if path:
        print("Solution found!")
        for step in path:
            for row in step:
                print(row)
            print()

        visualize(path, start, goal)

        animate_path(path)
    else:
        print("No solution found.")