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
            if state[i][j] == 0: continue
            target_i = (state[i][j] - 1) // 4
            target_j = (state[i][j] - 1) % 4
            distance += abs(i - target_i) + abs(j - target_j)
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
    #A*算法的主要逻辑,包括开放列表和关闭列表的管理、节点的扩展和选择等
    open_list = []
    closed_set = set()
    start_node = Node(start, None, 0, manhattan_distance(start, goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.state == goal:
            return reconstruct_path(current_node)

        closed_set.add(tuple(map(tuple, current_node.state)))
        for neighbor in get_neighbors(current_node.state):
            if tuple(map(tuple, neighbor)) in closed_set:
                continue

            g = current_node.g + 1
            h = manhattan_distance(neighbor, goal)
            neighbor_node = Node(neighbor, current_node, g, h)
            if neighbor_node not in open_list and tuple(map(tuple, neighbor)) not in closed_set:
                heapq.heappush(open_list, neighbor_node)

    return None

def reconstruct_path(node):
    #从目标节点回溯到起点，重建路径
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

if __name__ == "__main__":
    start = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
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
    else:
        print("No solution found.")