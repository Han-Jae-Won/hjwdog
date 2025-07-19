import numpy as np
import random
from collections import deque

def generate_maze(width, height, complexity=.75, density=.75):
    # Adjust complexity and density to be within 0-1 range
    complexity = int(complexity * (5 * (height + width)))
    density = int(density * ((height // 2) * (width // 2)))

    # Create a grid with walls
    Z = np.zeros((height, width), dtype=int)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1

    # Make aisles
    for i in range(density):
        # Ensure x, y are within inner bounds and are even
        x = random.randint(1, (width - 2) // 2) * 2
        y = random.randint(1, (height - 2) // 2) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1: neighbours.append((y, x - 2))
            if x < width - 2: neighbours.append((y, x + 2))
            if y > 1: neighbours.append((y - 2, x))
            if y < height - 2: neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = random.choice(neighbours)
                # Ensure the chosen neighbor is within valid bounds before updating
                if 0 < y_ < height - 1 and 0 < x_ < width - 1:
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        # Calculate the cell between current and neighbor, ensure it's within bounds
                        mid_y = y_ + (y - y_) // 2
                        mid_x = x_ + (x - x_) // 2
                        if 0 < mid_y < height - 1 and 0 < mid_x < width - 1:
                            Z[mid_y, mid_x] = 1
                            x, y = x_, y_
    return Z

def find_path_bfs(maze, start, end):
    rows, cols = maze.shape
    queue = deque([(start, [start])]) # (current_position, path_list)
    visited = set([start])

    while queue:
        (r, c), path = queue.popleft()

        if (r, c) == end:
            return path

        # Define possible movements (up, down, left, right)
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dr, dc in moves:
            nr, nc = r + dr, c + dc

            # Check boundaries and if it's not a wall and not visited
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None # No path found

def visualize_maze(maze, path=None, start=None, end=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red', 'green'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create a copy to draw on
    display_maze = np.copy(maze)

    # Mark start (2) and end (3) points
    if start:
        display_maze[start[0], start[1]] = 2
    if end:
        display_maze[end[0], end[1]] = 3

    # Mark path (4)
    if path:
        for r, c in path:
            if (r, c) != start and (r, c) != end:
                display_maze[r, c] = 4

    fig, ax = plt.subplots(figsize=(maze.shape[1] * 0.5, maze.shape[0] * 0.5))
    ax.imshow(display_maze, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig