import time
import heapq
from collections import deque
import math

class PuzzleState:
    def __init__(self, board, parent=None, move="", cost=0, depth=0, h_value=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.cost = cost  # g(n) - cost so far
        self.h_value = h_value  # h(n) - heuristic value
        self.depth = depth
        self.blank_pos = self.find_blank()
        
    def __lt__(self, other):
        # For A* and other priority queue-based algorithms
        # f(n) = g(n) + h(n)
        return (self.cost + self.h_value) < (other.cost + other.h_value)
        
    def __eq__(self, other):
        return self.board == other.board
        
    def __hash__(self):
        return hash(str(self.board))
    
    def find_blank(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return (i, j)
    
    def get_neighbors(self):
        neighbors = []
        i, j = self.blank_pos
        
        # Possible moves: up, down, left, right
        moves = [('UP', -1, 0), ('DOWN', 1, 0), ('LEFT', 0, -1), ('RIGHT', 0, 1)]
        
        for move_name, di, dj in moves:
            new_i, new_j = i + di, j + dj
            
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_board = [row[:] for row in self.board]
                new_board[i][j], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[i][j]
                neighbors.append(PuzzleState(new_board, self, move_name, self.cost + 1, self.depth + 1))
                
        return neighbors
    
    def is_goal(self, goal_state):
        return self.board == goal_state.board
    
    def get_path(self):
        path = []
        current = self
        while current:
            path.append((current.board, current.move))
            current = current.parent
        return list(reversed(path))

def print_board(board):
    for row in board:
        print(" ".join(str(x) if x != 0 else "_" for x in row))
    print()

# Heuristic functions for informed search

def manhattan_distance(board, goal_board):
    """
    Manhattan distance heuristic: sum of distances each tile is from its goal position
    """
    distance = 0
    goal_positions = {}
    
    # Create a map of each value to its goal position
    for i in range(3):
        for j in range(3):
            value = goal_board[i][j]
            if value != 0:
                goal_positions[value] = (i, j)
    
    # Calculate Manhattan distance for each tile
    for i in range(3):
        for j in range(3):
            value = board[i][j]
            if value != 0:  # Skip the blank tile
                goal_i, goal_j = goal_positions[value]
                distance += abs(i - goal_i) + abs(j - goal_j)
                
    return distance

def misplaced_tiles(board, goal_board):
    """
    Misplaced tiles heuristic: number of tiles not in their goal position
    """
    count = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] != 0 and board[i][j] != goal_board[i][j]:
                count += 1
    return count

def linear_conflict(board, goal_board):
    """
    Linear conflict heuristic: Manhattan distance + 2 * linear conflicts
    """
    # First calculate the manhattan distance
    man_dist = manhattan_distance(board, goal_board)
    conflicts = 0
    
    # Check rows for conflicts
    for i in range(3):
        for j1 in range(3):
            if board[i][j1] == 0:
                continue
                
            # Get the goal row for this tile
            goal_i = None
            goal_j = None
            for gi in range(3):
                for gj in range(3):
                    if goal_board[gi][gj] == board[i][j1]:
                        goal_i = gi
                        goal_j = gj
                        break
                if goal_i is not None:
                    break
            
            # If this tile is in its goal row
            if goal_i == i:
                for j2 in range(j1 + 1, 3):
                    if board[i][j2] == 0:
                        continue
                        
                    # Find goal position of the second tile
                    goal_i2 = None
                    goal_j2 = None
                    for gi in range(3):
                        for gj in range(3):
                            if goal_board[gi][gj] == board[i][j2]:
                                goal_i2 = gi
                                goal_j2 = gj
                                break
                        if goal_i2 is not None:
                            break
                    
                    # If second tile is also in its goal row, check for conflict
                    if goal_i2 == i and goal_j < goal_j2 and j1 > j2:
                        conflicts += 1
    
    # Check columns for conflicts
    for j in range(3):
        for i1 in range(3):
            if board[i1][j] == 0:
                continue
                
            # Get the goal column for this tile
            goal_i = None
            goal_j = None
            for gi in range(3):
                for gj in range(3):
                    if goal_board[gi][gj] == board[i1][j]:
                        goal_i = gi
                        goal_j = gj
                        break
                if goal_i is not None:
                    break
            
            # If this tile is in its goal column
            if goal_j == j:
                for i2 in range(i1 + 1, 3):
                    if board[i2][j] == 0:
                        continue
                        
                    # Find goal position of the second tile
                    goal_i2 = None
                    goal_j2 = None
                    for gi in range(3):
                        for gj in range(3):
                            if goal_board[gi][gj] == board[i2][j]:
                                goal_i2 = gi
                                goal_j2 = gj
                                break
                        if goal_i2 is not None:
                            break
                    
                    # If second tile is also in its goal column, check for conflict
                    if goal_j2 == j and goal_i < goal_i2 and i1 > i2:
                        conflicts += 1
    
    return man_dist + 2 * conflicts

# Informed Search Algorithms

def greedy_best_first_search(initial_state, goal_state, heuristic_func=manhattan_distance):
    """
    Greedy Best-First Search algorithm - expands the node with the lowest h(n) value
    """
    start_time = time.time()
    
    # Calculate the heuristic for the initial state
    initial_state.h_value = heuristic_func(initial_state.board, goal_state.board)
    
    frontier = []
    heapq.heappush(frontier, initial_state)
    explored = set()
    frontier_set = {hash(str(initial_state.board))}
    nodes_expanded = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        state = heapq.heappop(frontier)
        frontier_set.remove(hash(str(state.board)))
        
        if state.is_goal(goal_state):
            end_time = time.time()
            return {
                "path": state.get_path(),
                "nodes_expanded": nodes_expanded,
                "max_frontier_size": max_frontier_size,
                "time": end_time - start_time
            }
        
        explored.add(hash(str(state.board)))
        nodes_expanded += 1
        
        for neighbor in state.get_neighbors():
            neighbor_hash = hash(str(neighbor.board))
            if neighbor_hash not in explored and neighbor_hash not in frontier_set:
                # Calculate heuristic for neighbor (only considers h, not g for greedy)
                neighbor.h_value = heuristic_func(neighbor.board, goal_state.board)
                neighbor.cost = 0  # In greedy, we ignore the path cost
                
                heapq.heappush(frontier, neighbor)
                frontier_set.add(neighbor_hash)
    
    end_time = time.time()
    return {
        "path": None,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "time": end_time - start_time
    }

def astar_search(initial_state, goal_state, heuristic_func=manhattan_distance):
    """
    A* Search algorithm - expands the node with the lowest f(n) = g(n) + h(n) value
    """
    start_time = time.time()
    
    # Calculate the heuristic for the initial state
    initial_state.h_value = heuristic_func(initial_state.board, goal_state.board)
    
    frontier = []
    heapq.heappush(frontier, initial_state)
    explored = set()
    # Keep track of the best path to each state
    reached = {hash(str(initial_state.board)): initial_state}
    nodes_expanded = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        state = heapq.heappop(frontier)
        
        if state.is_goal(goal_state):
            end_time = time.time()
            return {
                "path": state.get_path(),
                "nodes_expanded": nodes_expanded,
                "max_frontier_size": max_frontier_size,
                "time": end_time - start_time
            }
        
        state_hash = hash(str(state.board))
        if state_hash in explored:
            continue
            
        explored.add(state_hash)
        nodes_expanded += 1
        
        for neighbor in state.get_neighbors():
            neighbor_hash = hash(str(neighbor.board))
            
            # Calculate heuristic for neighbor
            neighbor.h_value = heuristic_func(neighbor.board, goal_state.board)
            
            # If this is a new state or we found a better path to it
            if (neighbor_hash not in reached or 
                neighbor.cost < reached[neighbor_hash].cost):
                
                reached[neighbor_hash] = neighbor
                heapq.heappush(frontier, neighbor)
    
    end_time = time.time()
    return {
        "path": None,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "time": end_time - start_time
    }

def ida_star_search(initial_state, goal_state, heuristic_func=manhattan_distance):
    """
    IDA* (Iterative Deepening A*) Search algorithm
    """
    start_time = time.time()
    
    # Calculate initial heuristic value
    h0 = heuristic_func(initial_state.board, goal_state.board)
    initial_state.h_value = h0
    
    threshold = h0  # Initial threshold is just the heuristic value
    nodes_expanded = 0
    max_frontier_size = 1
    
    while True:
        # Start a DFS with the current threshold
        result = ida_star_dfs(initial_state, goal_state, threshold, nodes_expanded, 
                             max_frontier_size, heuristic_func)
        
        if isinstance(result, dict):  # Found a solution
            end_time = time.time()
            result["time"] = end_time - start_time
            return result
            
        if result == float('inf'):  # No solution within threshold and no larger thresholds to try
            end_time = time.time()
            return {
                "path": None,
                "nodes_expanded": nodes_expanded,
                "max_frontier_size": max_frontier_size,
                "time": end_time - start_time
            }
            
        # Otherwise, result is the new threshold for the next iteration
        threshold = result
        nodes_expanded = 0  # Reset counter for next iteration
        
def ida_star_dfs(state, goal_state, threshold, nodes_expanded, max_frontier_size, heuristic_func):
    """
    Depth-first search helper function for IDA*
    """
    # f(n) = g(n) + h(n)
    f = state.cost + state.h_value
    
    if f > threshold:
        return f
        
    if state.is_goal(goal_state):
        return {
            "path": state.get_path(),
            "nodes_expanded": nodes_expanded,
            "max_frontier_size": max_frontier_size
        }
        
    nodes_expanded += 1
    min_threshold = float('inf')
    
    neighbors = state.get_neighbors()
    max_frontier_size = max(max_frontier_size, len(neighbors))
    
    for neighbor in neighbors:
        neighbor.h_value = heuristic_func(neighbor.board, goal_state.board)
        
        result = ida_star_dfs(neighbor, goal_state, threshold, nodes_expanded, 
                             max_frontier_size, heuristic_func)
                             
        if isinstance(result, dict):  # Found a solution
            return result
            
        min_threshold = min(min_threshold, result)
        
    return min_threshold

def solve_puzzle_informed(initial_board, goal_board=None, algorithm="astar", heuristic="manhattan"):
    """
    Solve the 8-puzzle using informed search algorithms
    """
    if goal_board is None:
        goal_board = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]
    
    initial_state = PuzzleState(initial_board)
    goal_state = PuzzleState(goal_board)
    
    # Select heuristic function
    if heuristic.lower() == "manhattan":
        heuristic_func = manhattan_distance
    elif heuristic.lower() == "misplaced":
        heuristic_func = misplaced_tiles
    elif heuristic.lower() == "linear":
        heuristic_func = linear_conflict
    else:
        heuristic_func = manhattan_distance
    
    print(f"Solving with {algorithm.upper()} using {heuristic} heuristic:")
    print("Initial state:")
    print_board(initial_board)
    print("Goal state:")
    print_board(goal_board)
    
    if algorithm.lower() == "greedy":
        result = greedy_best_first_search(initial_state, goal_state, heuristic_func)
    elif algorithm.lower() == "astar":
        result = astar_search(initial_state, goal_state, heuristic_func)
    elif algorithm.lower() == "idastar":
        result = ida_star_search(initial_state, goal_state, heuristic_func)
    else:
        return {"error": "Unknown algorithm"}
    
    if result["path"]:
        print(f"Solution found in {len(result['path']) - 1} moves")
        print(f"Nodes expanded: {result['nodes_expanded']}")
        print(f"Max frontier size: {result['max_frontier_size']}")
        print(f"Time taken: {result['time']:.4f} seconds")
        return result
    else:
        print("No solution found")
        return result

def compare_informed_algorithms(initial_board, goal_board=None, heuristic="manhattan"):
    """
    Compare all informed search algorithms
    """
    results = {}
    
    for algorithm in ["greedy", "astar", "idastar"]:
        print(f"\nRunning {algorithm.upper()}")
        results[algorithm] = solve_puzzle_informed(initial_board, goal_board, algorithm, heuristic)
    
    print("\nComparison Results:")
    print("-" * 70)
    print(f"{'Algorithm':<10} | {'Nodes':<10} | {'Frontier':<10} | {'Time (s)':<10} | {'Steps':<5}")
    print("-" * 70)
    
    for algo, result in results.items():
        if result.get("path"):
            steps = len(result["path"]) - 1
            print(f"{algo.upper():<10} | {result['nodes_expanded']:<10} | {result['max_frontier_size']:<10} | {result['time']:<10.4f} | {steps:<5}")
        else:
            print(f"{algo.upper():<10} | {result.get('nodes_expanded', 'N/A'):<10} | {result.get('max_frontier_size', 'N/A'):<10} | {result.get('time', 'N/A'):<10} | {'N/A':<5}")
    
    print("-" * 70)

# Example usage:
if __name__ == "__main__":
    # Example initial state (solvable)
    initial_board = [
        [1, 2, 3],
        [4, 0, 6],
        [7, 5, 8]
    ]
    
    # For testing a single algorithm
    # solve_puzzle_informed(initial_board, algorithm="astar", heuristic="manhattan")
    
    # For comparing all algorithms with a specific heuristic
    compare_informed_algorithms(initial_board, heuristic="manhattan")
