
import time
import heapq
from collections import deque

class PuzzleState:
    def __init__(self, board, parent=None, move="", cost=0, depth=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.cost = cost
        self.depth = depth
        self.blank_pos = self.find_blank()
        
    def __lt__(self, other):
        return self.cost < other.cost
        
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

def bfs(initial_state, goal_state):
    start_time = time.time()
    
    frontier = deque([initial_state])
    explored = set()
    nodes_expanded = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        state = frontier.popleft()
        
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
            if neighbor_hash not in explored:
                explored.add(neighbor_hash)
                frontier.append(neighbor)
    
    end_time = time.time()
    return {
        "path": None,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "time": end_time - start_time
    }

def dfs(initial_state, goal_state, max_depth=float('inf')):
    start_time = time.time()
    
    frontier = [initial_state]
    explored = set()
    nodes_expanded = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        state = frontier.pop()
        
        if state.is_goal(goal_state):
            end_time = time.time()
            return {
                "path": state.get_path(),
                "nodes_expanded": nodes_expanded,
                "max_frontier_size": max_frontier_size,
                "time": end_time - start_time
            }
        
        if state.depth < max_depth:
            explored.add(hash(str(state.board)))
            nodes_expanded += 1
            
            neighbors = state.get_neighbors()
            # Reversed to maintain DFS order when using a stack
            for neighbor in reversed(neighbors):
                neighbor_hash = hash(str(neighbor.board))
                if neighbor_hash not in explored:
                    frontier.append(neighbor)
    
    end_time = time.time()
    return {
        "path": None,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "time": end_time - start_time
    }

def uniform_cost_search(initial_state, goal_state):
    start_time = time.time()
    
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
                heapq.heappush(frontier, neighbor)
                frontier_set.add(neighbor_hash)
    
    end_time = time.time()
    return {
        "path": None,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "time": end_time - start_time
    }

def iterative_deepening_search(initial_state, goal_state, max_depth=50):
    start_time = time.time()
    total_nodes_expanded = 0
    max_frontier_size = 0
    
    for depth in range(max_depth + 1):
        result = dfs(initial_state, goal_state, depth)
        
        total_nodes_expanded += result["nodes_expanded"]
        max_frontier_size = max(max_frontier_size, result["max_frontier_size"])
        
        if result["path"]:
            end_time = time.time()
            return {
                "path": result["path"],
                "nodes_expanded": total_nodes_expanded,
                "max_frontier_size": max_frontier_size,
                "time": end_time - start_time,
                "depth_found": depth
            }
    
    end_time = time.time()
    return {
        "path": None,
        "nodes_expanded": total_nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "time": end_time - start_time
    }

def solve_puzzle(initial_board, goal_board=None, algorithm="bfs"):
    if goal_board is None:
        goal_board = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]
    
    initial_state = PuzzleState(initial_board)
    goal_state = PuzzleState(goal_board)
    
    print(f"Solving with {algorithm}:")
    print("Initial state:")
    print_board(initial_board)
    print("Goal state:")
    print_board(goal_board)
    
    if algorithm.lower() == "bfs":
        result = bfs(initial_state, goal_state)
    elif algorithm.lower() == "dfs":
        result = dfs(initial_state, goal_state)
    elif algorithm.lower() == "ucs":
        result = uniform_cost_search(initial_state, goal_state)
    elif algorithm.lower() == "ids":
        result = iterative_deepening_search(initial_state, goal_state)
    else:
        return {"error": "Unknown algorithm"}
    
    if result["path"]:
        print(f"Solution found in {len(result['path']) - 1} moves")
        print(f"Nodes expanded: {result['nodes_expanded']}")
        print(f"Max frontier size: {result['max_frontier_size']}")
        print(f"Time taken: {result['time']:.4f} seconds")
        
        if "depth_found" in result:
            print(f"Solution found at depth: {result['depth_found']}")
            
        return result
    else:
        print("No solution found")
        return result

def compare_algorithms(initial_board, goal_board=None):
    results = {}
    
    for algorithm in ["bfs", "dfs", "ucs", "ids"]:
        print(f"\nRunning {algorithm.upper()}")
        results[algorithm] = solve_puzzle(initial_board, goal_board, algorithm)
    
    print("\nComparison Results:")
    print("-" * 60)
    print(f"{'Algorithm':<8} | {'Nodes':<10} | {'Frontier':<10} | {'Time (s)':<10} | {'Steps':<5}")
    print("-" * 60)
    
    for algo, result in results.items():
        if result.get("path"):
            steps = len(result["path"]) - 1
            print(f"{algo.upper():<8} | {result['nodes_expanded']:<10} | {result['max_frontier_size']:<10} | {result['time']:<10.4f} | {steps:<5}")
        else:
            print(f"{algo.upper():<8} | {result.get('nodes_expanded', 'N/A'):<10} | {result.get('max_frontier_size', 'N/A'):<10} | {result.get('time', 'N/A'):<10} | {'N/A':<5}")
    
    print("-" * 60)

# Example usage:
if __name__ == "__main__":
    # Example initial state (solvable)
    initial_board = [
        [1, 2, 3],
        [4, 0, 6],
        [7, 5, 8]
    ]
    
    # For testing multiple algorithms and comparing them
    compare_algorithms(initial_board)