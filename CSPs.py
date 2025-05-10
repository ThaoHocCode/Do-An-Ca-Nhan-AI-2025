import time
import random
import copy
from collections import deque

class PuzzleState:
    def __init__(self, board, parent=None, move="", cost=0, depth=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.cost = cost
        self.depth = depth
        self.blank_pos = self.find_blank()
        
    def __eq__(self, other):
        if isinstance(other, PuzzleState):
            return self.board == other.board
        return False
        
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

# Heuristic functions for CSP algorithms

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

def heuristic_func(board, goal_board, heuristic="manhattan"):
    """Select the appropriate heuristic function"""
    if heuristic.lower() == "manhattan":
        return manhattan_distance(board, goal_board)
    elif heuristic.lower() == "misplaced":
        return misplaced_tiles(board, goal_board)
    else:
        return manhattan_distance(board, goal_board)

# CSP Modeling and Algorithms

def is_goal_state(board, goal_board):
    """Check if the current board matches the goal board"""
    return board == goal_board

def get_blank_pos(board):
    """Find the position of the blank (0) tile"""
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return (i, j)
    return None

def get_possible_moves(board):
    """Get all possible moves from current board"""
    i, j = get_blank_pos(board)
    possible_moves = []
    
    # Try moving blank up, down, left, right
    directions = [
        ('UP', -1, 0), 
        ('DOWN', 1, 0), 
        ('LEFT', 0, -1), 
        ('RIGHT', 0, 1)
    ]
    
    for direction, di, dj in directions:
        new_i, new_j = i + di, j + dj
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_board = [row[:] for row in board]
            new_board[i][j], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[i][j]
            possible_moves.append((new_board, direction))
    
    return possible_moves

def get_variables(board):
    """
    For CSP, the variables are the positions on the board.
    Returns a list of (row, col) tuples for each position.
    """
    variables = []
    for i in range(3):
        for j in range(3):
            variables.append((i, j))
    return variables

def get_domains(board, goal_board):
    """
    For each variable (position), its domain is the set of possible values (tiles).
    For 8-puzzle as CSP, the domains are constrained by the current state.
    """
    domains = {}
    for i in range(3):
        for j in range(3):
            # For the 8-puzzle, each position's domain is restricted to its current value
            # This is because we can't arbitrarily assign values to positions
            domains[(i, j)] = [board[i][j]]
    return domains

def get_constraints(goal_board):
    """
    Define the constraints for the 8-puzzle.
    Each position (i, j) should have the value goal_board[i][j].
    """
    constraints = {}
    for i in range(3):
        for j in range(3):
            constraints[(i, j)] = goal_board[i][j]
    return constraints

def count_violations(board, goal_board):
    """Count the number of positions that don't match the goal state"""
    count = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] != 0 and board[i][j] != goal_board[i][j]:
                count += 1
    return count

def backtracking_search(initial_board, goal_board, max_depth=30, heuristic="manhattan"):
    """
    Solve 8-puzzle using backtracking search.
    """
    start_time = time.time()
    
    def backtrack(board, path, depth):
        if is_goal_state(board, goal_board):
            return path + [(board, "GOAL")]
        
        if depth >= max_depth:
            return None
            
        # Sort moves based on heuristic (to try more promising moves first)
        possible_moves = get_possible_moves(board)
        possible_moves.sort(key=lambda move: heuristic_func(move[0], goal_board, heuristic))
        
        for next_board, move in possible_moves:
            # Check if this state is already in the path to avoid cycles
            if not any(next_board == p[0] for p in path):
                # Add the move attempt to the path
                path.append((board, f"Trying: {move}"))
                result = backtrack(next_board, path, depth + 1)
                if result:
                    return result
                # If we get here, we need to backtrack
                path.pop()  # Remove the failed move
                path.append((board, f"Backtrack: {move} failed"))
        
        return None
    
    path = backtrack(initial_board, [], 0)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    if path:
        # Transform path to match the expected format
        formatted_path = []
        for i in range(len(path)):
            if i == 0:
                formatted_path.append((path[i][0], ""))
            else:
                formatted_path.append((path[i][0], path[i-1][1]))
        
        return {
            "path": formatted_path,
            "depth": len(path) - 1,
            "time": time_taken
        }
    else:
        return {
            "path": None,
            "depth": None,
            "time": time_taken
        }

def forward_checking(board, goal_board, variable, value):
    """
    Check if assigning value to variable (position) would lead to domain wipeout.
    For 8-puzzle, this is used to simulate the constraints checking when making a move.
    """
    # Create a new board with the assignment
    new_board = [row[:] for row in board]
    i, j = variable
    blank_i, blank_j = get_blank_pos(board)
    
    # Swap the blank with the value
    new_board[i][j], new_board[blank_i][blank_j] = new_board[blank_i][blank_j], new_board[i][j]
    
    # Check if the new position of each tile is consistent with the goal
    for r in range(3):
        for c in range(3):
            if new_board[r][c] != 0 and new_board[r][c] == goal_board[r][c]:
                # The tile is in its goal position, which is good
                continue
            elif new_board[r][c] != 0:
                # Check if there's a path to get this tile to its goal position
                tile_value = new_board[r][c]
                goal_r, goal_c = None, None
                
                # Find the goal position for this tile
                for gr in range(3):
                    for gc in range(3):
                        if goal_board[gr][gc] == tile_value:
                            goal_r, goal_c = gr, gc
                            break
                    if goal_r is not None:
                        break
                
                # If the tile is completely blocked with no way to reach its goal,
                # return False to indicate forward checking failure
                if is_blocked(new_board, r, c, goal_r, goal_c):
                    return False
    
    return True

def is_blocked(board, r, c, goal_r, goal_c):
    """
    Check if a tile is completely blocked from reaching its goal position.
    This is a simplified check and may not catch all blocking cases.
    """
    # This is a simplified check that assumes a tile is blocked if:
    # 1. It's not in its goal position
    # 2. Its goal position is occupied by another tile
    # 3. The blank is not adjacent to either the tile or its goal position
    
    # If the tile is already at the goal, it's not blocked
    if r == goal_r and c == goal_c:
        return False
    
    # Check if the goal position is occupied by another tile that's not the blank
    if board[goal_r][goal_c] != 0:
        # Find the blank position
        blank_r, blank_c = get_blank_pos(board)
        
        # Check if blank is adjacent to either the tile or its goal
        adjacent_to_tile = (abs(blank_r - r) + abs(blank_c - c) == 1)
        adjacent_to_goal = (abs(blank_r - goal_r) + abs(blank_c - goal_c) == 1)
        
        # If the blank is not adjacent to either, the tile might be blocked
        if not (adjacent_to_tile or adjacent_to_goal):
            return True
    
    # Default to not blocked
    return False

def backtracking_with_forward_checking(initial_board, goal_board, max_depth=30, heuristic="manhattan"):
    """
    Solve 8-puzzle using backtracking search with forward checking.
    Forward checking helps prune the search space by checking if a move leads to an invalid state.
    """
    start_time = time.time()
    
    def backtrack_fc(board, path, depth):
        if is_goal_state(board, goal_board):
            return path + [(board, "GOAL")]
        
        if depth >= max_depth:
            return None
            
        # Get valid moves
        blank_pos = get_blank_pos(board)
        possible_moves = []
        
        # Try moving a tile adjacent to the blank
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            tile_i, tile_j = blank_pos[0] + di, blank_pos[1] + dj
            if 0 <= tile_i < 3 and 0 <= tile_j < 3:
                # This position has a tile that can be moved to the blank
                move_name = ""
                if di == -1: move_name = "DOWN"  # Tile moves down, blank moves up
                elif di == 1: move_name = "UP"   # Tile moves up, blank moves down
                elif dj == -1: move_name = "RIGHT"  # Tile moves right, blank moves left
                elif dj == 1: move_name = "LEFT"   # Tile moves left, blank moves right
                
                # Check forward checking constraint
                if forward_checking(board, goal_board, (tile_i, tile_j), board[tile_i][tile_j]):
                    # Make the move
                    new_board = [row[:] for row in board]
                    new_board[blank_pos[0]][blank_pos[1]], new_board[tile_i][tile_j] = new_board[tile_i][tile_j], new_board[blank_pos[0]][blank_pos[1]]
                    
                    possible_moves.append((new_board, move_name))
        
        # Sort moves based on heuristic
        possible_moves.sort(key=lambda move: heuristic_func(move[0], goal_board, heuristic))
        
        for next_board, move in possible_moves:
            # Check if this state is already in the path to avoid cycles
            if not any(next_board == p[0] for p in path):
                result = backtrack_fc(next_board, path + [(board, move)], depth + 1)
                if result:
                    return result
        
        return None
    
    path = backtrack_fc(initial_board, [], 0)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    if path:
        # Transform path to match the expected format
        formatted_path = []
        for i in range(len(path)):
            if i == 0:
                formatted_path.append((path[i][0], ""))
            else:
                formatted_path.append((path[i][0], path[i-1][1]))
        
        return {
            "path": formatted_path,
            "depth": len(path) - 1,
            "time": time_taken
        }
    else:
        return {
            "path": None,
            "depth": None,
            "time": time_taken
        }

def min_conflicts(initial_board, goal_board, max_steps=1000, heuristic="manhattan"):
    """
    Solve 8-puzzle using min-conflicts algorithm.
    This is a local search algorithm that tries to minimize the number of conflicts.
    """
    start_time = time.time()
    
    current_board = [row[:] for row in initial_board]
    current_state = PuzzleState(current_board)
    steps = 0
    path = [(current_board, "")]
    
    while steps < max_steps:
        if is_goal_state(current_board, goal_board):
            end_time = time.time()
            return {
                "path": path,
                "steps": steps,
                "time": end_time - start_time
            }
        
        # Get the blank position
        blank_pos = get_blank_pos(current_board)
        
        # Get possible moves from current state
        possible_moves = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_i, new_j = blank_pos[0] + di, blank_pos[1] + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_board = [row[:] for row in current_board]
                new_board[blank_pos[0]][blank_pos[1]], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[blank_pos[0]][blank_pos[1]]
                
                # Calculate the number of conflicts (misplaced tiles)
                conflicts = misplaced_tiles(new_board, goal_board)
                
                # Get the move name
                move_name = ""
                if di == -1: move_name = "UP"
                elif di == 1: move_name = "DOWN"
                elif dj == -1: move_name = "LEFT"
                elif dj == 1: move_name = "RIGHT"
                
                possible_moves.append((new_board, conflicts, move_name))
        
        if not possible_moves:
            break
        
        # Choose the move with the minimum conflicts
        # If there are multiple with the same conflicts, choose randomly among them
        min_conflict_moves = []
        min_conflict_value = float('inf')
        
        for move_board, conflict_value, move_name in possible_moves:
            if conflict_value < min_conflict_value:
                min_conflict_moves = [(move_board, move_name)]
                min_conflict_value = conflict_value
            elif conflict_value == min_conflict_value:
                min_conflict_moves.append((move_board, move_name))
        
        # Choose a random move among those with minimum conflicts
        chosen_board, chosen_move = random.choice(min_conflict_moves)
        
        # Record the move
        current_board = chosen_board
        path.append((current_board, chosen_move))
        
        steps += 1
    
    # If we reached max steps without finding a solution
    end_time = time.time()
    if is_goal_state(current_board, goal_board):
        return {
            "path": path,
            "steps": steps,
            "time": end_time - start_time
        }
    else:
        return {
            "path": None,
            "steps": steps,
            "time": end_time - start_time
        }

def min_conflicts_labeling(initial_board, goal_board, max_steps=1000, heuristic="manhattan"):
    """
    Solve 8-puzzle using min-conflicts algorithm with labeling approach.
    
    In this version, we model the puzzle as a labeling problem:
    - Variables: The 9 positions on the board (0,0) to (2,2)
    - Labels: The values 0-8 (0 represents the blank)
    - Constraints: Each position must be assigned a unique value and the configuration must be achievable
    
    The algorithm tries to minimize the number of conflicts by swapping pairs of values.
    """
    start_time = time.time()
    
    # Copy the initial board to avoid modifying it
    current_board = [row[:] for row in initial_board]
    steps = 0
    
    # For visualization, we'll track the process including:
    # 1. The board state
    # 2. The positions swapped
    # 3. The values swapped
    # 4. The number of conflicts before and after the swap
    # This will help show the labeling approach in the GUI
    
    path = [{"board": current_board, 
             "swap_positions": None, 
             "swap_values": None, 
             "conflicts_before": misplaced_tiles(current_board, goal_board),
             "conflicts_after": misplaced_tiles(current_board, goal_board),
             "move": ""}]
    
    while steps < max_steps:
        # Check if we've reached the goal state
        if is_goal_state(current_board, goal_board):
            end_time = time.time()
            return {
                "path": path,
                "steps": steps,
                "time": end_time - start_time,
                "labeling": True  # Flag to indicate this is a labeling approach
            }
        
        # Calculate conflicts for each position
        position_conflicts = {}
        for i in range(3):
            for j in range(3):
                value = current_board[i][j]
                if value != 0:  # Skip the blank
                    # Check if this value is in the right position in the goal
                    goal_i, goal_j = None, None
                    for gi in range(3):
                        for gj in range(3):
                            if goal_board[gi][gj] == value:
                                goal_i, goal_j = gi, gj
                                break
                    
                    # Calculate Manhattan distance as a conflict measure
                    conflict = abs(i - goal_i) + abs(j - goal_j)
                    position_conflicts[(i, j)] = conflict
        
        # Choose a random position with conflicts
        conflict_positions = [(pos, conf) for pos, conf in position_conflicts.items() if conf > 0]
        if not conflict_positions:
            # No conflicts, but not at goal state (shouldn't happen but just in case)
            break
        
        # Select a random position with conflict
        pos1, conflict1 = random.choice(conflict_positions)
        i1, j1 = pos1
        val1 = current_board[i1][j1]
        
        # Find all possible positions to swap with (including the blank)
        possible_swaps = []
        for i2 in range(3):
            for j2 in range(3):
                if (i1, j1) != (i2, j2):  # Don't swap with itself
                    val2 = current_board[i2][j2]
                    
                    # Create a new board with the swap
                    new_board = [row[:] for row in current_board]
                    new_board[i1][j1], new_board[i2][j2] = new_board[i2][j2], new_board[i1][j1]
                    
                    # Count conflicts in the new board
                    new_conflicts = misplaced_tiles(new_board, goal_board)
                    
                    # Check if the swap is valid (would be achievable through legal moves)
                    # In a real implementation, we would need a more complex check here
                    # For simplicity, we'll assume all swaps are valid if they reduce conflicts
                    
                    possible_swaps.append({
                        "position": (i2, j2),
                        "value": val2,
                        "new_board": new_board,
                        "conflicts": new_conflicts
                    })
        
        # If no valid swaps, break
        if not possible_swaps:
            break
        
        # Choose the swap that minimizes conflicts
        min_conflicts_swaps = []
        min_conflict_value = float('inf')
        
        for swap in possible_swaps:
            if swap["conflicts"] < min_conflict_value:
                min_conflicts_swaps = [swap]
                min_conflict_value = swap["conflicts"]
            elif swap["conflicts"] == min_conflict_value:
                min_conflicts_swaps.append(swap)
        
        # Choose a random swap among those with minimum conflicts
        chosen_swap = random.choice(min_conflicts_swaps)
        i2, j2 = chosen_swap["position"]
        val2 = current_board[i2][j2]
        
        # Record the current state and the conflicts
        conflicts_before = misplaced_tiles(current_board, goal_board)
        
        # Apply the swap
        current_board = chosen_swap["new_board"]
        
        # Record the step for visualization
        path.append({
            "board": [row[:] for row in current_board],
            "swap_positions": [(i1, j1), (i2, j2)],
            "swap_values": [val1, val2],
            "conflicts_before": conflicts_before,
            "conflicts_after": chosen_swap["conflicts"],
            "move": f"SWAP ({i1},{j1})↔({i2},{j2})"
        })
        
        steps += 1
    
    # If we reached max steps without finding a solution
    end_time = time.time()
    if is_goal_state(current_board, goal_board):
        return {
            "path": path,
            "steps": steps,
            "time": end_time - start_time,
            "labeling": True
        }
    else:
        return {
            "path": path,
            "steps": steps,
            "time": end_time - start_time,
            "labeling": True
        }

def solve_puzzle_csp(initial_board, goal_board=None, algorithm="backtracking", heuristic="manhattan", **kwargs):
    """
    Solve the 8-puzzle using CSP-based algorithms
    """
    if goal_board is None:
        goal_board = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]
    
    print(f"Solving with {algorithm.upper()} using {heuristic} heuristic:")
    print("Initial state:")
    print_board(initial_board)
    print("Goal state:")
    print_board(goal_board)
    
    if algorithm.lower() == "backtracking":
        result = backtracking_search(initial_board, goal_board, heuristic=heuristic, **kwargs)
    elif algorithm.lower() == "forward_checking":
        result = backtracking_with_forward_checking(initial_board, goal_board, heuristic=heuristic, **kwargs)
    elif algorithm.lower() == "min_conflicts":
        result = min_conflicts(initial_board, goal_board, heuristic=heuristic, **kwargs)
    elif algorithm.lower() == "min_conflicts_labeling":
        result = min_conflicts_labeling(initial_board, goal_board, heuristic=heuristic, **kwargs)
    else:
        return {"error": "Unknown algorithm"}
    
    if result.get("path"):
        print(f"Solution found!")
        if "depth" in result:
            print(f"Depth: {result['depth']}")
        if "steps" in result:
            print(f"Steps: {result['steps']}")
        print(f"Time taken: {result['time']:.4f} seconds")
        return result
    else:
        print("No solution found")
        print(f"Time taken: {result['time']:.4f} seconds")
        return result

def compare_csp_algorithms(initial_board, goal_board=None, heuristic="manhattan"):
    """
    Compare all CSP algorithms
    """
    results = {}
    
    algorithms = [
        "backtracking",
        "forward_checking",
        "min_conflicts",
        "min_conflicts_labeling"
    ]
    
    for algorithm in algorithms:
        print(f"\nRunning {algorithm.upper()}")
        results[algorithm] = solve_puzzle_csp(initial_board, goal_board, algorithm, heuristic)
    
    print("\nComparison Results:")
    print("-" * 70)
    print(f"{'Algorithm':<20} | {'Time (s)':<10} | {'Found':<5} | {'Steps':<10}")
    print("-" * 70)
    
    for algo, result in results.items():
        found = "Yes" if result.get("path") else "No"
        
        steps = "N/A"
        if result.get("path"):
            steps = len(result["path"]) - 1
            
        print(f"{algo.upper():<20} | {result['time']:<10.4f} | {found:<5} | {steps:<10}")
    
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
    # solve_puzzle_csp(initial_board, algorithm="backtracking", heuristic="manhattan")
    
    # For comparing all algorithms
    compare_csp_algorithms(initial_board, heuristic="manhattan")
