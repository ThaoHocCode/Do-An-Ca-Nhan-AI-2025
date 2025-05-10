import time
import random
import math
import copy
from collections import deque, defaultdict

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

# Non-deterministic search (AND-OR Tree Search)
class AndOrNode:
    def __init__(self, state, is_and_node=False):
        self.state = state
        self.is_and_node = is_and_node
        self.children = []
        self.action = None
        self.parent = None

def add_noise_to_action(action, noise_prob=0.2):
    """
    With probability noise_prob, change the action to a different random action
    """
    if random.random() < noise_prob:
        all_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        all_actions.remove(action)
        return random.choice(all_actions)
    return action

def and_or_tree_search(initial_state, goal_state, max_depth=10):
    """
    AND-OR Tree Search for non-deterministic environments
    In 8-puzzle, we simulate non-determinism by adding noise to actions
    """
    start_time = time.time()
    
    # Cache to avoid re-exploring same states
    visited = {}
    nodes_expanded = 0
    
    def or_search(state, depth):
        nonlocal nodes_expanded
        
        # Check if we've seen this state before
        state_hash = hash(str(state.board))
        if state_hash in visited:
            return visited[state_hash]
        
        # Check if we've reached the goal
        if state.is_goal(goal_state):
            return AndOrNode(state)
        
        # Check if we've reached the maximum depth
        if depth >= max_depth:
            return None
        
        nodes_expanded += 1
        
        # Create OR node
        or_node = AndOrNode(state)
        
        # Try each action
        for neighbor in state.get_neighbors():
            action = neighbor.move
            
            # Create AND node for the action
            and_node = AndOrNode(neighbor, is_and_node=True)
            and_node.action = action
            and_node.parent = or_node
            
            # With non-determinism, an action can lead to different states
            # Simulate this by considering the intended action and potential "noisy" actions
            results = []
            
            # Primary intended result
            intended_result = and_search(neighbor, depth + 1)
            if intended_result:
                results.append(intended_result)
            
            # Possible noisy results (simulate by considering alternative actions)
            for alt_action in ["UP", "DOWN", "LEFT", "RIGHT"]:
                if alt_action != action:
                    # Create a temporary state to simulate the effect of noise
                    i, j = state.blank_pos
                    di, dj = 0, 0
                    if alt_action == "UP": di = -1
                    elif alt_action == "DOWN": di = 1
                    elif alt_action == "LEFT": dj = -1
                    elif alt_action == "RIGHT": dj = 1
                    
                    new_i, new_j = i + di, j + dj
                    if 0 <= new_i < 3 and 0 <= new_j < 3:
                        new_board = [row[:] for row in state.board]
                        new_board[i][j], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[i][j]
                        noisy_state = PuzzleState(new_board, state, alt_action)
                        
                        # Apply reduced probability for noisy actions (0.1)
                        if random.random() < 0.1:
                            noisy_result = or_search(noisy_state, depth + 1)
                            if noisy_result:
                                results.append(noisy_result)
            
            # If all potential results are successful, this AND node is successful
            if results and len(results) > 0:
                and_node.children = results
                or_node.children.append(and_node)
        
        # If any of the actions lead to successful plans, this OR node is successful
        if or_node.children:
            visited[state_hash] = or_node
            return or_node
        
        visited[state_hash] = None
        return None
    
    def and_search(state, depth):
        # AND nodes succeed if all possible results succeed
        # In our case, we're simulating by considering the action to be successful
        # if it can lead to the goal despite potential noise
        return or_search(state, depth)
    
    # Start the search
    result = or_search(initial_state, 0)
    
    # Extract the solution path
    if result:
        solution_path = []
        def extract_path(node):
            if not node.children:
                # Tại nút cuối cùng, trả về trạng thái và nước đi
                # Nếu đây là nút goal, đảm bảo có move
                if not node.state.move:
                    return [(node.state.board, "")]
                return [(node.state.board, node.state.move)]
            
            # For OR nodes, choose the first successful action
            if not node.is_and_node and node.children:
                child = node.children[0]  # First successful AND node
                # OR nodes trả về board hiện tại rồi thêm các board của con
                # Nếu node hiện tại không có move (đặc biệt là nút gốc), dùng chuỗi rỗng
                move = node.state.move if node.state.move else ""
                return [(node.state.board, move)] + extract_path(child)
            
            # For AND nodes, include all potential next states (first one in our case)
            if node.is_and_node and node.children:
                child = node.children[0]  # First potential result
                # AND nodes trả về hành động của nó và các trạng thái tiếp theo
                return [(node.state.board, node.action)] + extract_path(child)
            
            # Đảm bảo luôn có giá trị trả về
            return [(node.state.board, node.state.move if node.state.move else "")]
        
        solution_path = extract_path(result)
        
        # Đảm bảo đường đi có đúng format và không có nước đi rỗng ở giữa
        formatted_path = []
        for i, (board, move) in enumerate(solution_path):
            if i == 0:  # Đối với trạng thái đầu tiên, move có thể là rỗng
                formatted_path.append((board, ""))
            elif not move:  # Nếu move rỗng ở giữa đường đi, thay bằng một move mặc định
                # Tìm hướng di chuyển dựa vào sự thay đổi của blank position
                prev_board = solution_path[i-1][0]
                prev_blank = next((i, j) for i in range(3) for j in range(3) if prev_board[i][j] == 0)
                curr_blank = next((i, j) for i in range(3) for j in range(3) if board[i][j] == 0)
                
                # Xác định hướng di chuyển
                di = curr_blank[0] - prev_blank[0]
                dj = curr_blank[1] - prev_blank[1]
                
                move = ""
                if di == -1: move = "UP"
                elif di == 1: move = "DOWN"
                elif dj == -1: move = "LEFT"
                elif dj == 1: move = "RIGHT"
                
                formatted_path.append((board, move))
            else:
                formatted_path.append((board, move))
        
        end_time = time.time()
        return {
            "path": formatted_path,
            "nodes_expanded": nodes_expanded,
            "time": end_time - start_time
        }
    else:
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "time": end_time - start_time
        }

# Partially observable environment simulation for 8-puzzle
def partially_observable_search(initial_state, goal_state, max_depth=20, observable_tiles=None):
    """
    Search in a partially observable environment
    observable_tiles: list of positions that are observable, rest are hidden
    """
    if observable_tiles is None:
        # Default: only the center and corners are observable
        observable_tiles = [(0,0), (0,2), (1,1), (2,0), (2,2)]
    
    start_time = time.time()
    
    # Create a partial observation function
    def get_observation(state):
        observation = [[-1 for _ in range(3)] for _ in range(3)]
        for i, j in observable_tiles:
            if 0 <= i < 3 and 0 <= j < 3:
                observation[i][j] = state.board[i][j]
        return observation
    
    # Initial observation
    initial_observation = get_observation(initial_state)
    
    # Frontier (states to explore)
    frontier = deque([(initial_state, 0)])  # (state, depth)
    
    # Keep track of visited observations to avoid cycles
    visited_observations = set()
    visited_observations.add(str(initial_observation))
    
    nodes_expanded = 0
    
    while frontier:
        state, depth = frontier.popleft()
        
        # Check if goal is reached
        if state.is_goal(goal_state):
            end_time = time.time()
            return {
                "path": state.get_path(),
                "nodes_expanded": nodes_expanded,
                "time": end_time - start_time
            }
        
        # Check depth limit
        if depth >= max_depth:
            continue
        
        nodes_expanded += 1
        
        # Get all possible next states
        for neighbor in state.get_neighbors():
            # Get the observation for this neighbor
            observation = get_observation(neighbor)
            
            # Check if this observation has been seen before
            obs_str = str(observation)
            if obs_str not in visited_observations:
                visited_observations.add(obs_str)
                frontier.append((neighbor, depth + 1))
    
    end_time = time.time()
    return {
        "path": None,
        "nodes_expanded": nodes_expanded,
        "time": end_time - start_time
    }

# Belief State Search with 4 simultaneous belief states
class BeliefState:
    def __init__(self, states=None):
        # A belief state contains a set of possible states
        self.states = states if states is not None else []
        self.parent = None
        self.action = None
        self.convergence_map = None  # Track which states have converged
    
    def __hash__(self):
        return hash(tuple(hash(state) for state in self.states))
    
    def __eq__(self, other):
        if not isinstance(other, BeliefState):
            return False
        return sorted([hash(s) for s in self.states]) == sorted([hash(s) for s in other.states])
    
    def update_with_observation(self, observation, full_state_space=None):
        # This would filter the states based on an observation
        # Not needed for basic 8-puzzle, but would be used in partially observable scenarios
        pass
    
    def get_next_belief_states(self, action):
        """
        Apply action to all states in the belief state
        Returns a new belief state
        """
        next_states = []
        
        # Apply the action to each state and collect all possible next states
        for state in self.states:
            # Determine all successor states for this state
            neighbors = state.get_neighbors()
            
            # Find the next state that results from the given action (if any)
            for neighbor in neighbors:
                if neighbor.move == action:
                    next_states.append(neighbor)
        
        next_belief = BeliefState(next_states)
        next_belief.parent = self
        next_belief.action = action
        
        # Map converged states for visualization
        next_belief.track_convergence()
        
        return next_belief
    
    def track_convergence(self):
        """
        Identifies which states have converged to the same configuration
        Creates a mapping for visualization purposes
        """
        if not self.states:
            self.convergence_map = []
            return
            
        # Create a map of unique states
        unique_states = {}
        convergence_map = [None] * len(self.states)
        
        for idx, state in enumerate(self.states):
            state_tuple = tuple(tuple(row) for row in state.board)
            
            if state_tuple in unique_states:
                # This state is a duplicate of a previous one
                convergence_map[idx] = unique_states[state_tuple]
            else:
                # This is a new unique state
                unique_states[state_tuple] = idx
        
        self.convergence_map = convergence_map
    
    def is_goal_belief(self, goal_state):
        """
        Check if all states in the belief state are goal states
        """
        return all(state.is_goal(goal_state) for state in self.states)
    
    def get_path(self):
        """
        Get the path from the initial belief state to this belief state
        """
        path = []
        current = self
        while current:
            # For visualization, we'll include up to 4 states from the belief state
            belief_sample = current.states[:4]
            boards = [state.board for state in belief_sample]
            
            # Ensure we always have exactly 4 boards shown (this is important for visualization)
            # If we have fewer than 4 states, duplicate existing ones rather than showing None
            if len(boards) < 4:
                # Duplicate existing boards to fill up to 4
                while len(boards) < 4:
                    if len(boards) > 0:
                        # Clone an existing board (with small variation if possible)
                        base_board = boards[0]
                        boards.append([row[:] for row in base_board])
                    else:
                        # This should never happen, but just in case
                        boards.append(None)
            
            # Add convergence information if available
            convergence_data = current.convergence_map if current.convergence_map else [None] * 4
            
            # Include convergence_map in the path for visualization
            path.append((boards, current.action, convergence_data))
            current = current.parent
        
        return list(reversed(path))

def belief_state_search(initial_states, goal_state, max_depth=10):
    """
    Belief State Search algorithm for 8-puzzle
    initial_states: list of possible initial states (representing uncertainty)
    """
    start_time = time.time()
    
    # Create initial belief state with 4 different possible initial states
    initial_belief = BeliefState(initial_states[:4])
    initial_belief.track_convergence()  # Track initial convergence
    
    # We'll use breadth-first search over belief states
    frontier = deque([(initial_belief, 0)])  # (belief_state, depth)
    visited = set([hash(initial_belief)])
    
    nodes_expanded = 0
    
    while frontier:
        belief, depth = frontier.popleft()
        
        # Check if this is a goal belief
        if belief.is_goal_belief(goal_state):
            end_time = time.time()
            return {
                "path": belief.get_path(),
                "nodes_expanded": nodes_expanded,
                "time": end_time - start_time
            }
        
        # Check depth limit
        if depth >= max_depth:
            continue
        
        nodes_expanded += 1
        
        # Try each action
        for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            next_belief = belief.get_next_belief_states(action)
            
            # Check if next belief state has any states
            if not next_belief.states:
                continue
            
            # Check if we've seen this belief state before
            next_belief_hash = hash(next_belief)
            if next_belief_hash not in visited:
                visited.add(next_belief_hash)
                frontier.append((next_belief, depth + 1))
    
    end_time = time.time()
    return {
        "path": None,
        "nodes_expanded": nodes_expanded,
        "time": end_time - start_time
    }

def generate_initial_belief_states(goal_board, num_states=4, max_moves=15):
    """
    Generate multiple initial states for belief state search by making random moves from the goal
    """
    states = []
    goal_state = PuzzleState(goal_board)
    
    for _ in range(num_states):
        current = PuzzleState([row[:] for row in goal_board])
        moves_made = 0
        
        # Make a random number of moves
        num_moves = random.randint(max_moves // 2, max_moves)
        
        while moves_made < num_moves:
            neighbors = current.get_neighbors()
            if not neighbors:
                break
            
            current = random.choice(neighbors)
            moves_made += 1
        
        # Ensure we don't duplicate states
        if not any(current.board == s.board for s in states):
            states.append(current)
    
    # If we have fewer than 4 states, add duplicates with small changes
    while len(states) < num_states:
        # Copy a random existing state
        base_state = random.choice(states)
        new_state = PuzzleState([row[:] for row in base_state.board])
        
        # Make a small random change if possible
        neighbors = new_state.get_neighbors()
        if neighbors:
            new_state = random.choice(neighbors)
            if not any(new_state.board == s.board for s in states):
                states.append(new_state)
    
    return states

def solve_puzzle_complex(initial_board, goal_board=None, algorithm="belief_state", heuristic=None, **kwargs):
    """
    Solve 8-puzzle using algorithms for complex environments
    """
    if goal_board is None:
        goal_board = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]
    
    initial_state = PuzzleState(initial_board)
    goal_state = PuzzleState(goal_board)
    
    print(f"Solving with {algorithm.upper()}:")
    print("Initial state:")
    print_board(initial_board)
    print("Goal state:")
    print_board(goal_board)
    
    if algorithm.lower() == "and_or_tree":
        result = and_or_tree_search(initial_state, goal_state, max_depth=kwargs.get('max_depth', 10))
    elif algorithm.lower() == "partially_observable":
        observable_tiles = kwargs.get('observable_tiles', [(0,0), (0,2), (1,1), (2,0), (2,2)])
        result = partially_observable_search(initial_state, goal_state, max_depth=kwargs.get('max_depth', 20), 
                                            observable_tiles=observable_tiles)
    elif algorithm.lower() == "belief_state":
        # Generate multiple possible initial states for belief state search
        initial_states = generate_initial_belief_states(goal_board, num_states=4)
        # Add the actual initial state to make sure it's considered
        initial_states[0] = initial_state
        
        result = belief_state_search(initial_states, goal_state, max_depth=kwargs.get('max_depth', 10))
    else:
        return {"error": "Unknown algorithm"}
    
    if result["path"]:
        # For belief state search, the path contains up to 4 boards per step
        if algorithm.lower() == "belief_state":
            path_length = len(result["path"]) - 1  # Exclude initial state
            print(f"Solution found in {path_length} moves")
            print(f"Belief states maintained: 4")
        else:
            path_length = len(result["path"]) - 1  # Exclude initial state
            print(f"Solution found in {path_length} moves")
        
        print(f"Nodes expanded: {result['nodes_expanded']}")
        print(f"Time taken: {result['time']:.4f} seconds")
        return result
    else:
        print("No solution found")
        return result

def compare_complex_algorithms(initial_board, goal_board=None):
    """
    Compare all algorithms for complex environments
    """
    results = {}
    
    algorithms = [
        "and_or_tree",
        "partially_observable",
        "belief_state"
    ]
    
    for algorithm in algorithms:
        print(f"\nRunning {algorithm.upper()}")
        results[algorithm] = solve_puzzle_complex(initial_board, goal_board, algorithm)
    
    print("\nComparison Results:")
    print("-" * 70)
    print(f"{'Algorithm':<20} | {'Nodes':<10} | {'Time (s)':<10} | {'Steps':<5}")
    print("-" * 70)
    
    for algo, result in results.items():
        if result.get("path"):
            steps = len(result["path"]) - 1
            print(f"{algo.upper():<20} | {result['nodes_expanded']:<10} | {result['time']:<10.4f} | {steps:<5}")
        else:
            print(f"{algo.upper():<20} | {result.get('nodes_expanded', 'N/A'):<10} | {result.get('time', 'N/A'):<10} | {'N/A':<5}")
    
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
    # solve_puzzle_complex(initial_board, algorithm="belief_state")
    
    # For comparing all algorithms
    compare_complex_algorithms(initial_board)
