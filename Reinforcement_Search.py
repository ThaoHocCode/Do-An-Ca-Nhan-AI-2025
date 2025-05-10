import time
import random
import numpy as np
import math
from collections import defaultdict, deque

# Thử import PyTorch, nếu không có thì tạo các giả lập cơ bản
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch không được cài đặt. Các thuật toán DQN và Policy Gradient sẽ không hoạt động.")
    print("Cài đặt PyTorch bằng lệnh: pip install torch")

class PuzzleState:
    def __init__(self, board, parent=None, move="", cost=0, depth=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.cost = cost
        self.depth = depth
        self.blank_pos = self.find_blank()
        
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

# Heuristic functions for reinforcement learning

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

def board_to_state(board):
    """Convert a board to a string representation for state lookup"""
    return ''.join(str(board[i][j]) for i in range(3) for j in range(3))

def state_to_board(state_str):
    """Convert a string state back to a 3x3 board"""
    board = [[0, 0, 0] for _ in range(3)]
    for i in range(9):
        board[i // 3][i % 3] = int(state_str[i])
    return board

# Q-learning algorithm for 8-puzzle
def q_learning(initial_state, goal_state, heuristic_func=manhattan_distance, episodes=1000, max_steps=100):
    """
    Q-learning algorithm to learn optimal policies for solving the 8-puzzle
    """
    start_time = time.time()
    
    # Parameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.3  # Exploration rate (initially higher)
    
    # Initialize Q-table
    Q = defaultdict(lambda: defaultdict(float))
    
    goal_board = goal_state.board
    goal_str = board_to_state(goal_board)
    
    # Statistics
    iterations = 0
    nodes_expanded = 0
    best_solution = None
    best_solution_length = float('inf')
    
    for episode in range(episodes):
        # Decay epsilon over time for exploitation
        epsilon = max(0.1, 0.3 - 0.2 * episode / episodes)
        
        # Reset state for new episode
        current_state = PuzzleState([row[:] for row in initial_state.board])
        current_str = board_to_state(current_state.board)
        
        done = False
        steps = 0
        path = [(current_state.board, "")]
        
        while not done and steps < max_steps:
            # Get possible actions from current state
            neighbors = current_state.get_neighbors()
            if not neighbors:
                break
                
            nodes_expanded += 1
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Exploration: random action
                next_state = random.choice(neighbors)
            else:
                # Exploitation: best action from Q-table
                state_str = board_to_state(current_state.board)
                best_value = float('-inf')
                candidates = []
                
                for neighbor in neighbors:
                    action = neighbor.move
                    q_value = Q[state_str][action]
                    if q_value > best_value:
                        best_value = q_value
                        candidates = [neighbor]
                    elif q_value == best_value:
                        candidates.append(neighbor)
                        
                next_state = random.choice(candidates) if candidates else random.choice(neighbors)
            
            iterations += 1
            steps += 1
            
            # Get reward (negative heuristic + large reward for reaching goal)
            next_str = board_to_state(next_state.board)
            if next_str == goal_str:
                reward = 100  # High reward for reaching goal
                done = True
            else:
                # Negative reward based on heuristic (higher heuristic = lower reward)
                reward = -heuristic_func(next_state.board, goal_board) / 10
            
            # Update Q-value using the Q-learning formula
            current_q = Q[current_str][next_state.move]
            
            # Find max Q-value for the next state
            max_next_q = 0
            for possible_next in next_state.get_neighbors():
                max_next_q = max(max_next_q, Q[next_str][possible_next.move])
            
            # Q-learning update formula
            new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
            Q[current_str][next_state.move] = new_q
            
            # Move to next state
            current_state = next_state
            current_str = next_str
            path.append((current_state.board, next_state.move))
            
            # Check if we found a better solution
            if done and steps < best_solution_length:
                best_solution = path
                best_solution_length = steps
        
    # Construct the best solution path
    if best_solution:
        end_state = PuzzleState(best_solution[-1][0])
        # Reconstruct parent references
        for i in range(len(best_solution) - 1, 0, -1):
            board = best_solution[i][0]
            move = best_solution[i][1]
            parent_board = best_solution[i-1][0]
            parent = PuzzleState(parent_board)
            end_state.parent = parent
            end_state = parent
    else:
        # If no solution was found, use the final policy to construct a path
        # Start from initial state
        current_state = PuzzleState([row[:] for row in initial_state.board])
        solution_path = [(current_state.board, "")]
        while True:
            state_str = board_to_state(current_state.board)
            
            # If we reached the goal, break
            if state_str == goal_str:
                break
                
            # Get all possible next states
            neighbors = current_state.get_neighbors()
            if not neighbors:
                break
                
            # Choose the action with highest Q-value
            best_action = None
            best_q = float('-inf')
            
            for neighbor in neighbors:
                q_value = Q[state_str][neighbor.move]
                if q_value > best_q:
                    best_q = q_value
                    best_action = neighbor
            
            if best_action is None:
                # If no best action found, break the loop
                break
                
            # Move to the next state
            current_state = best_action
            solution_path.append((current_state.board, current_state.move))
            
            # Prevent infinite loops
            if len(solution_path) > 100:
                break
                
        best_solution = solution_path
    
    end_time = time.time()
    
    if best_solution:
        # Convert solution format to match other algorithms
        result_state = PuzzleState(best_solution[-1][0])
        for i in range(len(best_solution) - 1, 0, -1):
            parent = PuzzleState(best_solution[i-1][0])
            result_state.parent = parent
            result_state.move = best_solution[i][1]
            result_state = parent
        
        # Reconstruct path
        path = []
        current = PuzzleState(best_solution[-1][0])
        current.parent = None
        for i in range(len(best_solution) - 1, 0, -1):
            current = PuzzleState(best_solution[i][0])
            parent = PuzzleState(best_solution[i-1][0])
            current.parent = parent
            current.move = best_solution[i][1]
            path.append((current.board, current.move))
            current = parent
        
        path.insert(0, (initial_state.board, ""))
        
        return {
            "path": path,
            "nodes_expanded": nodes_expanded,
            "iterations": iterations,
            "time": end_time - start_time
        }
    else:
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "iterations": iterations,
            "time": end_time - start_time
        }

# Deep Q Network implementation for 8-puzzle
class DQN(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, output_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def get_action_index(action):
    """Convert action name to index"""
    action_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    return action_map.get(action, 0)

def get_index_action(index):
    """Convert index to action name"""
    index_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    return index_map.get(index, "UP")

def board_to_tensor(board):
    """Convert a board to a tensor for the neural network"""
    # Flatten the board and normalize values
    flat = [board[i][j] / 8.0 for i in range(3) for j in range(3)]
    return torch.FloatTensor(flat)

def deep_q_network(initial_state, goal_state, heuristic_func=manhattan_distance, episodes=500, max_steps=100):
    """
    Deep Q Network (DQN) algorithm to learn optimal policies for solving the 8-puzzle
    """
    start_time = time.time()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameters
    gamma = 0.99  # Discount factor
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    batch_size = 64
    target_update = 10  # Update target network every N episodes
    
    # Initialize networks
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Initialize optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    # Initialize replay buffer
    memory = ReplayBuffer(10000)
    
    goal_board = goal_state.board
    goal_str = board_to_state(goal_board)
    
    # Statistics
    iterations = 0
    nodes_expanded = 0
    best_solution = None
    best_solution_length = float('inf')
    epsilon = epsilon_start
    
    # Mapping to convert action index to move name and valid actions
    action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    for episode in range(episodes):
        # Reset state for new episode
        current_state = PuzzleState([row[:] for row in initial_state.board])
        current_tensor = board_to_tensor(current_state.board).to(device)
        
        done = False
        total_reward = 0
        steps = 0
        path = [(current_state.board, "")]
        
        while not done and steps < max_steps:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Get valid actions for the current state
                neighbors = current_state.get_neighbors()
                if not neighbors:
                    break
                
                # Random action among valid ones
                next_state = random.choice(neighbors)
                action_idx = get_action_index(next_state.move)
            else:
                # Get Q-values from policy network
                with torch.no_grad():
                    q_values = policy_net(current_tensor)
                
                # Get valid actions for the current state
                neighbors = current_state.get_neighbors()
                if not neighbors:
                    break
                
                # Filter valid actions
                valid_actions = [get_action_index(n.move) for n in neighbors]
                
                # Set very low values for invalid actions
                for i in range(4):
                    if i not in valid_actions:
                        q_values[i] = float('-inf')
                
                # Choose best valid action
                action_idx = q_values.argmax().item()
                
                # Find the corresponding neighbor
                for n in neighbors:
                    if get_action_index(n.move) == action_idx:
                        next_state = n
                        break
                else:
                    # Fallback to random if something went wrong
                    next_state = random.choice(neighbors)
                    action_idx = get_action_index(next_state.move)
            
            nodes_expanded += 1
            iterations += 1
            steps += 1
            
            # Get reward
            next_str = board_to_state(next_state.board)
            next_tensor = board_to_tensor(next_state.board).to(device)
            
            if next_str == goal_str:
                reward = 100  # High reward for reaching goal
                done = True
            else:
                # Negative reward based on heuristic
                reward = -heuristic_func(next_state.board, goal_board) / 10
            
            total_reward += reward
            
            # Store transition in replay buffer
            memory.push(
                current_tensor,
                action_idx,
                reward,
                next_tensor,
                done
            )
            
            # Move to next state
            current_state = next_state
            current_tensor = next_tensor
            path.append((current_state.board, next_state.move))
            
            # Check if we found a better solution
            if done and steps < best_solution_length:
                best_solution = path
                best_solution_length = steps
            
            # Train the network
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch_state = torch.stack([t[0] for t in transitions])
                batch_action = torch.LongTensor([[t[1]] for t in transitions]).to(device)
                batch_reward = torch.FloatTensor([[t[2]] for t in transitions]).to(device)
                batch_next_state = torch.stack([t[3] for t in transitions])
                batch_done = torch.FloatTensor([[t[4]] for t in transitions]).to(device)
                
                # Compute current Q values
                current_q_values = policy_net(batch_state).gather(1, batch_action)
                
                # Compute next Q values
                with torch.no_grad():
                    max_next_q_values = target_net(batch_next_state).max(1, keepdim=True)[0]
                    target_q_values = batch_reward + gamma * max_next_q_values * (1 - batch_done)
                
                # Compute loss
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    # Use the trained policy network to find the best solution
    if best_solution is None:
        # Start from initial state
        current_state = PuzzleState([row[:] for row in initial_state.board])
        current_tensor = board_to_tensor(current_state.board).to(device)
        path = [(current_state.board, "")]
        steps = 0
        
        while steps < 100:  # Limit to prevent infinite loops
            with torch.no_grad():
                q_values = policy_net(current_tensor)
            
            # Get valid actions
            neighbors = current_state.get_neighbors()
            if not neighbors:
                break
            
            # Filter valid actions
            valid_actions = [get_action_index(n.move) for n in neighbors]
            
            # Set very low values for invalid actions
            for i in range(4):
                if i not in valid_actions:
                    q_values[i] = float('-inf')
            
            # Choose best valid action
            action_idx = q_values.argmax().item()
            
            # Find the corresponding neighbor
            for n in neighbors:
                if get_action_index(n.move) == action_idx:
                    next_state = n
                    break
            else:
                break  # No valid action found
            
            steps += 1
            
            # Move to next state
            current_state = next_state
            current_tensor = board_to_tensor(current_state.board).to(device)
            path.append((current_state.board, next_state.move))
            
            # Check if goal reached
            if current_state.is_goal(goal_state):
                best_solution = path
                break
    
    end_time = time.time()
    
    if best_solution:
        return {
            "path": best_solution,
            "nodes_expanded": nodes_expanded,
            "iterations": iterations,
            "time": end_time - start_time
        }
    else:
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "iterations": iterations,
            "time": end_time - start_time
        }

# SARSA algorithm for 8-puzzle
def sarsa(initial_state, goal_state, heuristic_func=manhattan_distance, episodes=1000, max_steps=100):
    """
    SARSA (State-Action-Reward-State-Action) algorithm for the 8-puzzle
    """
    start_time = time.time()
    
    # Parameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon_start = 0.3  # Initial exploration rate
    epsilon_min = 0.1    # Minimum exploration rate
    epsilon_decay = 0.995  # Decay rate for exploration
    
    # Initialize Q-table
    Q = defaultdict(lambda: defaultdict(float))
    
    goal_board = goal_state.board
    goal_str = board_to_state(goal_board)
    
    # Statistics
    iterations = 0
    nodes_expanded = 0
    best_solution = None
    best_solution_length = float('inf')
    
    for episode in range(episodes):
        # Decay epsilon over time
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
        
        # Reset state for new episode
        current_state = PuzzleState([row[:] for row in initial_state.board])
        current_str = board_to_state(current_state.board)
        
        # Choose first action using epsilon-greedy
        neighbors = current_state.get_neighbors()
        if not neighbors:
            continue
            
        if random.random() < epsilon:
            # Exploration: random action
            next_state = random.choice(neighbors)
        else:
            # Exploitation: best action from Q-table
            best_value = float('-inf')
            candidates = []
            
            for neighbor in neighbors:
                action = neighbor.move
                q_value = Q[current_str][action]
                if q_value > best_value:
                    best_value = q_value
                    candidates = [neighbor]
                elif q_value == best_value:
                    candidates.append(neighbor)
                    
            next_state = random.choice(candidates) if candidates else random.choice(neighbors)
        
        current_action = next_state.move
        
        done = False
        steps = 0
        path = [(current_state.board, "")]
        
        while not done and steps < max_steps:
            # Take action
            next_str = board_to_state(next_state.board)
            
            # Check if goal reached
            if next_str == goal_str:
                reward = 100  # High reward for reaching goal
                done = True
            else:
                # Negative reward based on heuristic
                reward = -heuristic_func(next_state.board, goal_board) / 10
            
            iterations += 1
            nodes_expanded += 1
            
            # Get next state and action
            if not done:
                possible_neighbors = next_state.get_neighbors()
                if not possible_neighbors:
                    break
                
                # Choose next action using epsilon-greedy
                if random.random() < epsilon:
                    # Exploration: random action
                    next_next_state = random.choice(possible_neighbors)
                else:
                    # Exploitation: best action from Q-table
                    best_value = float('-inf')
                    candidates = []
                    
                    for neighbor in possible_neighbors:
                        action = neighbor.move
                        q_value = Q[next_str][action]
                        if q_value > best_value:
                            best_value = q_value
                            candidates = [neighbor]
                        elif q_value == best_value:
                            candidates.append(neighbor)
                            
                    next_next_state = random.choice(candidates) if candidates else random.choice(possible_neighbors)
                
                next_action = next_next_state.move
            else:
                next_action = None
            
            # SARSA update formula: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            current_q = Q[current_str][current_action]
            
            if done:
                target = reward
            else:
                target = reward + gamma * Q[next_str][next_action]
            
            # Update Q-value
            Q[current_str][current_action] = current_q + alpha * (target - current_q)
            
            # Move to next state
            current_state = next_state
            current_str = next_str
            current_action = next_action
            path.append((current_state.board, current_state.move))
            
            if not done and next_action:
                for neighbor in possible_neighbors:
                    if neighbor.move == next_action:
                        next_state = neighbor
                        break
            
            steps += 1
            
            # Check if we found a better solution
            if done and steps < best_solution_length:
                best_solution = path
                best_solution_length = steps
    
    # If no solution found during training, use the learned policy to find one
    if best_solution is None:
        # Start from initial state
        current_state = PuzzleState([row[:] for row in initial_state.board])
        solution_path = [(current_state.board, "")]
        steps = 0
        
        while steps < 100:  # Limit to prevent infinite loops
            current_str = board_to_state(current_state.board)
            
            # Check if goal reached
            if current_str == goal_str:
                best_solution = solution_path
                break
            
            # Get possible next states
            neighbors = current_state.get_neighbors()
            if not neighbors:
                break
            
            # Choose the action with highest Q-value
            best_action = None
            best_q = float('-inf')
            
            for neighbor in neighbors:
                q_value = Q[current_str][neighbor.move]
                if q_value > best_q:
                    best_q = q_value
                    best_action = neighbor
            
            if best_action is None:
                # If no best action found, break
                break
            
            # Move to next state
            current_state = best_action
            solution_path.append((current_state.board, current_state.move))
            steps += 1
        
        if current_str == goal_str:
            best_solution = solution_path
    
    end_time = time.time()
    
    if best_solution:
        return {
            "path": best_solution,
            "nodes_expanded": nodes_expanded,
            "iterations": iterations,
            "time": end_time - start_time
        }
    else:
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "iterations": iterations,
            "time": end_time - start_time
        }

# Policy Gradient Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, output_size=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

def select_action(policy_net, state_tensor, valid_actions):
    """Select an action using the policy network with probabilities filtered by valid actions"""
    with torch.no_grad():
        probs = policy_net(state_tensor)
    
    # Set probabilities of invalid actions to 0
    action_mask = torch.zeros_like(probs)
    for action_idx in valid_actions:
        action_mask[action_idx] = 1
    
    masked_probs = probs * action_mask
    
    # If all masked probabilities are 0, use a uniform distribution over valid actions
    if masked_probs.sum().item() == 0:
        masked_probs = action_mask / len(valid_actions)
    else:
        # Normalize probabilities
        masked_probs = masked_probs / masked_probs.sum()
    
    # Sample action from the probability distribution
    action_idx = torch.multinomial(masked_probs, 1).item()
    return action_idx, masked_probs[action_idx].item()

def policy_gradient(initial_state, goal_state, heuristic_func=manhattan_distance, episodes=500, max_steps=100):
    """
    Policy Gradient algorithm (REINFORCE) for solving the 8-puzzle
    """
    if not TORCH_AVAILABLE:
        print("Policy Gradient cần PyTorch để chạy. Hãy cài đặt PyTorch và thử lại.")
        return {"path": None, "nodes_expanded": 0, "iterations": 0, "time": 0}
        
    start_time = time.time()
    
    try:
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy network
        policy_net = PolicyNetwork().to(device)
        optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
        
        goal_board = goal_state.board
        goal_str = board_to_state(goal_board)
        
        # Statistics
        iterations = 0
        nodes_expanded = 0
        best_solution = None
        best_solution_length = float('inf')
        
        for episode in range(episodes):
            # Reset for new episode
            current_state = PuzzleState([row[:] for row in initial_state.board])
            current_tensor = board_to_tensor(current_state.board).to(device)
            
            # Storage for this episode
            saved_log_probs = []  # Lưu log probabilities
            rewards = []
            path = [(current_state.board, "")]
            
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                # Get valid actions
                neighbors = current_state.get_neighbors()
                if not neighbors:
                    break
                
                valid_actions = [get_action_index(n.move) for n in neighbors]
                
                try:
                    # Get probabilities from policy network
                    probs = policy_net(current_tensor)
                    
                    # Create action mask for valid actions
                    action_mask = torch.zeros_like(probs)
                    for action_idx in valid_actions:
                        action_mask[action_idx] = 1
                    
                    # Apply mask to probabilities
                    masked_probs = probs * action_mask
                    if masked_probs.sum().item() == 0:
                        masked_probs = action_mask / len(valid_actions)
                    else:
                        masked_probs = masked_probs / masked_probs.sum()
                    
                    # Sample action using the probability distribution
                    m = torch.distributions.Categorical(masked_probs)
                    action_idx = m.sample()
                    
                    # Save log probability for gradient calculation
                    saved_log_probs.append(m.log_prob(action_idx))
                    
                    # Find the corresponding neighbor
                    next_state = None
                    for n in neighbors:
                        if get_action_index(n.move) == action_idx.item():
                            next_state = n
                            break
                    
                    if next_state is None:
                        # Fallback to a random valid action if something went wrong
                        next_state = random.choice(neighbors)
                    
                    steps += 1
                    nodes_expanded += 1
                    iterations += 1
                    
                    # Get reward
                    next_str = board_to_state(next_state.board)
                    next_tensor = board_to_tensor(next_state.board).to(device)
                    
                    if next_str == goal_str:
                        reward = 100  # High reward for reaching goal
                        done = True
                    else:
                        # Smaller penalty for each step and based on heuristic value
                        reward = -0.1 - heuristic_func(next_state.board, goal_board) / 20
                    
                    rewards.append(reward)
                    
                    # Move to next state
                    current_state = next_state
                    current_tensor = next_tensor
                    path.append((current_state.board, next_state.move))
                    
                    # Check if we found a better solution
                    if done and steps < best_solution_length:
                        best_solution = path
                        best_solution_length = steps
                
                except Exception as e:
                    print(f"Lỗi trong vòng lặp: {e}")
                    break
            
            # If episode didn't end with goal state
            if not done and rewards:
                rewards[-1] = -1  # Penalty for not reaching goal
            
            # Calculate policy gradient
            if rewards and saved_log_probs:
                try:
                    # Calculate discounted rewards
                    R = 0
                    returns = []
                    gamma = 0.99
                    
                    for r in reversed(rewards):
                        R = r + gamma * R
                        returns.insert(0, R)
                    
                    returns = torch.tensor(returns, dtype=torch.float32)
                    
                    # Normalize returns
                    if len(returns) > 1:
                        std = returns.std()
                        if std > 0:  # Avoid division by zero
                            returns = (returns - returns.mean()) / (std + 1e-9)
                    
                    # Calculate policy loss
                    policy_loss = 0
                    for log_prob, R in zip(saved_log_probs, returns):
                        policy_loss += -log_prob * R
                    
                    # Optimize policy
                    optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    optimizer.step()
                    
                except Exception as e:
                    print(f"Lỗi trong tính toán loss: {e}")
                    print(f"  saved_log_probs: {len(saved_log_probs)}, returns: {len(returns)}")
                    continue
        
        # Use the trained policy to find a solution
        if best_solution is None:
            # Start from initial state
            current_state = PuzzleState([row[:] for row in initial_state.board])
            current_tensor = board_to_tensor(current_state.board).to(device)
            path = [(current_state.board, "")]
            steps = 0
            
            while steps < 100:  # Limit to prevent infinite loops
                try:
                    # Get valid actions
                    neighbors = current_state.get_neighbors()
                    if not neighbors:
                        break
                    
                    valid_actions = [get_action_index(n.move) for n in neighbors]
                    
                    # Select best action (no random sampling this time)
                    with torch.no_grad():
                        probs = policy_net(current_tensor)
                        
                        # Mask invalid actions
                        action_mask = torch.zeros_like(probs)
                        for action_idx in valid_actions:
                            action_mask[action_idx] = 1
                        
                        masked_probs = probs * action_mask
                        action_idx = masked_probs.argmax().item()
                    
                    # Find the corresponding neighbor
                    next_state = None
                    for n in neighbors:
                        if get_action_index(n.move) == action_idx:
                            next_state = n
                            break
                    
                    if next_state is None:
                        break
                    
                    steps += 1
                    
                    # Move to next state
                    current_state = next_state
                    current_tensor = board_to_tensor(next_state.board).to(device)
                    path.append((current_state.board, next_state.move))
                    
                    # Check if goal reached
                    if current_state.is_goal(goal_state):
                        best_solution = path
                        break
                except Exception as e:
                    print(f"Lỗi trong thời gian suy luận: {e}")
                    break
        
        end_time = time.time()
        
        if best_solution:
            return {
                "path": best_solution,
                "nodes_expanded": nodes_expanded,
                "iterations": iterations,
                "time": end_time - start_time
            }
        else:
            return {
                "path": None,
                "nodes_expanded": nodes_expanded,
                "iterations": iterations,
                "time": end_time - start_time
            }
    except Exception as e:
        print(f"Lỗi chung trong Policy Gradient: {e}")
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": 0,
            "iterations": 0,
            "time": end_time - start_time
        }

def solve_puzzle_reinforcement(initial_board, goal_board=None, algorithm="q_learning", heuristic="manhattan"):
    """
    Solve the 8-puzzle using reinforcement learning algorithms
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
    else:
        heuristic_func = manhattan_distance
    
    print(f"Solving with {algorithm.upper()} using {heuristic} heuristic:")
    print("Initial state:")
    print_board(initial_board)
    print("Goal state:")
    print_board(goal_board)
    
    # Adjust parameters based on algorithm
    if algorithm.lower() == "q_learning":
        result = q_learning(initial_state, goal_state, heuristic_func, episodes=1000, max_steps=100)
    elif algorithm.lower() == "dqn":
        result = deep_q_network(initial_state, goal_state, heuristic_func, episodes=500, max_steps=100)
    elif algorithm.lower() == "sarsa":
        result = sarsa(initial_state, goal_state, heuristic_func, episodes=1000, max_steps=100)
    elif algorithm.lower() == "policy_gradient":
        result = policy_gradient(initial_state, goal_state, heuristic_func, episodes=500, max_steps=100)
    else:
        return {"error": "Unknown algorithm"}
    
    if result["path"]:
        print(f"Solution found in {len(result['path']) - 1} moves")
        print(f"Nodes expanded: {result['nodes_expanded']}")
        print(f"Iterations: {result.get('iterations', 'N/A')}")
        print(f"Time taken: {result['time']:.4f} seconds")
        return result
    else:
        print("No solution found")
        return result

def compare_reinforcement_algorithms(initial_board, goal_board=None, heuristic="manhattan"):
    """
    Compare all reinforcement learning algorithms
    """
    results = {}
    
    for algorithm in ["q_learning", "sarsa", "dqn", "policy_gradient"]:
        print(f"\nRunning {algorithm.upper()}")
        results[algorithm] = solve_puzzle_reinforcement(initial_board, goal_board, algorithm, heuristic)
    
    print("\nComparison Results:")
    print("-" * 80)
    print(f"{'Algorithm':<15} | {'Nodes':<10} | {'Iterations':<12} | {'Time (s)':<10} | {'Steps':<5}")
    print("-" * 80)
    
    for algo, result in results.items():
        if result.get("path"):
            steps = len(result["path"]) - 1
            print(f"{algo.upper():<15} | {result['nodes_expanded']:<10} | {result.get('iterations', 'N/A'):<12} | {result['time']:<10.4f} | {steps:<5}")
        else:
            print(f"{algo.upper():<15} | {result.get('nodes_expanded', 'N/A'):<10} | {result.get('iterations', 'N/A'):<12} | {result.get('time', 'N/A'):<10} | {'N/A':<5}")
    
    print("-" * 80)

# Example usage:
if __name__ == "__main__":
    # Example initial state (solvable)
    initial_board = [
        [1, 2, 3],
        [4, 0, 6],
        [7, 5, 8]
    ]
    
    # For testing a single algorithm
    # solve_puzzle_reinforcement(initial_board, algorithm="q_learning", heuristic="manhattan")
    
    # For comparing all algorithms with a specific heuristic
    compare_reinforcement_algorithms(initial_board, heuristic="manhattan")
