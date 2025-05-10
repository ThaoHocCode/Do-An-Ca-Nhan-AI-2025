import time
import random
import math
import copy
import heapq
from collections import deque
import numpy as np

class PuzzleState:
    def __init__(self, board, parent=None, move="", cost=0, depth=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.cost = cost
        self.depth = depth
        self.blank_pos = self.find_blank()
        
    def __lt__(self, other):
        # For priority queue-based algorithms
        return self.cost < other.cost
        
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

# Heuristic functions for local search

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

# Local Search Algorithms

def simple_hill_climbing(initial_board, goal_board, max_iterations=1000, heuristic="manhattan"):
    """
    Simple Hill Climbing algorithm - moves to the first neighbor that improves the current state
    """
    start_time = time.time()
    
    current_state = PuzzleState(initial_board)
    current_value = heuristic_func(current_state.board, goal_board, heuristic)
    
    iterations = 0
    nodes_expanded = 0
    improvement_found = True
    
    while iterations < max_iterations and improvement_found:
        improvement_found = False
        iterations += 1
        
        neighbors = current_state.get_neighbors()
        nodes_expanded += len(neighbors)
        
        for neighbor in neighbors:
            neighbor_value = heuristic_func(neighbor.board, goal_board, heuristic)
            
            if neighbor_value < current_value:
                current_state = neighbor
                current_value = neighbor_value
                improvement_found = True
                break
        
        if current_value == 0:  # Goal state reached
            end_time = time.time()
            return {
                "path": current_state.get_path(),
                "nodes_expanded": nodes_expanded,
                "iterations": iterations,
                "time": end_time - start_time
            }
    
    end_time = time.time()
    return {
        "path": current_state.get_path() if current_value == 0 else None,
        "nodes_expanded": nodes_expanded,
        "iterations": iterations,
        "time": end_time - start_time,
        "final_state": current_state.board,
        "final_value": current_value
    }

def steepest_ascent_hill_climbing(initial_board, goal_board, max_iterations=1000, heuristic="manhattan"):
    """
    Steepest Ascent Hill Climbing algorithm - moves to the best neighbor
    """
    start_time = time.time()
    
    current_state = PuzzleState(initial_board)
    current_value = heuristic_func(current_state.board, goal_board, heuristic)
    
    iterations = 0
    nodes_expanded = 0
    improvement_found = True
    
    while iterations < max_iterations and improvement_found:
        improvement_found = False
        iterations += 1
        
        neighbors = current_state.get_neighbors()
        nodes_expanded += len(neighbors)
        
        best_neighbor = None
        best_value = current_value
        
        for neighbor in neighbors:
            neighbor_value = heuristic_func(neighbor.board, goal_board, heuristic)
            
            if neighbor_value < best_value:
                best_neighbor = neighbor
                best_value = neighbor_value
        
        if best_value < current_value:
            current_state = best_neighbor
            current_value = best_value
            improvement_found = True
        
        if current_value == 0:  # Goal state reached
            end_time = time.time()
            return {
                "path": current_state.get_path(),
                "nodes_expanded": nodes_expanded,
                "iterations": iterations,
                "time": end_time - start_time
            }
    
    end_time = time.time()
    return {
        "path": current_state.get_path() if current_value == 0 else None,
        "nodes_expanded": nodes_expanded,
        "iterations": iterations,
        "time": end_time - start_time,
        "final_state": current_state.board,
        "final_value": current_value
    }

def stochastic_hill_climbing(initial_board, goal_board, max_iterations=1000, heuristic="manhattan"):
    """
    Stochastic Hill Climbing algorithm - probabilistically selects neighbors with higher probability for better states
    """
    start_time = time.time()
    
    current_state = PuzzleState(initial_board)
    current_value = heuristic_func(current_state.board, goal_board, heuristic)
    
    iterations = 0
    nodes_expanded = 0
    
    while iterations < max_iterations:
        iterations += 1
        
        neighbors = current_state.get_neighbors()
        nodes_expanded += len(neighbors)
        
        # Filter neighbors that are better than current
        better_neighbors = []
        for neighbor in neighbors:
            neighbor_value = heuristic_func(neighbor.board, goal_board, heuristic)
            if neighbor_value <= current_value:
                # Store neighbor and its value
                better_neighbors.append((neighbor, neighbor_value))
        
        if not better_neighbors:
            # No better neighbors, reached local optimum
            break
        
        # Calculate selection probabilities (better states have higher probability)
        # Invert values so lower heuristic values get higher probability
        values = [1.0 / (value + 1) for _, value in better_neighbors]  # Add 1 to avoid division by zero
        total = sum(values)
        probabilities = [value / total for value in values]
        
        # Select a neighbor based on probabilities
        index = random.choices(range(len(better_neighbors)), probabilities)[0]
        selected_neighbor, selected_value = better_neighbors[index]
        
        current_state = selected_neighbor
        current_value = selected_value
        
        if current_value == 0:  # Goal state reached
            end_time = time.time()
            return {
                "path": current_state.get_path(),
                "nodes_expanded": nodes_expanded,
                "iterations": iterations,
                "time": end_time - start_time
            }
    
    end_time = time.time()
    return {
        "path": current_state.get_path() if current_value == 0 else None,
        "nodes_expanded": nodes_expanded,
        "iterations": iterations,
        "time": end_time - start_time,
        "final_state": current_state.board,
        "final_value": current_value
    }

def simulated_annealing(initial_board, goal_board, max_iterations=10000, initial_temp=1.0, cooling_rate=0.995, heuristic="manhattan"):
    """
    Simulated Annealing algorithm - allows for hill-climbing but occasionally accepts worse solutions based on temperature
    """
    start_time = time.time()
    
    current_state = PuzzleState(initial_board)
    current_value = heuristic_func(current_state.board, goal_board, heuristic)
    
    temperature = initial_temp
    iterations = 0
    nodes_expanded = 0
    
    while iterations < max_iterations and temperature > 0.01:
        iterations += 1
        
        neighbors = current_state.get_neighbors()
        nodes_expanded += len(neighbors)
        
        if not neighbors:
            break
            
        # Select a random neighbor
        next_state = random.choice(neighbors)
        next_value = heuristic_func(next_state.board, goal_board, heuristic)
        
        # Calculate acceptance probability
        delta_e = current_value - next_value  # We want to minimize, so reverse the sign
        
        # If it's better, accept it; if not, accept with a probability
        if delta_e > 0 or random.random() < math.exp(delta_e / temperature):
            current_state = next_state
            current_value = next_value
        
        # Cool the system
        temperature *= cooling_rate
        
        if current_value == 0:  # Goal state reached
            end_time = time.time()
            return {
                "path": current_state.get_path(),
                "nodes_expanded": nodes_expanded,
                "iterations": iterations,
                "time": end_time - start_time
            }
    
    end_time = time.time()
    return {
        "path": current_state.get_path() if current_value == 0 else None,
        "nodes_expanded": nodes_expanded,
        "iterations": iterations,
        "time": end_time - start_time,
        "final_state": current_state.board,
        "final_value": current_value
    }

def beam_search(initial_board, goal_board, beam_width=10, max_iterations=1000, heuristic="manhattan"):
    """
    Beam Search algorithm - keeps track of k best states at each iteration
    """
    start_time = time.time()
    
    initial_state = PuzzleState(initial_board)
    current_states = [initial_state]
    
    iterations = 0
    nodes_expanded = 0
    
    while iterations < max_iterations and current_states:
        iterations += 1
        
        next_states = []
        
        for state in current_states:
            if heuristic_func(state.board, goal_board, heuristic) == 0:
                # Goal found
                end_time = time.time()
                return {
                    "path": state.get_path(),
                    "nodes_expanded": nodes_expanded,
                    "iterations": iterations,
                    "time": end_time - start_time
                }
            
            neighbors = state.get_neighbors()
            nodes_expanded += len(neighbors)
            next_states.extend(neighbors)
        
        # Keep only the k best states
        if next_states:
            next_states.sort(key=lambda x: heuristic_func(x.board, goal_board, heuristic))
            current_states = next_states[:beam_width]
    
    # If we reach here, no solution was found within the iterations limit
    end_time = time.time()
    
    # Return the best state found, if any
    if current_states:
        best_state = min(current_states, key=lambda x: heuristic_func(x.board, goal_board, heuristic))
        best_value = heuristic_func(best_state.board, goal_board, heuristic)
        
        return {
            "path": best_state.get_path() if best_value == 0 else None,
            "nodes_expanded": nodes_expanded,
            "iterations": iterations,
            "time": end_time - start_time,
            "final_state": best_state.board,
            "final_value": best_value
        }
    else:
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "iterations": iterations,
            "time": end_time - start_time
        }

# Genetic Algorithm functions

def generate_random_board():
    """Generate a random 8-puzzle board"""
    numbers = list(range(9))  # 0-8
    random.shuffle(numbers)
    return [numbers[i:i+3] for i in range(0, 9, 3)]

def is_solvable(board):
    """Check if a board configuration is solvable"""
    # Flatten the board and remove the blank (0)
    flat_board = [num for row in board for num in row if num != 0]
    
    # Count inversions
    inversions = 0
    for i in range(len(flat_board)):
        for j in range(i+1, len(flat_board)):
            if flat_board[i] > flat_board[j]:
                inversions += 1
                
    # For a 3x3 board, the puzzle is solvable if the number of inversions is even
    return inversions % 2 == 0

def generate_initial_population(pop_size, goal_board):
    """Generate an initial population of solvable boards"""
    population = []
    while len(population) < pop_size:
        board = generate_random_board()
        if is_solvable(board) == is_solvable(goal_board):
            population.append(board)
    return population

def fitness(board, goal_board, heuristic="manhattan"):
    """Fitness function for GA - higher is better"""
    # Inverse of the heuristic function (since we want to maximize fitness)
    h_value = heuristic_func(board, goal_board, heuristic)
    return 1.0 / (h_value + 1)  # Add 1 to avoid division by zero

def selection(population, fitnesses, num_parents):
    """Select parents using tournament selection"""
    parents = []
    for _ in range(num_parents):
        # Tournament selection
        tournament_size = 3
        if len(population) <= tournament_size:
            tournament_indices = list(range(len(population)))
        else:
            tournament_indices = random.sample(range(len(population)), tournament_size)
        
        tournament_fitness = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        parents.append(population[winner_idx])
    
    return parents

def crossover(parent1, parent2):
    """Perform crossover operation between two parents"""
    # Convert boards to 1D for easier manipulation
    parent1_flat = [num for row in parent1 for num in row]
    parent2_flat = [num for row in parent2 for num in row]
    
    # Choose a random crossover point
    crossover_point = random.randint(1, 7)
    
    # Create child combining parts from both parents
    child1_flat = parent1_flat[:crossover_point] + parent2_flat[crossover_point:]
    
    # Ensure the child has all numbers 0-8 exactly once
    # Track missing and duplicate numbers
    counts = [0] * 9
    for num in child1_flat:
        counts[num] += 1
    
    missing = [i for i, count in enumerate(counts) if count == 0]
    duplicates = [i for i, count in enumerate(counts) if count > 1]
    
    # Replace duplicates with missing numbers
    for i in range(len(child1_flat)):
        if child1_flat[i] in duplicates and counts[child1_flat[i]] > 1:
            missing_num = missing.pop(0)
            counts[child1_flat[i]] -= 1
            child1_flat[i] = missing_num
    
    # Convert back to 2D board
    child = [child1_flat[i:i+3] for i in range(0, 9, 3)]
    
    # Check if the child is solvable
    if is_solvable(child) != is_solvable(parent1):
        # Swap two non-blank tiles to change parity
        positions = [(i, j) for i in range(3) for j in range(3) if child[i][j] != 0]
        pos1, pos2 = random.sample(positions, 2)
        child[pos1[0]][pos1[1]], child[pos2[0]][pos2[1]] = child[pos2[0]][pos2[1]], child[pos1[0]][pos1[1]]
    
    return child

def mutation(board, mutation_rate=0.2):
    """Randomly swap two adjacent tiles with probability mutation_rate"""
    if random.random() > mutation_rate:
        return board
    
    # Create a copy of the board
    new_board = [row[:] for row in board]
    
    # Find the blank position
    blank_pos = None
    for i in range(3):
        for j in range(3):
            if new_board[i][j] == 0:
                blank_pos = (i, j)
                break
        if blank_pos:
            break
    
    # Get valid moves from blank position
    moves = []
    i, j = blank_pos
    if i > 0: moves.append((-1, 0))  # Up
    if i < 2: moves.append((1, 0))   # Down
    if j > 0: moves.append((0, -1))  # Left
    if j < 2: moves.append((0, 1))   # Right
    
    # Choose a random move
    di, dj = random.choice(moves)
    new_i, new_j = i + di, j + dj
    
    # Swap blank with the selected adjacent tile
    new_board[i][j], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[i][j]
    
    return new_board

def genetic_algorithm(initial_board, goal_board, pop_size=100, max_generations=100, 
                    mutation_rate=0.2, heuristic="manhattan"):
    """
    Genetic Algorithm for solving 8-puzzle
    """
    start_time = time.time()
    
    # Create initial population
    population = [initial_board]  # Include the initial board
    
    # Fill the rest of the population with random solvable boards
    while len(population) < pop_size:
        board = generate_random_board()
        if is_solvable(board) == is_solvable(goal_board):
            population.append(board)
    
    best_solution = None
    best_fitness = 0
    generations_without_improvement = 0
    
    for generation in range(max_generations):
        # Calculate fitness for each individual
        fitnesses = [fitness(board, goal_board, heuristic) for board in population]
        
        # Find the best individual
        max_fitness_idx = fitnesses.index(max(fitnesses))
        current_best = population[max_fitness_idx]
        current_best_fitness = fitnesses[max_fitness_idx]
        
        # Check if this is the best solution so far
        if current_best_fitness > best_fitness:
            best_solution = current_best
            best_fitness = current_best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        # Check if we've found the goal
        if heuristic_func(current_best, goal_board, heuristic) == 0:
            # We've found the goal state, but GA doesn't maintain the path
            # We'll construct a simple representation of the solution
            end_time = time.time()
            solution_state = PuzzleState(current_best)
            
            return {
                "path": [(current_best, "")],  # GA doesn't provide a step-by-step path
                "generations": generation + 1,
                "population_size": pop_size,
                "time": end_time - start_time,
                "final_state": current_best,
                "final_value": heuristic_func(current_best, goal_board, heuristic)
            }
        
        # Early stopping if no improvement for a while
        if generations_without_improvement > 20:
            break
        
        # Selection
        num_parents = pop_size // 2
        parents = selection(population, fitnesses, num_parents)
        
        # Create new population
        new_population = [current_best]  # Elitism - keep the best individual
        
        while len(new_population) < pop_size:
            # Select two random parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutation(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    # If we reach here, we didn't find the goal
    end_time = time.time()
    
    # Return the best state found
    return {
        "path": None,
        "generations": max_generations,
        "population_size": pop_size,
        "time": end_time - start_time,
        "final_state": best_solution,
        "final_value": heuristic_func(best_solution, goal_board, heuristic)
    }

def solve_puzzle_local_search(initial_board, goal_board=None, algorithm="simulated_annealing", heuristic="manhattan", **kwargs):
    """
    Solve the 8-puzzle using local search algorithms
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
    
    if algorithm.lower() == "simple_hill_climbing":
        result = simple_hill_climbing(initial_board, goal_board, heuristic=heuristic, **kwargs)
    elif algorithm.lower() == "steepest_hill_climbing":
        result = steepest_ascent_hill_climbing(initial_board, goal_board, heuristic=heuristic, **kwargs)
    elif algorithm.lower() == "stochastic_hill_climbing":
        result = stochastic_hill_climbing(initial_board, goal_board, heuristic=heuristic, **kwargs)
    elif algorithm.lower() == "beam_search":
        result = beam_search(initial_board, goal_board, heuristic=heuristic, **kwargs)
    elif algorithm.lower() == "simulated_annealing":
        result = simulated_annealing(initial_board, goal_board, heuristic=heuristic, **kwargs)
    elif algorithm.lower() == "genetic_algorithm":
        result = genetic_algorithm(initial_board, goal_board, heuristic=heuristic, **kwargs)
    else:
        return {"error": "Unknown algorithm"}
    
    if result.get("path"):
        print(f"Solution found!")
        if "iterations" in result:
            print(f"Iterations: {result['iterations']}")
        if "generations" in result:
            print(f"Generations: {result['generations']}")
        if "nodes_expanded" in result:
            print(f"Nodes expanded: {result['nodes_expanded']}")
        print(f"Time taken: {result['time']:.4f} seconds")
        return result
    else:
        print("No solution found")
        print(f"Final heuristic value: {result.get('final_value', 'N/A')}")
        print(f"Time taken: {result['time']:.4f} seconds")
        return result

def compare_local_search_algorithms(initial_board, goal_board=None, heuristic="manhattan"):
    """
    Compare all local search algorithms
    """
    results = {}
    
    algorithms = [
        "simple_hill_climbing",
        "steepest_hill_climbing",
        "stochastic_hill_climbing",
        "beam_search",
        "simulated_annealing",
        "genetic_algorithm"
    ]
    
    for algorithm in algorithms:
        print(f"\nRunning {algorithm.upper()}")
        results[algorithm] = solve_puzzle_local_search(initial_board, goal_board, algorithm, heuristic)
    
    print("\nComparison Results:")
    print("-" * 90)
    print(f"{'Algorithm':<25} | {'Time (s)':<10} | {'Found':<5} | {'Final Value':<12} | {'Steps/Iterations':<15}")
    print("-" * 90)
    
    for algo, result in results.items():
        found = "Yes" if result.get("path") else "No"
        final_value = result.get("final_value", "N/A")
        
        steps = "N/A"
        if result.get("path"):
            steps = len(result["path"]) - 1
        elif "iterations" in result:
            steps = result["iterations"]
        elif "generations" in result:
            steps = result["generations"]
            
        print(f"{algo.upper():<25} | {result['time']:<10.4f} | {found:<5} | {final_value:<12} | {steps:<15}")
    
    print("-" * 90)

# Example usage:
if __name__ == "__main__":
    # Example initial state (solvable)
    initial_board = [
        [1, 2, 3],
        [4, 0, 6],
        [7, 5, 8]
    ]
    
    # For testing a single algorithm
    # solve_puzzle_local_search(initial_board, algorithm="simulated_annealing", heuristic="manhattan")
    
    # For comparing all algorithms
    compare_local_search_algorithms(initial_board, heuristic="manhattan")
