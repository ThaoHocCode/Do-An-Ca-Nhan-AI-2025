import json
import time
import os
from Uniformed_Search import PuzzleState, solve_puzzle
from Informed_Search import solve_puzzle_informed
from Local_Search import solve_puzzle_local_search
from CSPs import solve_puzzle_csp
from Reinforcement_Search import solve_puzzle_reinforcement
from Complex_Environments import solve_puzzle_complex

# Create results directory if it doesn't exist
RESULTS_DIR = 'results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def get_filename(algo_name):
    """Convert algorithm name to valid filename"""
    if algo_name == 'A*':
        return 'results_a_star'
    elif algo_name == 'IDA*':
        return 'results_ida_star'
    else:
        return f'results_{algo_name.lower().replace(" ", "_")}'

def get_algorithm_name(algo_name):
    """Convert display name to internal algorithm name"""
    if algo_name == 'A*':
        return 'astar'
    elif algo_name == 'IDA*':
        return 'idastar'
    else:
        return algo_name.lower()

def run_experiment(algorithm_func, initial_state, goal_state, algorithm_name, num_trials=20):
    """Run experiment for a specific algorithm"""
    total_moves = 0
    total_nodes = 0
    total_frontier = 0
    total_time = 0
    successful_trials = 0
    
    # Get internal algorithm name
    internal_name = get_algorithm_name(algorithm_name)
    print(f"Running {algorithm_name} (internal name: {internal_name})")
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials} for {algorithm_name}")
        
        # Measure time
        start_time = time.time()
        
        try:
            # Run algorithm
            if algorithm_name in ['BFS', 'DFS', 'UCS', 'IDS']:
                result = solve_puzzle(initial_state, goal_state, internal_name)
            elif algorithm_name in ['Greedy', 'A*', 'IDA*']:
                heuristic = 'manhattan'
                # Convert display name to internal name for informed search
                if algorithm_name == 'A*':
                    internal_name = 'astar'
                elif algorithm_name == 'IDA*':
                    internal_name = 'idastar'
                else:
                    internal_name = 'greedy'
                result = solve_puzzle_informed(initial_state, goal_state, internal_name, heuristic)
            elif algorithm_name in ['Simple Hill Climbing', 'Steepest Hill Climbing', 'Stochastic Hill Climbing', 
                                  'Beam Search', 'Simulated Annealing', 'Genetic Algorithm']:
                heuristic = 'manhattan'
                result = solve_puzzle_local_search(initial_state, goal_state, internal_name, heuristic)
            elif algorithm_name in ['Backtracking', 'Min-Conflicts (Labeling)']:
                heuristic = 'manhattan'
                result = solve_puzzle_csp(initial_state, goal_state, internal_name, heuristic)
            elif algorithm_name in ['Q-Learning', 'DQN', 'SARSA', 'Policy Gradient']:
                result = solve_puzzle_reinforcement(initial_state, goal_state, internal_name)
            else:  # Complex environments
                result = solve_puzzle_complex(initial_state, goal_state, internal_name)
            
            end_time = time.time()
            
            # Collect metrics if result is valid
            if result and isinstance(result, dict) and 'path' in result:
                total_moves += len(result['path']) - 1
                total_nodes += result.get('nodes_expanded', 0)
                total_frontier = max(total_frontier, result.get('max_frontier_size', 0))
                total_time += (end_time - start_time)
                successful_trials += 1
                print(f"Successful trial for {algorithm_name}: {len(result['path'])-1} moves")
            else:
                print(f"Invalid result for {algorithm_name}: {result}")
        except Exception as e:
            print(f"Error in trial {trial + 1} for {algorithm_name}: {str(e)}")
            continue
    
    # Calculate averages only if we have successful trials
    if successful_trials > 0:
        return {
            'moves': total_moves / successful_trials,
            'nodes_expanded': total_nodes / successful_trials,
            'max_frontier_size': total_frontier,
            'time': total_time / successful_trials
        }
    else:
        return {
            'moves': None,
            'nodes_expanded': None,
            'max_frontier_size': None,
            'time': None
        }

def main():
    # Define initial and goal states
    initial_state = [
        [0, 1, 8],
        [4, 3, 2],
        [7, 6, 5]
    ]
    goal_state = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    
    # Define algorithms to test
    algorithms = {
        'BFS': solve_puzzle,
        'DFS': solve_puzzle,
        'UCS': solve_puzzle,
        'IDS': solve_puzzle,
        'Greedy': solve_puzzle_informed,
        'A*': solve_puzzle_informed,
        'IDA*': solve_puzzle_informed,
        'Simple Hill Climbing': solve_puzzle_local_search,
        'Steepest Hill Climbing': solve_puzzle_local_search,
        'Stochastic Hill Climbing': solve_puzzle_local_search,
        'Beam Search': solve_puzzle_local_search,
        'Simulated Annealing': solve_puzzle_local_search,
        'Genetic Algorithm': solve_puzzle_local_search,
        'Backtracking': solve_puzzle_csp,
        'Forward Checking': solve_puzzle_csp,
        'Min-Conflicts': solve_puzzle_csp,
        'Min-Conflicts (Labeling)': solve_puzzle_csp,
        'Q-Learning': solve_puzzle_reinforcement,
        'DQN': solve_puzzle_reinforcement,
        'SARSA': solve_puzzle_reinforcement,
        'Policy Gradient': solve_puzzle_reinforcement,
        'AND-OR Tree': solve_puzzle_complex,
        'Partially Observable': solve_puzzle_complex,
        'Belief State': solve_puzzle_complex
    }
    
    # Run experiments for each algorithm
    for algo_name, algo_func in algorithms.items():
        print(f"\nTesting {algo_name}...")
        try:
            results = run_experiment(algo_func, initial_state, goal_state, algo_name)
            
            # Save results to JSON file in results directory
            filename = os.path.join(RESULTS_DIR, f'{get_filename(algo_name)}.json')
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error testing {algo_name}: {str(e)}")

if __name__ == "__main__":
    main() 