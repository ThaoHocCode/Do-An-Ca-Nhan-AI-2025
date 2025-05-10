import sys
import tkinter as tk
from Uniformed_Search import compare_algorithms, solve_puzzle
from Informed_Search import compare_informed_algorithms, solve_puzzle_informed
from Local_Search import compare_local_search_algorithms, solve_puzzle_local_search
from CSPs import compare_csp_algorithms, solve_puzzle_csp
from Reinforcement_Search import compare_reinforcement_algorithms, solve_puzzle_reinforcement
from Complex_Environments import compare_complex_algorithms, solve_puzzle_complex
from Puzzle_GUI import PuzzleGUI

def display_help():
    print("8-Puzzle Solver")
    print("===============")
    print("Usage:")
    print("  python main.py [mode] [algorithm_group] [algorithm] [heuristic]")
    print()
    print("Modes:")
    print("  --gui       : Launch the graphical user interface (default)")
    print("  --console   : Run in console mode")
    print()
    print("Algorithm Groups:")
    print("  --uninformed    : Use uninformed search algorithms (default)")
    print("  --informed      : Use informed search algorithms")
    print("  --local         : Use local search algorithms")
    print("  --csp           : Use constraint satisfaction problem algorithms")
    print("  --reinforcement : Use reinforcement learning algorithms")
    print("  --complex       : Use algorithms for complex environments")
    print()
    print("Uninformed Search Algorithms:")
    print("  --bfs       : Breadth-First Search")
    print("  --dfs       : Depth-First Search")
    print("  --ucs       : Uniform Cost Search")
    print("  --ids       : Iterative Deepening Search")
    print("  --all       : Compare all algorithms (default)")
    print()
    print("Informed Search Algorithms:")
    print("  --greedy    : Greedy Best-First Search")
    print("  --astar     : A* Search (default)")
    print("  --idastar   : IDA* Search")
    print("  --all       : Compare all algorithms")
    print()
    print("Local Search Algorithms:")
    print("  --simple_hill_climbing    : Simple Hill Climbing")
    print("  --steepest_hill_climbing  : Steepest Ascent Hill Climbing")
    print("  --stochastic_hill_climbing: Stochastic Hill Climbing")
    print("  --beam_search             : Beam Search")
    print("  --simulated_annealing     : Simulated Annealing (default)")
    print("  --genetic_algorithm       : Genetic Algorithm")
    print("  --all                     : Compare all algorithms")
    print()
    print("CSP Algorithms:")
    print("  --backtracking       : Backtracking (default)")
    print("  --forward_checking   : Backtracking with Forward Checking")
    print("  --min_conflicts      : Min-Conflicts")
    print("  --min_conflicts_label: Min-Conflicts with Labeling Approach")
    print("  --all                : Compare all algorithms")
    print()
    print("Reinforcement Learning Algorithms:")
    print("  --q_learning      : Q-Learning (default)")
    print("  --dqn             : Deep Q-Network")
    print("  --sarsa           : SARSA")
    print("  --policy_gradient : Policy Gradient")
    print("  --all             : Compare all algorithms")
    print()
    print("Complex Environment Algorithms:")
    print("  --and_or_tree           : AND-OR Tree Search (for non-deterministic environments)")
    print("  --partially_observable  : Search with Partially Observable States")
    print("  --belief_state          : Belief State Search (default)")
    print("  --all                   : Compare all algorithms")
    print()
    print("Heuristics (for informed, local search, and reinforcement learning):")
    print("  --manhattan : Manhattan distance heuristic (default)")
    print("  --misplaced : Misplaced tiles heuristic")
    print("  --linear    : Linear conflict heuristic (only for informed search)")
    print()
    print("Examples:")
    print("  python main.py                                 : Launch GUI")
    print("  python main.py --console --uninformed --bfs    : Run BFS algorithm in console")
    print("  python main.py --console --complex --belief_state : Run Belief State Search")

def run_gui():
    root = tk.Tk()
    app = PuzzleGUI(root)
    root.mainloop()

def run_console(algorithm_group="uninformed", algorithm="all", heuristic="manhattan"):
    # Example initial state (solvable)
    initial_board = [
        [1, 2, 3],
        [4, 0, 6],
        [7, 5, 8]
    ]
    
    goal_board = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    
    if algorithm_group == "uninformed":
        if algorithm == "all":
            compare_algorithms(initial_board, goal_board)
        else:
            solve_puzzle(initial_board, goal_board, algorithm)
    elif algorithm_group == "informed":
        if algorithm == "all":
            compare_informed_algorithms(initial_board, goal_board, heuristic)
        else:
            solve_puzzle_informed(initial_board, goal_board, algorithm, heuristic)
    elif algorithm_group == "local":
        if algorithm == "all":
            compare_local_search_algorithms(initial_board, goal_board, heuristic)
        else:
            solve_puzzle_local_search(initial_board, goal_board, algorithm, heuristic)
    elif algorithm_group == "csp":
        if algorithm == "all":
            compare_csp_algorithms(initial_board, goal_board, heuristic)
        else:
            solve_puzzle_csp(initial_board, goal_board, algorithm, heuristic)
    elif algorithm_group == "reinforcement":
        if algorithm == "all":
            compare_reinforcement_algorithms(initial_board, goal_board, heuristic)
        else:
            solve_puzzle_reinforcement(initial_board, goal_board, algorithm, heuristic)
    elif algorithm_group == "complex":
        if algorithm == "all":
            compare_complex_algorithms(initial_board, goal_board)
        else:
            solve_puzzle_complex(initial_board, goal_board, algorithm)

if __name__ == "__main__":
    # Default settings
    mode = "gui"
    algorithm_group = "uninformed"
    algorithm = "all"
    heuristic = "manhattan"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if "--help" in sys.argv or "-h" in sys.argv:
            display_help()
            sys.exit(0)
            
        # Mode
        if "--gui" in sys.argv:
            mode = "gui"
        elif "--console" in sys.argv:
            mode = "console"
            
        # Algorithm group
        if "--uninformed" in sys.argv:
            algorithm_group = "uninformed"
        elif "--informed" in sys.argv:
            algorithm_group = "informed"
        elif "--local" in sys.argv:
            algorithm_group = "local"
        elif "--csp" in sys.argv:
            algorithm_group = "csp"
        elif "--reinforcement" in sys.argv:
            algorithm_group = "reinforcement"
        elif "--complex" in sys.argv:
            algorithm_group = "complex"
            
        # Uninformed algorithms
        if "--bfs" in sys.argv:
            algorithm = "bfs"
        elif "--dfs" in sys.argv:
            algorithm = "dfs"
        elif "--ucs" in sys.argv:
            algorithm = "ucs"
        elif "--ids" in sys.argv:
            algorithm = "ids"
        
        # Informed algorithms
        elif "--greedy" in sys.argv:
            algorithm = "greedy"
        elif "--astar" in sys.argv:
            algorithm = "astar"
        elif "--idastar" in sys.argv:
            algorithm = "idastar"
            
        # Local search algorithms
        elif "--simple_hill_climbing" in sys.argv:
            algorithm = "simple_hill_climbing"
        elif "--steepest_hill_climbing" in sys.argv:
            algorithm = "steepest_hill_climbing"
        elif "--stochastic_hill_climbing" in sys.argv:
            algorithm = "stochastic_hill_climbing"
        elif "--beam_search" in sys.argv:
            algorithm = "beam_search"
        elif "--simulated_annealing" in sys.argv:
            algorithm = "simulated_annealing"
        elif "--genetic_algorithm" in sys.argv:
            algorithm = "genetic_algorithm"
            
        # CSP algorithms
        elif "--backtracking" in sys.argv:
            algorithm = "backtracking"
        elif "--forward_checking" in sys.argv:
            algorithm = "forward_checking"
        elif "--min_conflicts" in sys.argv:
            algorithm = "min_conflicts"
        elif "--min_conflicts_label" in sys.argv:
            algorithm = "min_conflicts_labeling"
            
        # Reinforcement Learning algorithms
        elif "--q_learning" in sys.argv:
            algorithm = "q_learning"
        elif "--dqn" in sys.argv:
            algorithm = "dqn"
        elif "--sarsa" in sys.argv:
            algorithm = "sarsa"
        elif "--policy_gradient" in sys.argv:
            algorithm = "policy_gradient"
            
        # Complex Environment algorithms
        elif "--and_or_tree" in sys.argv:
            algorithm = "and_or_tree"
        elif "--partially_observable" in sys.argv:
            algorithm = "partially_observable"
        elif "--belief_state" in sys.argv:
            algorithm = "belief_state"
        
        # All algorithms comparison
        elif "--all" in sys.argv:
            algorithm = "all"
            
        # Heuristics
        if "--manhattan" in sys.argv:
            heuristic = "manhattan"
        elif "--misplaced" in sys.argv:
            heuristic = "misplaced"
        elif "--linear" in sys.argv:
            heuristic = "linear"
    
    # Run the selected mode
    if mode == "gui":
        run_gui()
    else:
        run_console(algorithm_group, algorithm, heuristic) 