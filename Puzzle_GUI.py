import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading
import random
from Uniformed_Search import PuzzleState, solve_puzzle, print_board
from Informed_Search import solve_puzzle_informed, misplaced_tiles ,manhattan_distance
from Local_Search import solve_puzzle_local_search, misplaced_tiles, manhattan_distance
from CSPs import solve_puzzle_csp, misplaced_tiles
from Reinforcement_Search import solve_puzzle_reinforcement
from Complex_Environments import solve_puzzle_complex

class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.geometry("800x800")
        self.root.resizable(True, True)
        
        # Tạo style cho các nút
        style = ttk.Style()
        style.configure('Tile.TButton', font=('Arial', 14, 'bold'))
        style.configure('EmptyTile.TButton', font=('Arial', 14, 'bold'), background='black', foreground='white')
        style.configure('BeliefTile.TButton', font=('Arial', 10, 'bold'), background='#e6ffff')
        style.map('BeliefTile.TButton', background=[('active', '#b3ffff')])
        style.configure('DuplicateTile.TButton', font=('Arial', 10), background='#f0f0f0')
        style.map('DuplicateTile.TButton', background=[('active', '#e0e0e0')])
        style.configure('Move.TButton', background='light green', font=('Arial', 14, 'bold'))
        style.configure('Error.TButton', background='red', foreground='white', font=('Arial', 14, 'bold'))
        style.configure('TRadiobutton', font=('Arial', 9))
        style.configure('Belief.TLabelframe', background='#f0f0ff')
        style.configure('Belief.TLabelframe.Label', font=('Arial', 11, 'bold'), background='#f0f0ff', foreground='#000088')
        style.configure('Duplicate.TLabelframe', background='#f5f5f5')
        style.configure('Duplicate.TLabelframe.Label', font=('Arial', 11), background='#f5f5f5', foreground='#666666')
        
        # Highlight các nút nhóm thuật toán
        style.map('Uninformed.TRadiobutton', background=[('selected', '#e6f2ff')], foreground=[('selected', '#0066cc')])
        style.map('Informed.TRadiobutton', background=[('selected', '#e6ffe6')], foreground=[('selected', '#006600')])
        style.map('Local.TRadiobutton', background=[('selected', '#fff2e6')], foreground=[('selected', '#cc6600')])
        style.map('CSP.TRadiobutton', background=[('selected', '#ffe6e6')], foreground=[('selected', '#cc0000')])
        style.map('RL.TRadiobutton', background=[('selected', '#f2e6ff')], foreground=[('selected', '#6600cc')])
        style.map('Complex.TRadiobutton', background=[('selected', '#e6ffff')], foreground=[('selected', '#006666')])
        
        # Default goal state
        self.goal_board = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]
        
        # Initial random board (will be set to a solvable state)
        self.board = self.generate_solvable_board()
        
        # Selected algorithm 
        self.algorithm_group = tk.StringVar(value="uninformed")
        
        # Uninformed algorithms
        self.uninformed_algorithm = tk.StringVar(value="bfs")
        
        # Informed algorithms
        self.informed_algorithm = tk.StringVar(value="astar")
        self.heuristic = tk.StringVar(value="manhattan")
        
        # Local search algorithms
        self.local_algorithm = tk.StringVar(value="simulated_annealing")
        self.local_heuristic = tk.StringVar(value="manhattan")
        
        # CSP algorithms
        self.csp_algorithm = tk.StringVar(value="backtracking")
        self.csp_heuristic = tk.StringVar(value="manhattan")
        
        # Reinforcement learning algorithms
        self.reinforcement_algorithm = tk.StringVar(value="q_learning")
        self.reinforcement_heuristic = tk.StringVar(value="manhattan")
        
        # Complex environments algorithms
        self.complex_algorithm = tk.StringVar(value="belief_state")
        self.complex_algorithm.trace_add("write", self.on_complex_algorithm_change)
        
        # Animation control
        self.animation_speed = tk.IntVar(value=500)  # milliseconds between steps
        self.animation_running = False
        self.animation_job = None
        
        # Results
        self.result = None
        self.current_step = 0
        
        # For belief state visualization
        self.belief_states = []
        self.belief_boards = []
        
        # For step visualization
        self.steps_text = ""
        
        # Style configurations for the app
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 16, 'bold'), padding=10)
        self.style.configure('BeliefTile.TButton', font=('Arial', 12, 'bold'), padding=5, background='#d0e0f0')
        self.style.configure('DuplicateTile.TButton', font=('Arial', 12, 'bold'), padding=5, background='#f0d0d0')
        self.style.configure('Move.TButton', font=('Arial', 12, 'bold'), padding=5, background='#c0f0c0')
        self.style.configure('ConvergedState.TButton', font=('Arial', 12, 'bold'), padding=5, background='#f0e0c0')
        self.style.configure('SourceState.TButton', font=('Arial', 12, 'bold'), padding=5, background='#d0f0e0')
        self.style.configure('ActiveState.TButton', font=('Arial', 12, 'bold'), padding=5, background='#e0f0d0')
        
        # Add styles for the belief state frames
        self.style.configure('Belief.TLabelframe', font=('Arial', 12, 'bold'))
        self.style.configure('Belief.TLabelframe.Label', font=('Arial', 12, 'bold'), foreground='#000000')
        self.style.configure('BeliefActive.TLabelframe', font=('Arial', 12, 'bold'))
        self.style.configure('BeliefActive.TLabelframe.Label', font=('Arial', 12, 'bold'), foreground='#0000ff')
        self.style.configure('BeliefSource.TLabelframe', font=('Arial', 12, 'bold'))
        self.style.configure('BeliefSource.TLabelframe.Label', font=('Arial', 12, 'bold'), foreground='#006600')
        self.style.configure('BeliefConverged.TLabelframe', font=('Arial', 12, 'bold'))
        self.style.configure('BeliefConverged.TLabelframe.Label', font=('Arial', 12, 'bold'), foreground='#cc0000')
        
        # Thêm vào __init__ sau các style khác
        self.style.configure('BacktrackTile.TButton', font=('Arial', 14, 'bold'), background='#ffcccc')  # Đỏ nhạt
        self.style.configure('TryingTile.TButton', font=('Arial', 14, 'bold'), background='#cce6ff')    # Xanh dương nhạt
        self.style.configure('GoalTile.TButton', font=('Arial', 14, 'bold'), background='#ccffcc')      # Xanh lá nhạt
        
        # Create the UI
        self.create_ui()
        
        # Update the board display
        self.update_board_display()
    
    def create_ui(self):
        # Main frame với layout tối ưu hơn
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Phân chia thành 2 panel chính: bên trái cho bảng và kết quả, bên phải cho điều khiển
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Title
        title_label = ttk.Label(left_panel, text="8-Puzzle Solver", font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Board frame
        self.board_frame = ttk.Frame(left_panel)
        self.board_frame.pack(pady=10)
        
        # Create the board UI
        self.board_buttons = []
        for i in range(3):
            row_buttons = []
            for j in range(3):
                button = ttk.Button(self.board_frame, text="", width=5, style='Tile.TButton')
                button.grid(row=i, column=j, padx=5, pady=5, ipadx=10, ipady=10)
                row_buttons.append(button)
            self.board_buttons.append(row_buttons)
        
        # Belief state visualization frames (initially hidden)
        # Create the frame in center_panel between algorithm selection and buttons
        self.belief_frames = ttk.Frame(main_frame)
        
        # Create a container frame with a title and border for belief states
        belief_title_frame = ttk.LabelFrame(self.belief_frames, text="4 SIMULTANEOUS BELIEF STATES", 
                                           padding=10, style='Belief.TLabelframe')
        belief_title_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Explanation text
        explanation_frame = ttk.Frame(belief_title_frame)
        explanation_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(explanation_frame, 
                 text="Belief state search finds a solution that works for all possible states",
                 font=("Arial", 10, "italic"), foreground='#000088').pack(side=tk.LEFT)
        
        # Create a grid layout for the 4 belief states
        belief_states_grid = ttk.Frame(belief_title_frame)
        belief_states_grid.pack(fill=tk.BOTH, expand=True)
        
        # Create 2x2 grid of smaller boards for belief states
        self.belief_board_buttons = []
        for idx in range(4):
            belief_board_frame = ttk.LabelFrame(belief_states_grid, text=f"Belief State {idx+1}", padding=5, style='Belief.TLabelframe')
            belief_board_frame.grid(row=idx//2, column=idx%2, padx=10, pady=5, sticky=tk.NSEW)
            
            board_frame = ttk.Frame(belief_board_frame)
            board_frame.pack(padx=5, pady=5)
            
            board = []
            for i in range(3):
                row_buttons = []
                for j in range(3):
                    # Make buttons larger and more visually appealing
                    button = ttk.Button(board_frame, text="", width=3, style='BeliefTile.TButton')
                    button.grid(row=i, column=j, padx=3, pady=3, ipadx=5, ipady=5)
                    row_buttons.append(button)
                board.append(row_buttons)
            self.belief_board_buttons.append(board)
        
        # Thêm khu vực hiển thị trạng thái các bước đi
        self.step_display_frame = ttk.LabelFrame(left_panel, text="Step Visualization", padding=10)
        self.step_display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tạo container cho text và scrollbar
        step_container = ttk.Frame(self.step_display_frame)
        step_container.pack(fill=tk.BOTH, expand=True)
        
        # Tạo Text widget để hiển thị chi tiết các bước
        self.step_display = tk.Text(step_container, height=8, width=50, wrap=tk.WORD, state="disabled")
        self.step_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Thêm thanh cuộn cho khu vực hiển thị bước
        step_scrollbar = ttk.Scrollbar(step_container, orient="vertical", command=self.step_display.yview)
        self.step_display.configure(yscrollcommand=step_scrollbar.set)
        step_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Results frame - đặt ở bên dưới bảng, trong left_panel
        self.results_frame = ttk.LabelFrame(left_panel, text="Results", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tạo container cho text và scrollbar
        results_container = ttk.Frame(self.results_frame)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        # Tăng kích thước của khu vực hiển thị kết quả
        self.results_text = tk.Text(results_container, height=10, width=50, wrap=tk.WORD, state="disabled")
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Thêm thanh cuộn cho khu vực kết quả
        results_scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Algorithm selection - đặt trong right_panel
        algo_frame = ttk.LabelFrame(right_panel, text="Algorithm Selection", padding=10)
        algo_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Algorithm group selection label
        ttk.Label(algo_frame, text="Chọn nhóm thuật toán:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Algorithm group selection (Tabs)
        group_frame = ttk.Frame(algo_frame)
        group_frame.pack(fill=tk.X, pady=5)
        
        # Tạo grid layout để hiển thị thuật toán rõ ràng hơn
        uninformed_btn = ttk.Radiobutton(group_frame, text="Uninformed Search", value="uninformed", 
                                      variable=self.algorithm_group, command=self.update_algorithm_options,
                                      style='Uninformed.TRadiobutton')
        uninformed_btn.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        
        informed_btn = ttk.Radiobutton(group_frame, text="Informed Search", value="informed", 
                                     variable=self.algorithm_group, command=self.update_algorithm_options,
                                     style='Informed.TRadiobutton')
        informed_btn.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        local_btn = ttk.Radiobutton(group_frame, text="Local Search", value="local", 
                                  variable=self.algorithm_group, command=self.update_algorithm_options,
                                  style='Local.TRadiobutton')
        local_btn.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        
        csp_btn = ttk.Radiobutton(group_frame, text="CSP", value="csp", 
                               variable=self.algorithm_group, command=self.update_algorithm_options,
                               style='CSP.TRadiobutton')
        csp_btn.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        reinforcement_btn = ttk.Radiobutton(group_frame, text="Reinforcement", value="reinforcement", 
                                variable=self.algorithm_group, command=self.update_algorithm_options,
                                style='RL.TRadiobutton')
        reinforcement_btn.grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        
        complex_btn = ttk.Radiobutton(group_frame, text="Complex Env", value="complex", 
                                variable=self.algorithm_group, command=self.update_algorithm_options,
                                style='Complex.TRadiobutton')
        complex_btn.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Create frames for each algorithm group
        self.uninformed_frame = ttk.Frame(algo_frame)
        self.uninformed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.uninformed_frame, text="Uninformed Search Algorithms:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(self.uninformed_frame, text="Breadth-First Search (BFS)", 
                       value="bfs", variable=self.uninformed_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(self.uninformed_frame, text="Depth-First Search (DFS)", 
                       value="dfs", variable=self.uninformed_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(self.uninformed_frame, text="Uniform Cost Search (UCS)", 
                       value="ucs", variable=self.uninformed_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(self.uninformed_frame, text="Iterative Deepening Search (IDS)", 
                       value="ids", variable=self.uninformed_algorithm).pack(anchor=tk.W, pady=2)
        
        # Informed search frame
        self.informed_frame = ttk.Frame(algo_frame)
        
        # Informed Algorithms
        algo_options_frame = ttk.Frame(self.informed_frame)
        algo_options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        ttk.Label(algo_options_frame, text="Informed Search Algorithms:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(algo_options_frame, text="Greedy Best-First Search", 
                       value="greedy", variable=self.informed_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(algo_options_frame, text="A* Search", 
                       value="astar", variable=self.informed_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(algo_options_frame, text="IDA* Search", 
                       value="idastar", variable=self.informed_algorithm).pack(anchor=tk.W, pady=2)
        
        # Informed Heuristics
        heuristic_frame = ttk.Frame(self.informed_frame)
        heuristic_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(heuristic_frame, text="Heuristic:").pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(heuristic_frame, text="Manhattan Distance", 
                       value="manhattan", variable=self.heuristic).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(heuristic_frame, text="Misplaced Tiles", 
                       value="misplaced", variable=self.heuristic).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(heuristic_frame, text="Linear Conflict", 
                       value="linear", variable=self.heuristic).pack(anchor=tk.W, pady=2)
        
        # Local search frame
        self.local_frame = ttk.Frame(algo_frame)
        
        # Local Search Algorithms
        local_algo_frame = ttk.Frame(self.local_frame)
        local_algo_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        ttk.Label(local_algo_frame, text="Local Search Algorithms:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(local_algo_frame, text="Simple Hill Climbing", 
                       value="simple_hill_climbing", variable=self.local_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(local_algo_frame, text="Steepest Hill Climbing", 
                       value="steepest_hill_climbing", variable=self.local_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(local_algo_frame, text="Stochastic Hill Climbing", 
                       value="stochastic_hill_climbing", variable=self.local_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(local_algo_frame, text="Beam Search", 
                       value="beam_search", variable=self.local_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(local_algo_frame, text="Simulated Annealing", 
                       value="simulated_annealing", variable=self.local_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(local_algo_frame, text="Genetic Algorithm", 
                       value="genetic_algorithm", variable=self.local_algorithm).pack(anchor=tk.W, pady=2)
        
        # Local Search Heuristics
        local_heuristic_frame = ttk.Frame(self.local_frame)
        local_heuristic_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(local_heuristic_frame, text="Heuristic:").pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(local_heuristic_frame, text="Manhattan Distance", 
                       value="manhattan", variable=self.local_heuristic).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(local_heuristic_frame, text="Misplaced Tiles", 
                       value="misplaced", variable=self.local_heuristic).pack(anchor=tk.W, pady=2)
        
        # CSP frame
        self.csp_frame = ttk.Frame(algo_frame)
        
        # CSP Algorithms
        csp_algorithms_frame = ttk.Frame(self.csp_frame)
        csp_algorithms_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        self.csp_alg_frame = csp_algorithms_frame
        
        ttk.Label(csp_algorithms_frame, text="CSP Algorithms:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(csp_algorithms_frame, text="Backtracking", value="backtracking", 
                     variable=self.csp_algorithm, style='CSP.TRadiobutton').pack(anchor=tk.W)
        ttk.Radiobutton(csp_algorithms_frame, text="Forward Checking", value="forward_checking", 
                     variable=self.csp_algorithm, style='CSP.TRadiobutton').pack(anchor=tk.W)
        ttk.Radiobutton(csp_algorithms_frame, text="Min-Conflicts", value="min_conflicts", 
                     variable=self.csp_algorithm, style='CSP.TRadiobutton').pack(anchor=tk.W)
        ttk.Radiobutton(csp_algorithms_frame, text="Min-Conflicts (Labeling)", value="min_conflicts_labeling", 
                     variable=self.csp_algorithm, style='CSP.TRadiobutton').pack(anchor=tk.W)
        
        # CSP Heuristics
        csp_heuristic_frame = ttk.Frame(self.csp_frame)
        csp_heuristic_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(csp_heuristic_frame, text="Heuristic:").pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(csp_heuristic_frame, text="Manhattan Distance", 
                       value="manhattan", variable=self.csp_heuristic).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(csp_heuristic_frame, text="Misplaced Tiles", 
                       value="misplaced", variable=self.csp_heuristic).pack(anchor=tk.W, pady=2)
        
        # Reinforcement learning frame
        self.reinforcement_frame = ttk.Frame(algo_frame)
        
        # Reinforcement Learning Algorithms
        rl_algo_frame = ttk.Frame(self.reinforcement_frame)
        rl_algo_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        ttk.Label(rl_algo_frame, text="Reinforcement Learning Algorithms:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(rl_algo_frame, text="Q-Learning", 
                       value="q_learning", variable=self.reinforcement_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(rl_algo_frame, text="Deep Q-Network (DQN)", 
                       value="dqn", variable=self.reinforcement_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(rl_algo_frame, text="SARSA", 
                       value="sarsa", variable=self.reinforcement_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(rl_algo_frame, text="Policy Gradient", 
                       value="policy_gradient", variable=self.reinforcement_algorithm).pack(anchor=tk.W, pady=2)
        
        # Reinforcement Learning Heuristics
        rl_heuristic_frame = ttk.Frame(self.reinforcement_frame)
        rl_heuristic_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(rl_heuristic_frame, text="Heuristic:").pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(rl_heuristic_frame, text="Manhattan Distance", 
                       value="manhattan", variable=self.reinforcement_heuristic).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(rl_heuristic_frame, text="Misplaced Tiles", 
                       value="misplaced", variable=self.reinforcement_heuristic).pack(anchor=tk.W, pady=2)
        
        # Complex environments frame
        self.complex_frame = ttk.Frame(algo_frame)
        
        # Complex Environment Algorithms
        complex_algo_frame = ttk.Frame(self.complex_frame)
        complex_algo_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(complex_algo_frame, text="Complex Environment Algorithms:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(complex_algo_frame, text="AND-OR Tree Search (Non-deterministic)", 
                       value="and_or_tree", variable=self.complex_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(complex_algo_frame, text="Partially Observable Search", 
                       value="partially_observable", variable=self.complex_algorithm).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(complex_algo_frame, text="Belief State Search (4 simultaneous states)", 
                       value="belief_state", variable=self.complex_algorithm).pack(anchor=tk.W, pady=2)
        
        # Initialize the visibility of algorithm frames
        self.update_algorithm_options()
        
        # Buttons frame - đặt ở dưới algo_frame trong right_panel
        buttons_frame = ttk.Frame(right_panel)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Shuffle", command=self.shuffle_board).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Solve", command=self.solve).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Run", command=self.run_animation).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Stop", command=self.stop_animation).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Show All Steps", command=self.show_all_steps).pack(side=tk.LEFT, padx=5)
        
        # Animation speed slider
        speed_frame = ttk.Frame(right_panel)
        speed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(speed_frame, text="Animation Speed:").pack(side=tk.LEFT, padx=5)
        speed_slider = ttk.Scale(speed_frame, from_=100, to=2000, orient=tk.HORIZONTAL, 
                                variable=self.animation_speed)
        speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Solution navigation - đặt ở dưới speed_frame trong right_panel
        self.nav_frame = ttk.Frame(right_panel)
        self.nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(self.nav_frame, text="<<", command=self.first_step).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_frame, text="<", command=self.prev_step).pack(side=tk.LEFT, padx=5)
        self.step_label = ttk.Label(self.nav_frame, text="Step: 0/0")
        self.step_label.pack(side=tk.LEFT, padx=20)
        ttk.Button(self.nav_frame, text=">", command=self.next_step).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_frame, text=">>", command=self.last_step).pack(side=tk.LEFT, padx=5)
        self.nav_frame.pack_forget()  # Hide until we have a solution
    
    def update_algorithm_options(self):
        """Update the visibility of algorithm options based on selected group"""
        if self.algorithm_group.get() == "uninformed":
            self.uninformed_frame.pack(fill=tk.X, pady=5)
            self.informed_frame.pack_forget()
            self.local_frame.pack_forget()
            self.csp_frame.pack_forget()
            self.reinforcement_frame.pack_forget()
            self.complex_frame.pack_forget()
            self.belief_frames.pack_forget()  # Hide belief states visualization
        elif self.algorithm_group.get() == "informed":
            self.uninformed_frame.pack_forget()
            self.informed_frame.pack(fill=tk.X, pady=5)
            self.local_frame.pack_forget()
            self.csp_frame.pack_forget()
            self.reinforcement_frame.pack_forget()
            self.complex_frame.pack_forget()
            self.belief_frames.pack_forget()  # Hide belief states visualization
        elif self.algorithm_group.get() == "local":
            self.uninformed_frame.pack_forget()
            self.informed_frame.pack_forget()
            self.local_frame.pack(fill=tk.X, pady=5)
            self.csp_frame.pack_forget()
            self.reinforcement_frame.pack_forget()
            self.complex_frame.pack_forget()
            self.belief_frames.pack_forget()  # Hide belief states visualization
        elif self.algorithm_group.get() == "csp":
            self.uninformed_frame.pack_forget()
            self.informed_frame.pack_forget()
            self.local_frame.pack_forget()
            self.csp_frame.pack(fill=tk.X, pady=5)
            self.reinforcement_frame.pack_forget()
            self.complex_frame.pack_forget()
            self.belief_frames.pack_forget()  # Hide belief states visualization
        elif self.algorithm_group.get() == "reinforcement":
            self.uninformed_frame.pack_forget()
            self.informed_frame.pack_forget()
            self.local_frame.pack_forget()
            self.csp_frame.pack_forget()
            self.reinforcement_frame.pack(fill=tk.X, pady=5)
            self.complex_frame.pack_forget()
            self.belief_frames.pack_forget()  # Hide belief states visualization
        else:  # Complex environments
            self.uninformed_frame.pack_forget()
            self.informed_frame.pack_forget()
            self.local_frame.pack_forget()
            self.csp_frame.pack_forget()
            self.reinforcement_frame.pack_forget()
            self.complex_frame.pack(fill=tk.X, pady=5)
            
            # Check if belief state algorithm is selected
            if self.complex_algorithm.get() == "belief_state":
                # Position the belief frames after the board frame for maximum visibility
                self.belief_frames.pack(after=self.board_frame, fill=tk.BOTH, expand=True, pady=10)
                self.initialize_belief_states()
            else:
                self.belief_frames.pack_forget()  # Hide belief states visualization
    
    def update_board_display(self):
        for i in range(3):
            for j in range(3):
                value = self.board[i][j]
                if value == 0:
                    self.board_buttons[i][j].config(text="", style='EmptyTile.TButton')
                else:
                    self.board_buttons[i][j].config(text=str(value), style='Tile.TButton')
    
    def highlight_move(self, move):
        """Highlight the move direction in the UI"""
        # Kiểm tra nếu move không hợp lệ hoặc trống
        if not move or move not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            # Reset tất cả các button về style mặc định
            for r in range(3):
                for c in range(3):
                    self.board_buttons[r][c].configure(style='Tile.TButton')
            return
            
        i, j = None, None
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == 0:
                    i, j = r, c
                    break
            if i is not None:
                break
        
        # Reset all button styles
        for r in range(3):
            for c in range(3):
                self.board_buttons[r][c].configure(style='Tile.TButton')
        
        # Kiểm tra nếu không tìm thấy ô trống
        if i is None or j is None:
            return
                
        # Highlight the tile that will move
        if move == "UP" and i > 0:
            self.board_buttons[i-1][j].configure(style='Move.TButton')
        elif move == "DOWN" and i < 2:
            self.board_buttons[i+1][j].configure(style='Move.TButton')
        elif move == "LEFT" and j > 0:
            self.board_buttons[i][j-1].configure(style='Move.TButton')
        elif move == "RIGHT" and j < 2:
            self.board_buttons[i][j+1].configure(style='Move.TButton')
    
    def shuffle_board(self):
        """Generate a new random but solvable board configuration"""
        # Check if belief state algorithm is selected
        if self.algorithm_group.get() == "complex" and self.complex_algorithm.get() == "belief_state":
            # Generate 4 different random but solvable boards
            self.board = self.generate_solvable_board()
            self.initialize_belief_states()
            
            # Add explanation about the shuffle function for belief states
            self.results_text.config(state="normal")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Belief States shuffled!\n\n")
            self.results_text.insert(tk.END, "The Shuffle function creates 4 random but solvable belief states by:\n")
            self.results_text.insert(tk.END, "1. Starting from the solved goal state\n")
            self.results_text.insert(tk.END, "2. Performing ~30 random valid moves for each belief state\n")
            self.results_text.insert(tk.END, "This creates 4 different but solvable puzzle states that the\n")
            self.results_text.insert(tk.END, "belief state search algorithm will try to solve simultaneously.")
            self.results_text.config(state="disabled")
        else:
            # Regular shuffling for non-belief state algorithms
            self.board = self.generate_solvable_board()
            self.update_board_display()
            
            # Add explanation about the shuffle function
            self.results_text.config(state="normal")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Board shuffled!\n\n")
            self.results_text.insert(tk.END, "The Shuffle function creates a random but solvable board by:\n")
            self.results_text.insert(tk.END, "1. Starting from the solved goal state\n")
            self.results_text.insert(tk.END, "2. Performing ~30 random valid moves\n")
            self.results_text.insert(tk.END, "This ensures the puzzle is always solvable.")
            self.results_text.config(state="disabled")
        
        self.hide_solution_ui()
    
    def generate_solvable_board(self):
        # Start with the goal state
        board = [row[:] for row in self.goal_board]
        
        # Do a series of random moves (ensures the puzzle is solvable)
        state = PuzzleState(board)
        
        for _ in range(30):  # Do 30 random moves
            neighbors = state.get_neighbors()
            if neighbors:
                state = random.choice(neighbors)
        
        return state.board
    
    def is_solvable(self, board):
        # Flatten the board
        flat_board = [num for row in board for num in row if num != 0]
        
        # Count inversions
        inversions = 0
        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[i] > flat_board[j]:
                    inversions += 1
        
        # For a 3x3 puzzle, if the number of inversions is even, the puzzle is solvable
        return inversions % 2 == 0
    
    def solve(self):
        """Solve the puzzle using the selected algorithm"""
        self.hide_solution_ui()
        self.stop_animation()
        
        # Xóa thông tin bước cũ
        self.steps_text = ""
        
        # Determine which algorithm to use
        algorithm_group = self.algorithm_group.get()
        
        if algorithm_group == "uninformed":
            algorithm = self.uninformed_algorithm.get()
            algorithm_name = algorithm.upper()
            heuristic = None
        elif algorithm_group == "informed":
            algorithm = self.informed_algorithm.get()
            heuristic = self.heuristic.get()
            algorithm_name = f"{algorithm.upper()} with {heuristic} heuristic"
        elif algorithm_group == "local":
            algorithm = self.local_algorithm.get()
            heuristic = self.local_heuristic.get()
            algorithm_name = f"{algorithm.upper()} with {heuristic} heuristic"
        elif algorithm_group == "csp":
            algorithm = self.csp_algorithm.get()
            heuristic = self.csp_heuristic.get()
            algorithm_name = f"{algorithm.upper()} with {heuristic} heuristic"
        elif algorithm_group == "reinforcement":
            algorithm = self.reinforcement_algorithm.get()
            heuristic = self.reinforcement_heuristic.get()
            algorithm_name = f"{algorithm.upper()} with {heuristic} heuristic"
        else:  # Complex environments
            algorithm = self.complex_algorithm.get()
            algorithm_name = f"{algorithm.upper()}"
        
        # Update the results text
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Solving with {algorithm_name}...\n")
        self.results_text.config(state="disabled")
        self.root.update()
        
        # Start solving in a separate thread to keep the UI responsive
        thread = threading.Thread(target=self.solve_thread)
        thread.daemon = True
        thread.start()
    
    def solve_thread(self):
        start_time = time.time()
        
        try:
            # Determine which algorithm to use
            algorithm_group = self.algorithm_group.get()
            
            if algorithm_group == "uninformed":
                algorithm = self.uninformed_algorithm.get()
                self.result = solve_puzzle(self.board, self.goal_board, algorithm)
            elif algorithm_group == "informed":
                algorithm = self.informed_algorithm.get()
                heuristic = self.heuristic.get()
                self.result = solve_puzzle_informed(self.board, self.goal_board, algorithm, heuristic)
            elif algorithm_group == "local":
                algorithm = self.local_algorithm.get()
                heuristic = self.local_heuristic.get()
                self.result = solve_puzzle_local_search(self.board, self.goal_board, algorithm, heuristic)
            elif algorithm_group == "csp":
                algorithm = self.csp_algorithm.get()
                heuristic = self.csp_heuristic.get()
                self.result = solve_puzzle_csp(self.board, self.goal_board, algorithm, heuristic)
            elif algorithm_group == "reinforcement":
                algorithm = self.reinforcement_algorithm.get()
                heuristic = self.reinforcement_heuristic.get()
                self.result = solve_puzzle_reinforcement(self.board, self.goal_board, algorithm, heuristic)
            elif algorithm_group == "complex":
                algorithm = self.complex_algorithm.get()
                
                # For belief state algorithm, use the 4 belief states if they exist
                if algorithm == "belief_state" and hasattr(self, 'belief_boards') and len(self.belief_boards) == 4:
                    # Make sure we have the necessary imports
                    from Complex_Environments import PuzzleState, belief_state_search
                    
                    # Convert the board representations to PuzzleState objects
                    initial_states = [PuzzleState(board) for board in self.belief_boards]
                    goal_state = PuzzleState(self.goal_board)
                    
                    # Run the belief state search directly
                    bss_start_time = time.time()
                    result = belief_state_search(initial_states, goal_state, max_depth=20)
                    bss_end_time = time.time()
                    
                    # Make sure the time field is set correctly
                    result["time"] = bss_end_time - bss_start_time
                    self.result = result
                else:
                    # Use the standard solve_puzzle_complex function for other complex algorithms
                    self.result = solve_puzzle_complex(self.board, self.goal_board, algorithm)
            
            # Update UI in the main thread
            self.root.after(0, self.update_results)
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}\n\n{traceback_str}"))
    
    def update_results(self):
        if not self.result or not self.result.get("path"):
            self.results_text.config(state="normal")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No solution found.\n")
            
            if "final_value" in self.result:
                self.results_text.insert(tk.END, f"Final heuristic value: {self.result['final_value']}\n")
            
            if "nodes_expanded" in self.result:
                self.results_text.insert(tk.END, f"Nodes expanded: {self.result.get('nodes_expanded', 'N/A')}\n")
            
            if "iterations" in self.result:
                self.results_text.insert(tk.END, f"Iterations: {self.result['iterations']}\n")
            
            if "generations" in self.result:
                self.results_text.insert(tk.END, f"Generations: {self.result['generations']}\n")
                
            self.results_text.insert(tk.END, f"Time taken: {self.result.get('time', 'N/A'):.4f} seconds\n")
            self.results_text.config(state="disabled")
            return
        
        # Update results text
        path = self.result["path"]
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Solution found in {len(path) - 1} moves\n")
        
        # Thêm thông tin đặc biệt cho AND-OR Tree
        if self.algorithm_group.get() == "complex" and self.complex_algorithm.get() == "and_or_tree":
            self.results_text.insert(tk.END, "\nAND-OR Tree Search chi tiết:\n")
            self.results_text.insert(tk.END, "- Môi trường không xác định (non-deterministic environment)\n")
            self.results_text.insert(tk.END, "- Mỗi hành động có 10% xác suất dẫn đến trạng thái ngẫu nhiên\n")
            self.results_text.insert(tk.END, "- Ô màu XANH LÁ: nước đi dự kiến từ kế hoạch\n")
            self.results_text.insert(tk.END, "- Ô nhấp nháy màu ĐỎ: mô phỏng hành động ngẫu nhiên\n")
            self.results_text.insert(tk.END, "- Thuật toán tạo kế hoạch mạnh mẽ (robust plan) có thể đạt đến\n  mục tiêu dù có nhiễu ngẫu nhiên\n")
            self.results_text.insert(tk.END, "----------------------------------------------------\n")
        
        # Thêm thông tin đặc biệt cho Belief State Search
        elif self.algorithm_group.get() == "complex" and self.complex_algorithm.get() == "belief_state":
            self.results_text.insert(tk.END, "\nBelief State Search chi tiết:\n")
            self.results_text.insert(tk.END, "- Duy trì 4 trạng thái niềm tin đồng thời\n")
            self.results_text.insert(tk.END, "- Mỗi ô hiển thị là một trạng thái niềm tin có thể xảy ra\n")
            self.results_text.insert(tk.END, "- Hỗ trợ tìm kiếm trong môi trường không chắc chắn\n")
            self.results_text.insert(tk.END, "- Thuật toán tìm kế hoạch chung cho tất cả niềm tin\n")
            self.results_text.insert(tk.END, "- Mỗi ô hiển thị là một trạng thái có thể xảy ra thực sự\n")
            self.results_text.insert(tk.END, "\nLƯU Ý VỀ SỰ HỘI TỤ:\n")
            self.results_text.insert(tk.END, "- Các trạng thái niềm tin có thể HỘI TỤ theo thời gian\n")
            self.results_text.insert(tk.END, "- Khi 2 trạng thái trở nên giống hệt nhau, chúng được đánh dấu\n")
            self.results_text.insert(tk.END, "- Các trạng thái đã hội tụ hiển thị màu XÁM\n") 
            self.results_text.insert(tk.END, "- Đây là hiện tượng tự nhiên của thuật toán: giảm sự\n  không chắc chắn khi thu thập thêm thông tin\n")
            self.results_text.insert(tk.END, "----------------------------------------------------\n")
        
        # Thêm thông tin đặc biệt cho Partially Observable Search
        elif self.algorithm_group.get() == "complex" and self.complex_algorithm.get() == "partially_observable":
            self.results_text.insert(tk.END, "\nPartially Observable Search chi tiết:\n")
            self.results_text.insert(tk.END, "- Chỉ quan sát được một phần của trạng thái\n")
            self.results_text.insert(tk.END, "- Các ô ẩn: không thấy được giá trị thực sự\n")
            self.results_text.insert(tk.END, "- Thuật toán phải quyết định dựa vào thông tin không đầy đủ\n")
            self.results_text.insert(tk.END, "----------------------------------------------------\n")
        
        if "nodes_expanded" in self.result:
            self.results_text.insert(tk.END, f"Nodes expanded: {self.result['nodes_expanded']}\n")
        
        if "max_frontier_size" in self.result:
            self.results_text.insert(tk.END, f"Max frontier size: {self.result['max_frontier_size']}\n")
        
        if "iterations" in self.result:
            self.results_text.insert(tk.END, f"Iterations: {self.result['iterations']}\n")
        
        if "generations" in self.result:
            self.results_text.insert(tk.END, f"Generations: {self.result['generations']}\n")
            
        self.results_text.insert(tk.END, f"Time taken: {self.result['time']:.4f} seconds\n")
        
        if "depth_found" in self.result:
            self.results_text.insert(tk.END, f"Solution found at depth: {self.result['depth_found']}\n")
            
        self.results_text.config(state="disabled")
        
        # Show solution navigation
        self.current_step = 0
        self.show_solution_ui()
        self.update_step_display()
    
    def run_animation(self):
        """Start animating the solution"""
        if not self.result or not self.result.get("path") or self.animation_running:
            return
        
        self.animation_running = True
        self.animate_solution()
    
    def animate_solution(self):
        """Display the next step of the solution with animation"""
        if not self.animation_running:
            return
            
        path = self.result["path"]
        if self.current_step < len(path) - 1:
            self.current_step += 1
            
            # Cập nhật trạng thái bảng với trạng thái hiện tại từ path
            current_state = path[self.current_step]
            if isinstance(current_state, tuple):
                self.board = current_state[0]
            elif isinstance(current_state, dict) and "board" in current_state:
                self.board = current_state["board"]
            else:
                self.board = current_state
                
            # Cập nhật hiển thị bảng
            self.update_board_display()
            
            # Cập nhật hiển thị thông tin bước
            self.update_step_display()
            
            # Nếu step có thông tin về move, highlight nó
            if isinstance(current_state, tuple) and len(current_state) > 1:
                move = current_state[1]
                self.highlight_move(move)
            
            # Mô phỏng hành động ngẫu nhiên cho AND-OR Tree
            if self.algorithm_group.get() == "complex" and self.complex_algorithm.get() == "and_or_tree":
                # Hiển thị thông báo về hành động ngẫu nhiên với xác suất 10%
                if random.random() < 0.1 and self.current_step < len(path) - 1:
                    # Hiển thị thông báo về hành động ngẫu nhiên
                    self.results_text.config(state="normal")
                    self.results_text.insert(tk.END, "\nSimulating random action - robust plan continues!\n")
                    self.results_text.see(tk.END)  # Cuộn xuống dưới cùng để hiện thông báo
                    self.results_text.config(state="disabled")
                    
                    # Nhấp nháy màu đỏ để thể hiện hành động ngẫu nhiên
                    for r in range(3):
                        for c in range(3):
                            if self.board[r][c] == 0:  # Tìm ô trống
                                # Tạo hiệu ứng nhấp nháy màu đỏ
                                self.board_buttons[r][c].configure(style='Error.TButton')
                                self.root.after(200, lambda r=r, c=c: self.board_buttons[r][c].configure(style='Tile.TButton'))
                                break
            
            # Cập nhật nhãn bước hiện tại
            if hasattr(self, 'step_label'):
                self.step_label.config(text=f"Step: {self.current_step}/{len(path)-1}")
            
            # Schedule the next step
            self.animation_job = self.root.after(self.animation_speed.get(), self.animate_solution)
        else:
            # Animation complete
            self.animation_running = False
    
    def stop_animation(self):
        """Stop the running animation"""
        if self.animation_running:
            self.animation_running = False
            if self.animation_job:
                self.root.after_cancel(self.animation_job)
                self.animation_job = None
    
    def show_solution_ui(self):
        self.nav_frame.pack(fill=tk.X, pady=10)
    
    def hide_solution_ui(self):
        self.nav_frame.pack_forget()
        
        # Only hide the belief frames if not using the belief state algorithm 
        # or if we're resetting everything
        if not (self.algorithm_group.get() == "complex" and 
                self.complex_algorithm.get() == "belief_state"):
            self.belief_frames.pack_forget()
            
        self.result = None
        self.current_step = 0
        self.stop_animation()
    
    def update_step_display(self):
        # Clear current step display
        for widget in self.step_display_frame.winfo_children():
            if widget != self.steps_text:
                widget.destroy()
        
        if not self.result or not self.result.get("path") or self.current_step >= len(self.result["path"]):
            return
        
        # Create new display based on the type of algorithm
        if self.algorithm_group.get() == "complex" and self.complex_algorithm.get() == "belief_state":
            self.show_belief_states()
            return
            
        if (self.algorithm_group.get() == "csp" and self.csp_algorithm.get() == "min_conflicts_labeling" and 
            "labeling" in self.result and self.result["labeling"]):
            self.show_min_conflicts_labeling_step()
            return
            
        # Add special display for backtracking
        if self.algorithm_group.get() == "csp" and self.csp_algorithm.get() == "backtracking":
            self.show_backtracking_step()
            return
            
        # Standard step display
        step_info = ttk.Frame(self.step_display_frame)
        step_info.pack(fill=tk.X)
        
        # Show current step info
        if self.algorithm_group.get() == "complex" and self.complex_algorithm.get() == "partially_observable":
            ttk.Label(step_info, text=f"Step {self.current_step} of {len(self.result['path'])-1}", 
                    font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
            
            # Add special note for partially observable
            ttk.Label(step_info, text="(Hidden tiles shown in gray)", 
                    font=("Arial", 10, "italic")).pack(side=tk.LEFT, padx=10)
        else:
            ttk.Label(step_info, text=f"Step {self.current_step} of {len(self.result['path'])-1}", 
                    font=("Arial", 12, "bold")).pack(padx=5)
        
        # Show the move that led to this state
        move_frame = ttk.Frame(self.step_display_frame)
        move_frame.pack(fill=tk.X, pady=5)
        
        # Get current state information
        current_state = self.result["path"][self.current_step]
        board_to_display = None
        
        if isinstance(current_state, tuple):
            board_to_display = current_state[0]
            move = current_state[1] if len(current_state) > 1 else ""
        elif isinstance(current_state, dict) and "board" in current_state:
            board_to_display = current_state["board"]
            move = current_state.get("move", "")
        else:
            board_to_display = current_state
            move = "MOVE"
        
        # Create a frame for board visualization and move information
        board_move_frame = ttk.Frame(self.step_display_frame)
        board_move_frame.pack(fill=tk.X, pady=5)
        
        # Create a nested frame for the board visualization
        board_frame = ttk.LabelFrame(board_move_frame, text="Trạng thái bảng", padding=5)
        board_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Display the board state
        for i in range(3):
            for j in range(3):
                value = board_to_display[i][j]
                if value == 0:
                    ttk.Button(board_frame, text="", width=3, style='EmptyTile.TButton').grid(row=i, column=j, padx=2, pady=2)
                else:
                    ttk.Button(board_frame, text=str(value), width=3, style='Tile.TButton').grid(row=i, column=j, padx=2, pady=2)
        
        # Create a frame for move information
        move_info_frame = ttk.Frame(board_move_frame)
        move_info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        if self.current_step > 0:
            ttk.Label(move_info_frame, text=f"Move: {move}", font=("Arial", 12)).pack(anchor=tk.W)
        else:
            ttk.Label(move_info_frame, text="Initial State", font=("Arial", 12)).pack(anchor=tk.W)
        
        # Additional details depending on algorithm
        if self.algorithm_group.get() == "informed" or self.algorithm_group.get() == "local":
            detail_frame = ttk.Frame(self.step_display_frame)
            detail_frame.pack(fill=tk.X, pady=5)
            
            if self.algorithm_group.get() == "informed":
                heuristic = self.heuristic.get().capitalize()
            else:
                heuristic = self.local_heuristic.get().capitalize()
            
            h_value = 0
            if heuristic.lower() == "manhattan":
                h_value = manhattan_distance(board_to_display, self.goal_board)
            elif heuristic.lower() == "misplaced":
                h_value = misplaced_tiles(board_to_display, self.goal_board)
            
            ttk.Label(detail_frame, text=f"{heuristic} Distance: {h_value}", 
                    font=("Arial", 12)).pack(padx=5)
        
        # Show performance stats
        stats_frame = ttk.Frame(self.step_display_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(stats_frame, text="Performance Statistics:", 
                font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5)
        
        if "time" in self.result:
            ttk.Label(stats_frame, text=f"Time: {self.result['time']:.4f} seconds", 
                    font=("Arial", 10)).pack(anchor=tk.W, padx=20)
        
        if "steps" in self.result:
            ttk.Label(stats_frame, text=f"Total Steps: {self.result['steps']}", 
                    font=("Arial", 10)).pack(anchor=tk.W, padx=20)
    
    def show_min_conflicts_labeling_step(self):
        """Display a Min-Conflicts step using the labeling approach"""
        if not self.result or not self.result.get("path") or self.current_step >= len(self.result["path"]):
            return
            
        step_data = self.result["path"][self.current_step]
        
        # Header frame
        header_frame = ttk.Frame(self.step_display_frame)
        header_frame.pack(fill=tk.X)
        
        ttk.Label(header_frame, text=f"Step {self.current_step} of {len(self.result['path'])-1} - Min-Conflicts (Labeling)", 
                font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Move information
        if self.current_step > 0:
            move_frame = ttk.Frame(self.step_display_frame)
            move_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(move_frame, text=f"Action: {step_data['move']}", 
                    font=("Arial", 11, "bold"), foreground="#0066cc").pack(padx=5)
        else:
            ttk.Label(header_frame, text=" - Initial State", 
                    font=("Arial", 11), foreground="#666666").pack(side=tk.LEFT)
        
        # Create frame for swap visualization
        if self.current_step > 0 and step_data['swap_positions']:
            swap_frame = ttk.LabelFrame(self.step_display_frame, text="Swap Visualization", padding=10)
            swap_frame.pack(fill=tk.X, padx=5, pady=10)
            
            pos1, pos2 = step_data['swap_positions']
            val1, val2 = step_data['swap_values']
            
            # Explanation of the swap
            explanation = ttk.Frame(swap_frame)
            explanation.pack(fill=tk.X)
            
            ttk.Label(explanation, 
                    text=f"Swapped position ({pos1[0]},{pos1[1]}) containing value {val1}", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            ttk.Label(explanation, 
                    text=f"with position ({pos2[0]},{pos2[1]}) containing value {val2}", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            
            # Visual representation of the swap with small board layouts
            visual_frame = ttk.Frame(swap_frame)
            visual_frame.pack(pady=10)
            
            # Get the board before the swap
            prev_board = self.result["path"][self.current_step-1]["board"] if self.current_step > 0 else self.result["path"][0]["board"]
            
            # Create small board representation for "before swap"
            before_frame = ttk.LabelFrame(visual_frame, text="Before Swap", padding=5)
            before_frame.pack(side=tk.LEFT, padx=10)
            
            for i in range(3):
                for j in range(3):
                    value = prev_board[i][j]
                    text = str(value) if value != 0 else " "
                    style = 'TButton'
                    
                    # Highlight the positions being swapped
                    if (i, j) == pos1 or (i, j) == pos2:
                        style = 'Move.TButton'
                    
                    button = ttk.Button(before_frame, text=text, width=2, style=style)
                    button.grid(row=i, column=j, padx=2, pady=2)
            
            # Arrow pointing to "after swap"
            arrow_label = ttk.Label(visual_frame, text="→", font=("Arial", 16, "bold"))
            arrow_label.pack(side=tk.LEFT, padx=5)
            
            # Create small board representation for "after swap"
            after_frame = ttk.LabelFrame(visual_frame, text="After Swap", padding=5)
            after_frame.pack(side=tk.LEFT, padx=10)
            
            for i in range(3):
                for j in range(3):
                    value = step_data["board"][i][j]
                    text = str(value) if value != 0 else " "
                    style = 'TButton'
                    
                    # Highlight the positions that were swapped
                    if (i, j) == pos1 or (i, j) == pos2:
                        style = 'Move.TButton'
                    
                    button = ttk.Button(after_frame, text=text, width=2, style=style)
                    button.grid(row=i, column=j, padx=2, pady=2)
        
        # Conflict information
        conflicts_frame = ttk.LabelFrame(self.step_display_frame, text="Conflicts Information", padding=10)
        conflicts_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Display conflict counts
        if self.current_step > 0:
            before_after = ttk.Frame(conflicts_frame)
            before_after.pack(fill=tk.X)
            
            ttk.Label(before_after, text=f"Conflicts before: {step_data['conflicts_before']}", 
                    font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
            
            ttk.Label(before_after, text="→", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            
            ttk.Label(before_after, text=f"Conflicts after: {step_data['conflicts_after']}", 
                    font=("Arial", 10, "bold"), 
                    foreground="#006600" if step_data['conflicts_after'] < step_data['conflicts_before'] else "#cc0000").pack(side=tk.LEFT, padx=10)
            
            # Display conflict reduction
            reduction = step_data['conflicts_before'] - step_data['conflicts_after']
            if reduction > 0:
                ttk.Label(conflicts_frame, text=f"Reduced conflicts by: {reduction}", 
                        font=("Arial", 10, "bold"), foreground="#006600").pack(anchor=tk.W, padx=10, pady=5)
            elif reduction < 0:
                ttk.Label(conflicts_frame, text=f"Increased conflicts by: {abs(reduction)} (accepted to escape local minimum)", 
                        font=("Arial", 10), foreground="#cc0000").pack(anchor=tk.W, padx=10, pady=5)
            else:
                ttk.Label(conflicts_frame, text="No change in conflicts", 
                        font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=5)
        else:
            ttk.Label(conflicts_frame, text=f"Initial conflicts: {step_data['conflicts_after']}", 
                    font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10)
        
        # Explanation of the labeling approach
        explanation_frame = ttk.LabelFrame(self.step_display_frame, text="Labeling Approach Explanation", padding=10)
        explanation_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(explanation_frame, 
                text="In the labeling approach, Min-Conflicts treats positions as variables and values as labels.", 
                font=("Arial", 10), wraplength=500).pack(anchor=tk.W)
        
        ttk.Label(explanation_frame, 
                text="Instead of moving the blank space, we directly swap pairs of values to reduce conflicts.", 
                font=("Arial", 10), wraplength=500).pack(anchor=tk.W, pady=5)
        
        ttk.Label(explanation_frame, 
                text="A conflict occurs when a value is not in its goal position.", 
                font=("Arial", 10), wraplength=500).pack(anchor=tk.W, pady=5)
        
        # Performance stats
        stats_frame = ttk.Frame(self.step_display_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(stats_frame, text="Performance Statistics:", 
                font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5)
        
        if "time" in self.result:
            ttk.Label(stats_frame, text=f"Time: {self.result['time']:.4f} seconds", 
                    font=("Arial", 10)).pack(anchor=tk.W, padx=20)
        
        if "steps" in self.result:
            ttk.Label(stats_frame, text=f"Total Steps: {self.result['steps']}", 
                    font=("Arial", 10)).pack(anchor=tk.W, padx=20)
    
    def show_backtracking_step(self):
        """Display a backtracking step with backtracking information and highlight the board using tk.Button for color."""
        if not self.result or not self.result.get("path") or self.current_step >= len(self.result["path"]):
            return
        
        step_data = self.result["path"][self.current_step]
        
        # Header frame
        header_frame = ttk.Frame(self.step_display_frame)
        header_frame.pack(fill=tk.X)
        
        ttk.Label(header_frame, text=f"Step {self.current_step} of {len(self.result['path'])-1} - Backtracking Search", 
                font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Move information
        if self.current_step > 0:
            move_frame = ttk.Frame(self.step_display_frame)
            move_frame.pack(fill=tk.X, pady=5)
            
            move = step_data[1] if isinstance(step_data, tuple) else ""
            if move.startswith("Trying:"):
                ttk.Label(move_frame, text=f"Action: {move}", 
                        font=("Arial", 11, "bold"), foreground="#0066cc").pack(padx=5)
                board_bg = '#cce6ff'  # Xanh dương nhạt
            elif move.startswith("Backtrack:"):
                ttk.Label(move_frame, text=f"Action: {move}", 
                        font=("Arial", 11, "bold"), foreground="#cc0000").pack(padx=5)
                board_bg = '#ffcccc'  # Đỏ nhạt
            elif move == "GOAL":
                ttk.Label(move_frame, text="Goal State Reached!", 
                        font=("Arial", 11, "bold"), foreground="#006600").pack(padx=5)
                board_bg = '#ccffcc'  # Xanh lá nhạt
            else:
                ttk.Label(move_frame, text=f"Action: {move}", 
                        font=("Arial", 11, "bold")).pack(padx=5)
                board_bg = 'white'
        else:
            ttk.Label(header_frame, text=" - Initial State", 
                    font=("Arial", 11), foreground="#666666").pack(side=tk.LEFT)
            board_bg = 'white'
        
        # Create frame for board visualization
        board_frame = ttk.LabelFrame(self.step_display_frame, text="Board State", padding=10)
        board_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Get the board state
        board = step_data[0] if isinstance(step_data, tuple) else step_data
        
        # Display the board with highlight color using tk.Button
        for i in range(3):
            for j in range(3):
                value = board[i][j]
                text = str(value) if value != 0 else " "
                if value == 0:
                    btn = tk.Button(board_frame, text=text, width=3, height=1, font=("Arial", 14, "bold"), bg='black', fg='white', relief=tk.RAISED)
                else:
                    btn = tk.Button(board_frame, text=text, width=3, height=1, font=("Arial", 14, "bold"), bg=board_bg, relief=tk.RAISED)
                btn.grid(row=i, column=j, padx=2, pady=2)
        
        # Explanation frame (giữ nguyên như cũ)
        explanation_frame = ttk.LabelFrame(self.step_display_frame, text="Backtracking Explanation", padding=10)
        explanation_frame.pack(fill=tk.X, padx=5, pady=10)
        
        if self.current_step > 0:
            move = step_data[1] if isinstance(step_data, tuple) else ""
            if move.startswith("Trying:"):
                ttk.Label(explanation_frame, 
                        text="The algorithm is trying a new move to explore a potential solution path.", 
                        font=("Arial", 10), wraplength=500).pack(anchor=tk.W)
                ttk.Label(explanation_frame, 
                        text="This move is chosen based on the heuristic function to prioritize more promising paths.", 
                        font=("Arial", 10), wraplength=500).pack(anchor=tk.W, pady=5)
            elif move.startswith("Backtrack:"):
                ttk.Label(explanation_frame, 
                        text="The algorithm is backtracking because the previous move led to a dead end.", 
                        font=("Arial", 10), wraplength=500).pack(anchor=tk.W)
                ttk.Label(explanation_frame, 
                        text="This is a key feature of backtracking: when a path fails, we go back and try a different move.", 
                        font=("Arial", 10), wraplength=500).pack(anchor=tk.W, pady=5)
            elif move == "GOAL":
                ttk.Label(explanation_frame, 
                        text="The goal state has been reached! The solution path has been found.", 
                        font=("Arial", 10), wraplength=500).pack(anchor=tk.W)
                ttk.Label(explanation_frame, 
                        text="The backtracking algorithm has successfully found a path from the initial state to the goal state.", 
                        font=("Arial", 10), wraplength=500).pack(anchor=tk.W, pady=5)
        else:
            ttk.Label(explanation_frame, 
                    text="This is the initial state. The backtracking algorithm will systematically explore possible moves.", 
                    font=("Arial", 10), wraplength=500).pack(anchor=tk.W)
            ttk.Label(explanation_frame, 
                    text="The algorithm will use a heuristic function to guide its search and try to find the most efficient path to the goal.", 
                    font=("Arial", 10), wraplength=500).pack(anchor=tk.W, pady=5)
        
        # Performance stats
        stats_frame = ttk.Frame(self.step_display_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(stats_frame, text="Performance Statistics:", 
                font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5)
        
        if "time" in self.result:
            ttk.Label(stats_frame, text=f"Time: {self.result['time']:.4f} seconds", 
                    font=("Arial", 10)).pack(anchor=tk.W, padx=20)
        
        if "depth" in self.result:
            ttk.Label(stats_frame, text=f"Solution Depth: {self.result['depth']}", 
                    font=("Arial", 10)).pack(anchor=tk.W, padx=20)
    
    def first_step(self):
        if not self.result or not self.result.get("path"):
            return
        
        self.stop_animation()
        self.current_step = 0
        
        # Cập nhật trạng thái bảng với bước đầu tiên
        path = self.result["path"]
        current_state = path[self.current_step]
        if isinstance(current_state, tuple):
            self.board = current_state[0]
        elif isinstance(current_state, dict) and "board" in current_state:
            self.board = current_state["board"]
        else:
            self.board = current_state
            
        # Cập nhật hiển thị bảng
        self.update_board_display()
        
        # Cập nhật nhãn bước hiện tại
        if hasattr(self, 'step_label'):
            self.step_label.config(text=f"Step: {self.current_step}/{len(path)-1}")
            
        self.update_step_display()
    
    def prev_step(self):
        if not self.result or not self.result.get("path"):
            return
        
        self.stop_animation()
        if self.current_step > 0:
            self.current_step -= 1
            
            # Cập nhật trạng thái bảng với bước trước đó
            path = self.result["path"]
            current_state = path[self.current_step]
            if isinstance(current_state, tuple):
                self.board = current_state[0]
            elif isinstance(current_state, dict) and "board" in current_state:
                self.board = current_state["board"]
            else:
                self.board = current_state
                
            # Cập nhật hiển thị bảng
            self.update_board_display()
            
            # Highlight nước đi nếu có
            if isinstance(current_state, tuple) and len(current_state) > 1:
                move = current_state[1]
                self.highlight_move(move)
                
            # Cập nhật nhãn bước hiện tại
            if hasattr(self, 'step_label'):
                self.step_label.config(text=f"Step: {self.current_step}/{len(path)-1}")
                
            self.update_step_display()
    
    def next_step(self):
        if not self.result or not self.result.get("path"):
            return
        
        self.stop_animation()
        path = self.result["path"]
        if self.current_step < len(path) - 1:
            self.current_step += 1
            
            # Cập nhật trạng thái bảng với bước tiếp theo
            current_state = path[self.current_step]
            if isinstance(current_state, tuple):
                self.board = current_state[0]
            elif isinstance(current_state, dict) and "board" in current_state:
                self.board = current_state["board"]
            else:
                self.board = current_state
                
            # Cập nhật hiển thị bảng
            self.update_board_display()
            
            # Highlight nước đi nếu có
            if isinstance(current_state, tuple) and len(current_state) > 1:
                move = current_state[1]
                self.highlight_move(move)
                
            # Cập nhật nhãn bước hiện tại
            if hasattr(self, 'step_label'):
                self.step_label.config(text=f"Step: {self.current_step}/{len(path)-1}")
                
            self.update_step_display()
    
    def last_step(self):
        if not self.result or not self.result.get("path"):
            return
        
        self.stop_animation()
        path = self.result["path"]
        self.current_step = len(path) - 1
        
        # Cập nhật trạng thái bảng với bước cuối cùng
        current_state = path[self.current_step]
        if isinstance(current_state, tuple):
            self.board = current_state[0]
        elif isinstance(current_state, dict) and "board" in current_state:
            self.board = current_state["board"]
        else:
            self.board = current_state
            
        # Cập nhật hiển thị bảng
        self.update_board_display()
        
        # Highlight nước đi nếu có
        if isinstance(current_state, tuple) and len(current_state) > 1:
            move = current_state[1]
            self.highlight_move(move)
            
        # Cập nhật nhãn bước hiện tại
        if hasattr(self, 'step_label'):
            self.step_label.config(text=f"Step: {self.current_step}/{len(path)-1}")
            
        self.update_step_display()
    
    def reset(self):
        # Reset to goal state
        self.board = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]
        
        # Update the board display
        self.update_board_display()
        
        # Reset belief states if showing belief state algorithm
        if self.algorithm_group.get() == "complex" and self.complex_algorithm.get() == "belief_state":
            # Initialize all 4 belief states to the goal state
            self.belief_boards = [[row[:] for row in self.goal_board] for _ in range(4)]
            
            # Update all belief state displays
            for idx in range(4):
                # Reset the frame style to default
                belief_board_frame = self.belief_board_buttons[idx][0][0].master.master
                belief_board_frame.configure(text=f"Belief State {idx+1}", style='Belief.TLabelframe')
                
                for i in range(3):
                    for j in range(3):
                        value = self.goal_board[i][j]
                        button = self.belief_board_buttons[idx][i][j]
                        if value == 0:
                            button.config(text="")
                        else:
                            button.config(text=str(value))
                        button.config(style='BeliefTile.TButton')
            
            # Make sure the belief frames are shown in the correct position
            if not self.belief_frames.winfo_ismapped():
                self.belief_frames.pack(after=self.board_frame, fill=tk.BOTH, expand=True, pady=10)
        else:
            # Hide belief frames if not using belief state algorithm
            self.belief_frames.pack_forget()
        
        self.hide_solution_ui()
        
        # Clear results
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state="disabled")
        
        # Clear step display
        self.step_display.config(state="normal")
        self.step_display.delete(1.0, tk.END)
        self.step_display.config(state="disabled")
        
        # Reset steps text
        self.steps_text = ""
        
        # Reset step counter if it exists
        if hasattr(self, 'step_label'):
            self.step_label.config(text="Step: 0/0")
        
        # Clear any previous solution
        self.result = None
        self.current_step = 0
    
    def save_steps_to_file(self):
        """Lưu thông tin các bước vào file Step.txt"""
        with open("Step.txt", "w", encoding="utf-8") as file:
            # Viết thông tin thuật toán
            algo_group = self.algorithm_group.get()
            if algo_group == "uninformed":
                algorithm = self.uninformed_algorithm.get().upper()
                file.write(f"Algorithm: {algorithm}\n")
            elif algo_group == "informed":
                algorithm = self.informed_algorithm.get().upper()
                heuristic = self.heuristic.get()
                file.write(f"Algorithm: {algorithm} with {heuristic} heuristic\n")
            elif algo_group == "local":
                algorithm = self.local_algorithm.get().upper()
                heuristic = self.local_heuristic.get()
                file.write(f"Algorithm: {algorithm} with {heuristic} heuristic\n")
            elif algo_group == "csp":
                algorithm = self.csp_algorithm.get().upper()
                file.write(f"Algorithm: {algorithm}\n")
            elif algo_group == "reinforcement":
                algorithm = self.reinforcement_algorithm.get().upper()
                file.write(f"Algorithm: {algorithm}\n")
            else:  # Complex
                algorithm = self.complex_algorithm.get().upper()
                file.write(f"Algorithm: {algorithm}\n")
                
            file.write("\n" + "=" * 40 + "\n\n")
            file.write(self.steps_text)

    def initialize_belief_states(self):
        """Initialize 4 different belief states for visualization"""
        # Create 4 different random but solvable boards for the belief states
        belief_boards = []
        
        # First belief state is the current board
        belief_boards.append(self.board)
        
        # Generate 3 more random but solvable boards
        for _ in range(3):
            belief_boards.append(self.generate_solvable_board())
        
        # Update the belief state visualization
        for idx, board in enumerate(belief_boards):
            # Reset the frame style to default
            belief_board_frame = self.belief_board_buttons[idx][0][0].master.master
            belief_board_frame.configure(text=f"Belief State {idx+1}", style='Belief.TLabelframe')
            
            # Update the buttons
            for i in range(3):
                for j in range(3):
                    value = board[i][j]
                    button = self.belief_board_buttons[idx][i][j]
                    if value == 0:
                        button.config(text="")
                    else:
                        button.config(text=str(value))
                    button.config(style='BeliefTile.TButton')
        
        # Store the belief boards for later use
        self.belief_boards = belief_boards
        
        # Update results text with explanation
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Belief State Search đã được khởi tạo!\n\n")
        self.results_text.insert(tk.END, "- Hiển thị 4 trạng thái niềm tin đồng thời\n")
        self.results_text.insert(tk.END, "- Mỗi trạng thái thể hiện một khả năng xảy ra của puzzle\n")
        self.results_text.insert(tk.END, "- Thuật toán sẽ tìm một giải pháp hoạt động cho TẤT CẢ các trạng thái\n")
        self.results_text.insert(tk.END, "- Các trạng thái có thể hội tụ (converge) trong quá trình tìm kiếm\n")
        self.results_text.insert(tk.END, "\nChú thích màu sắc:\n")
        self.results_text.insert(tk.END, "- Xanh lá cây: Ô sẽ di chuyển trong bước tiếp theo\n")
        self.results_text.insert(tk.END, "- Xanh dương: Trạng thái niềm tin đang hoạt động\n")
        self.results_text.insert(tk.END, "- Đỏ nhạt: Trạng thái đã hội tụ với trạng thái khác\n")
        self.results_text.insert(tk.END, "- Xanh lục: Trạng thái nguồn (có trạng thái khác hội tụ vào)\n")
        self.results_text.config(state="disabled")
        
        # Make sure the belief frames are shown
        if not self.belief_frames.winfo_ismapped():
            self.belief_frames.pack(after=self.board_frame, fill=tk.BOTH, expand=True, pady=10)

    def on_complex_algorithm_change(self, *args):
        """Callback function when complex algorithm changes"""
        # Only act if the complex algorithm tab is currently selected
        if self.algorithm_group.get() == "complex":
            self.update_algorithm_options()

    def show_all_steps(self):
        """Hiển thị toàn bộ các bước đi trong một cửa sổ mới"""
        if not self.result or not self.result.get("path"):
            messagebox.showinfo("Thông báo", "Vui lòng giải puzzle trước khi xem các bước đi!")
            return
            
        # Tạo cửa sổ mới
        steps_window = tk.Toplevel(self.root)
        steps_window.title("Tất cả các bước đi")
        steps_window.geometry("800x600")
        
        # Tạo container với thanh cuộn
        container = ttk.Frame(steps_window)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tạo canvas và thanh cuộn
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # Tạo frame trong canvas để chứa toàn bộ bước đi
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Cấu hình canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Đặt canvas và scrollbar vào container
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Tiêu đề
        algorithm_group = self.algorithm_group.get()
        if algorithm_group == "uninformed":
            algorithm = self.uninformed_algorithm.get().upper()
            title = f"Giải pháp sử dụng {algorithm}"
        elif algorithm_group == "informed":
            algorithm = self.informed_algorithm.get().upper()
            heuristic = self.heuristic.get()
            title = f"Giải pháp sử dụng {algorithm} với heuristic {heuristic}"
        elif algorithm_group == "local":
            algorithm = self.local_algorithm.get().upper()
            heuristic = self.local_heuristic.get()
            title = f"Giải pháp sử dụng {algorithm} với heuristic {heuristic}"
        elif algorithm_group == "csp":
            algorithm = self.csp_algorithm.get().upper()
            heuristic = self.csp_heuristic.get()
            title = f"Giải pháp sử dụng {algorithm} với heuristic {heuristic}"
        elif algorithm_group == "reinforcement":
            algorithm = self.reinforcement_algorithm.get().upper()
            heuristic = self.reinforcement_heuristic.get()
            title = f"Giải pháp sử dụng {algorithm} với heuristic {heuristic}"
        else:  # Complex environments
            algorithm = self.complex_algorithm.get().upper()
            title = f"Giải pháp sử dụng {algorithm}"
            
        ttk.Label(scrollable_frame, text=title, font=("Arial", 14, "bold")).pack(pady=10)
        
        path = self.result["path"]
        total_steps = len(path) - 1
        ttk.Label(scrollable_frame, text=f"Số bước di chuyển: {total_steps}", font=("Arial", 12)).pack(pady=5)
        
        # Hiển thị từng bước
        for step_num in range(len(path)):
            step_frame = ttk.LabelFrame(scrollable_frame, text=f"Bước {step_num}/{total_steps}", padding=10)
            step_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Hiển thị bảng của bước hiện tại
            current_state = path[step_num]
            board_to_display = None
            move_text = "Trạng thái ban đầu"
            
            if isinstance(current_state, tuple):
                board_to_display = current_state[0]
                if step_num > 0:
                    move_text = f"Di chuyển: {current_state[1]}"
            elif isinstance(current_state, dict) and "board" in current_state:
                board_to_display = current_state["board"]
                if step_num > 0 and "move" in current_state:
                    move_text = f"Hành động: {current_state['move']}"
            else:
                board_to_display = current_state
            
            # Tạo frame để hiển thị bảng 3x3
            board_display = ttk.Frame(step_frame)
            board_display.pack(side=tk.LEFT, padx=10, pady=5)
            
            # Hiển thị bảng
            for i in range(3):
                for j in range(3):
                    value = board_to_display[i][j]
                    if value == 0:
                        ttk.Button(board_display, text="", width=3, style='EmptyTile.TButton').grid(row=i, column=j, padx=2, pady=2)
                    else:
                        ttk.Button(board_display, text=str(value), width=3, style='Tile.TButton').grid(row=i, column=j, padx=2, pady=2)
            
            # Hiển thị thông tin về bước đi
            info_frame = ttk.Frame(step_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            
            ttk.Label(info_frame, text=move_text, font=("Arial", 11)).pack(anchor=tk.W)
            
            # Nếu là thuật toán có thông tin hoặc tìm kiếm cục bộ, hiển thị giá trị heuristic
            if algorithm_group in ["informed", "local"]:
                heuristic_name = self.heuristic.get() if algorithm_group == "informed" else self.local_heuristic.get()
                
                h_value = 0
                if heuristic_name.lower() == "manhattan":
                    h_value = manhattan_distance(board_to_display, self.goal_board)
                elif heuristic_name.lower() == "misplaced":
                    h_value = misplaced_tiles(board_to_display, self.goal_board)
                
                ttk.Label(info_frame, text=f"{heuristic_name.capitalize()} Distance: {h_value}", font=("Arial", 10)).pack(anchor=tk.W)
            
            # Nếu là Min-Conflicts với dán nhãn, hiển thị thông tin về xung đột
            if algorithm_group == "csp" and self.csp_algorithm.get() == "min_conflicts_labeling" and isinstance(current_state, dict):
                if "conflicts_after" in current_state:
                    ttk.Label(info_frame, text=f"Số xung đột: {current_state['conflicts_after']}", font=("Arial", 10)).pack(anchor=tk.W)
                if step_num > 0 and "conflicts_before" in current_state and "conflicts_after" in current_state:
                    reduction = current_state['conflicts_before'] - current_state['conflicts_after']
                    if reduction > 0:
                        ttk.Label(info_frame, text=f"Giảm {reduction} xung đột", font=("Arial", 10), foreground="green").pack(anchor=tk.W)
                    elif reduction < 0:
                        ttk.Label(info_frame, text=f"Tăng {abs(reduction)} xung đột", font=("Arial", 10), foreground="red").pack(anchor=tk.W)

if __name__ == "__main__":
    root = tk.Tk()
    
    # Create a style for highlighted moves
    style = ttk.Style()
    style.configure('Move.TButton', background='light green', font=('Arial', 14, 'bold'))
    
    app = PuzzleGUI(root)
    root.mainloop() 