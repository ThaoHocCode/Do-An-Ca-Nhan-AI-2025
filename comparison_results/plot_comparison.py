import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Define results directory
RESULTS_DIR = 'results'

def get_filename(algo_name):
    if algo_name == 'A*':
        return 'results_a_star.json'
    elif algo_name == 'IDA*':
        return 'results_ida_star.json'
    else:
        return f'results_{algo_name.lower().replace(" ", "_")}.json'

def load_results(algorithm_names):
    results = {}
    for algo in algorithm_names:
        filename = os.path.join(RESULTS_DIR, get_filename(algo))
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                results[algo] = json.load(f)
        else:
            results[algo] = None
    return results

def plot_group_comparison(group_name, algorithms, metrics, results):
    """Plot comparison charts for a group of algorithms"""
    # Create a figure with 4 subplots (one for each metric)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Comparison of {group_name} Algorithms', fontsize=16)
    
    # Plot each metric
    for idx, (metric, title) in enumerate(metrics.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Get data for this metric
        data = []
        for algo in algorithms:
            if algo in results and metric in results[algo] and results[algo][metric] is not None:
                data.append(results[algo][metric])
            else:
                data.append(0)  # Use 0 for missing data
        
        # Create bar chart
        bars = ax.bar(algorithms, data)
        
        # Add value labels on top of bars
        for bar, algo in zip(bars, algorithms):
            height = bar.get_height()
            if height == 0:  # No data
                label = 'N/A'
            elif metric == 'time':
                # Format time in milliseconds if less than 1 second
                if height < 1:
                    label = f'{height*1000:.2f}ms'
                else:
                    label = f'{height:.2f}s'
            else:
                label = f'{height:.2f}'
            
            # Position the label
            if height == 0:
                y_pos = 0.1  # Place N/A label near the bottom
            else:
                y_pos = height
                
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   label,
                   ha='center', va='bottom')
        
        # Customize the plot
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        # Add a light gray background for bars with no data
        for bar, height in zip(bars, data):
            if height == 0:
                bar.set_color('lightgray')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure to results directory
    filename = os.path.join(RESULTS_DIR, f'{group_name.lower().replace(" ", "_")}_comparison.png')
    plt.savefig(filename)
    plt.close()

def main():
    # Define metrics to compare
    metrics = {
        'moves': 'Number of Moves',
        'nodes_expanded': 'Nodes Expanded',
        'max_frontier_size': 'Max Frontier Size',
        'time': 'Time Taken'
    }
    
    # Chỉ so sánh 2 thuật toán cho nhóm CSP
    groups = {
        'CSP': ['Backtracking', 'Min-Conflicts (Labeling)']
    }
    
    # Load results for each group
    for group_name, algorithms in groups.items():
        results = load_results(algorithms)
        
        # Plot comparison for this group
        plot_group_comparison(group_name, algorithms, metrics, results)

if __name__ == "__main__":
    main() 