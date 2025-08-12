#!/usr/bin/env python3
"""
Water Jug Problem Interactive GUI
A visual interface for solving and exploring the Water Jug Problem using various AI search algorithms.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import time
import threading
from collections import deque
import math

# Import the solver - execute the file directly since it's not a module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load the solver code
exec(open('water-jug-ai.py').read(), globals())


class WaterJugGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Water Jug Problem AI Solver - Interactive Visualization")
        self.root.geometry("1400x900")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Variables
        self.current_problem = None
        self.current_solution = None
        self.current_tree = None
        self.animation_speed = tk.IntVar(value=500)  # milliseconds
        self.is_animating = False
        self.animation_thread = None
        self.current_step = 0
        
        # Colors
        self.colors = {
            'water': '#4A90E2',
            'jug': '#E0E0E0',
            'jug_outline': '#333333',
            'success': '#4CAF50',
            'current': '#FFC107',
            'explored': '#9E9E9E',
            'path': '#2196F3',
            'background': '#F5F5F5'
        }
        
        self.setup_ui()
        self.load_default_problem()
        
    def setup_ui(self):
        """Setup the main UI components"""
        # Create menu bar
        self.create_menu()
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create three main panels
        self.create_left_panel(main_container)
        self.create_middle_panel(main_container)
        self.create_right_panel(main_container)
        
        # Create status bar
        self.create_status_bar()
        
    def create_menu(self):
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Problem", command=self.new_problem)
        file_menu.add_command(label="Export Solution", command=self.export_solution)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Search Tree", command=self.show_search_tree)
        view_menu.add_command(label="Show Statistics", command=self.show_statistics)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About Algorithms", command=self.show_algorithm_help)
        help_menu.add_command(label="How to Use", command=self.show_usage_help)
        
    def create_left_panel(self, parent):
        """Create the left panel for jug visualization"""
        left_frame = ttk.LabelFrame(parent, text="Jug Visualization", padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Canvas for drawing jugs
        self.canvas = tk.Canvas(left_frame, width=600, height=400, bg=self.colors['background'])
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.play_button = ttk.Button(control_frame, text="▶ Play", command=self.play_solution)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(control_frame, text="⏸ Pause", command=self.pause_animation, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(control_frame, text="⏮ Reset", command=self.reset_visualization)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.step_button = ttk.Button(control_frame, text="⏭ Step", command=self.step_forward)
        self.step_button.pack(side=tk.LEFT, padx=5)
        
        # Speed control
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=(20, 5))
        speed_scale = ttk.Scale(control_frame, from_=100, to=2000, variable=self.animation_speed, 
                                orient=tk.HORIZONTAL, length=150)
        speed_scale.pack(side=tk.LEFT)
        
        # Step indicator
        self.step_label = ttk.Label(left_frame, text="Step: 0 / 0", font=("Arial", 12))
        self.step_label.pack(pady=5)
        
    def create_middle_panel(self, parent):
        """Create the middle panel for problem configuration"""
        middle_frame = ttk.LabelFrame(parent, text="Problem Configuration", padding=10)
        middle_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Algorithm selection
        ttk.Label(middle_frame, text="Algorithm:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.algorithm_var = tk.StringVar(value="bfs")
        algorithm_combo = ttk.Combobox(middle_frame, textvariable=self.algorithm_var, 
                                       values=["bfs", "dfs", "astar", "bidirectional"], 
                                       state="readonly", width=15)
        algorithm_combo.grid(row=0, column=1, pady=5, padx=5)
        
        # Jug configuration
        ttk.Label(middle_frame, text="Jug Sizes:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        
        # Jug size inputs
        jug_frame = ttk.Frame(middle_frame)
        jug_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        self.jug_entries = []
        for i in range(3):
            ttk.Label(jug_frame, text=f"Jug {i+1}:").grid(row=i, column=0, sticky="w", padx=5)
            entry = ttk.Entry(jug_frame, width=10)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.jug_entries.append(entry)
        
        # Initial state
        ttk.Label(middle_frame, text="Initial State:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w", pady=5)
        
        initial_frame = ttk.Frame(middle_frame)
        initial_frame.grid(row=4, column=0, columnspan=2, pady=5)
        
        self.initial_entries = []
        for i in range(3):
            ttk.Label(initial_frame, text=f"Initial {i+1}:").grid(row=i, column=0, sticky="w", padx=5)
            entry = ttk.Entry(initial_frame, width=10)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.initial_entries.append(entry)
        
        # Target
        ttk.Label(middle_frame, text="Target:", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky="w", pady=5)
        self.target_entry = ttk.Entry(middle_frame, width=10)
        self.target_entry.grid(row=5, column=1, pady=5, padx=5)
        
        # Source and Sink options
        self.source_var = tk.BooleanVar(value=True)
        self.sink_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(middle_frame, text="Has Source", variable=self.source_var).grid(row=6, column=0, pady=5)
        ttk.Checkbutton(middle_frame, text="Has Sink", variable=self.sink_var).grid(row=6, column=1, pady=5)
        
        # Solve button
        solve_button = ttk.Button(middle_frame, text="🔍 Solve Problem", command=self.solve_problem)
        solve_button.grid(row=7, column=0, columnspan=2, pady=20)
        
        # Compare algorithms button
        compare_button = ttk.Button(middle_frame, text="📊 Compare All Algorithms", command=self.compare_algorithms)
        compare_button.grid(row=8, column=0, columnspan=2, pady=5)
        
        # Results area
        ttk.Label(middle_frame, text="Solution Summary:", font=("Arial", 10, "bold")).grid(row=9, column=0, sticky="w", pady=5)
        
        self.result_text = scrolledtext.ScrolledText(middle_frame, height=10, width=35, wrap=tk.WORD)
        self.result_text.grid(row=10, column=0, columnspan=2, pady=5, sticky="nsew")
        
        # Configure grid weights
        middle_frame.grid_rowconfigure(10, weight=1)
        
    def create_right_panel(self, parent):
        """Create the right panel for search tree and statistics"""
        right_frame = ttk.LabelFrame(parent, text="Search Tree & Statistics", padding=10)
        right_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tree tab
        tree_frame = ttk.Frame(notebook)
        notebook.add(tree_frame, text="Search Tree")
        
        # Create Treeview for search tree
        self.tree_view = ttk.Treeview(tree_frame, columns=("State", "Parent", "Action"), show="tree headings")
        self.tree_view.heading("#0", text="ID")
        self.tree_view.heading("State", text="State")
        self.tree_view.heading("Parent", text="Parent")
        self.tree_view.heading("Action", text="Action")
        
        self.tree_view.column("#0", width=50)
        self.tree_view.column("State", width=100)
        self.tree_view.column("Parent", width=50)
        self.tree_view.column("Action", width=150)
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree_view.yview)
        self.tree_view.configure(yscrollcommand=tree_scroll.set)
        
        self.tree_view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=20, width=40, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Path tab
        path_frame = ttk.Frame(notebook)
        notebook.add(path_frame, text="Solution Path")
        
        self.path_listbox = tk.Listbox(path_frame, font=("Courier", 10))
        path_scroll = ttk.Scrollbar(path_frame, orient="vertical", command=self.path_listbox.yview)
        self.path_listbox.configure(yscrollcommand=path_scroll.set)
        
        self.path_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        path_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure grid weights
        parent.grid_columnconfigure(0, weight=2)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(2, weight=2)
        parent.grid_rowconfigure(0, weight=1)
        
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_default_problem(self):
        """Load the default water jug problem"""
        # Set default values
        sizes = [8, 5, 3]
        initial = [0, 0, 0]
        target = 4
        
        # Fill in the entries
        for i, (size, init) in enumerate(zip(sizes, initial)):
            self.jug_entries[i].delete(0, tk.END)
            self.jug_entries[i].insert(0, str(size))
            self.initial_entries[i].delete(0, tk.END)
            self.initial_entries[i].insert(0, str(init))
        
        self.target_entry.delete(0, tk.END)
        self.target_entry.insert(0, str(target))
        
        # Draw initial state
        self.draw_jugs(initial, sizes)
        
    def draw_jugs(self, state, sizes):
        """Draw the water jugs on canvas"""
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not yet rendered
            canvas_width = 600
            canvas_height = 400
        
        num_jugs = len(sizes)
        jug_width = 80
        jug_spacing = 40
        total_width = num_jugs * jug_width + (num_jugs - 1) * jug_spacing
        start_x = (canvas_width - total_width) // 2
        
        max_height = 250
        base_y = canvas_height - 50
        
        for i, (current, capacity) in enumerate(zip(state, sizes)):
            x = start_x + i * (jug_width + jug_spacing)
            
            # Calculate jug height based on capacity
            jug_height = int((capacity / max(sizes)) * max_height)
            
            # Draw jug outline
            self.canvas.create_rectangle(
                x, base_y - jug_height, x + jug_width, base_y,
                outline=self.colors['jug_outline'], width=3, fill=self.colors['jug']
            )
            
            # Draw water
            if current > 0:
                water_height = int((current / capacity) * jug_height)
                self.canvas.create_rectangle(
                    x + 3, base_y - water_height, x + jug_width - 3, base_y - 3,
                    fill=self.colors['water'], outline=""
                )
            
            # Draw labels
            self.canvas.create_text(
                x + jug_width // 2, base_y + 20,
                text=f"Jug {i+1}", font=("Arial", 10, "bold")
            )
            self.canvas.create_text(
                x + jug_width // 2, base_y - jug_height - 10,
                text=f"{current}/{capacity}", font=("Arial", 12)
            )
            
            # Draw graduation marks
            for j in range(1, capacity + 1):
                mark_y = base_y - int((j / capacity) * jug_height)
                self.canvas.create_line(
                    x, mark_y, x + 10, mark_y,
                    fill=self.colors['jug_outline'], width=1
                )
                self.canvas.create_text(
                    x - 10, mark_y, text=str(j), font=("Arial", 8), anchor="e"
                )
        
        # Draw target line
        if self.current_problem and 'target' in self.current_problem:
            target = self.current_problem['target']
            self.canvas.create_text(
                canvas_width // 2, 30,
                text=f"Target: {target} liters in any jug",
                font=("Arial", 14, "bold"), fill=self.colors['current']
            )
    
    def solve_problem(self):
        """Solve the current problem with selected algorithm"""
        try:
            # Get problem parameters
            sizes = [int(entry.get()) for entry in self.jug_entries if entry.get()]
            initial = [int(entry.get()) for entry in self.initial_entries if entry.get()]
            target = int(self.target_entry.get())
            
            if len(sizes) != len(initial):
                messagebox.showerror("Error", "Number of jugs and initial states must match")
                return
            
            # Create problem dictionary
            self.current_problem = {
                "size": sizes,
                "filled": initial,
                "source": self.source_var.get(),
                "sink": self.sink_var.get(),
                "target": target
            }
            
            # Update status
            self.status_bar.config(text=f"Solving with {self.algorithm_var.get().upper()}...")
            self.root.update()
            
            # Solve problem
            player = Player()
            player.set_algorithm(self.algorithm_var.get())
            
            start_time = time.time()
            path, tree = player.run(self.current_problem)
            solve_time = time.time() - start_time
            
            if path:
                self.current_solution = path
                self.current_tree = tree
                self.current_step = 0
                
                # Update UI
                self.display_solution(path, tree, solve_time)
                self.populate_search_tree(tree)
                self.update_statistics(tree, path, solve_time)
                
                # Enable playback controls
                self.play_button.config(state=tk.NORMAL)
                self.step_button.config(state=tk.NORMAL)
                self.reset_button.config(state=tk.NORMAL)
                
                self.status_bar.config(text=f"Solution found in {len(path)-1} steps ({solve_time:.3f}s)")
            else:
                messagebox.showinfo("No Solution", "No solution found for this problem")
                self.status_bar.config(text="No solution found")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
            self.status_bar.config(text="Error: Invalid input")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_bar.config(text="Error occurred")
    
    def display_solution(self, path, tree, solve_time):
        """Display solution in the result text area"""
        self.result_text.delete(1.0, tk.END)
        
        result = f"Algorithm: {self.algorithm_var.get().upper()}\n"
        result += f"Solution found in {len(path)-1} steps\n"
        result += f"States explored: {len(tree)}\n"
        result += f"Time: {solve_time:.3f} seconds\n"
        result += "-" * 30 + "\n\n"
        
        # Add solution path to listbox
        self.path_listbox.delete(0, tk.END)
        
        for i, state in enumerate(path):
            if i == 0:
                step_text = f"Initial: {state}"
            else:
                step_text = f"Step {i}: {state}"
            
            self.path_listbox.insert(tk.END, step_text)
            
            if i < len(path) - 1:
                result += f"Step {i}: {state}\n"
        
        result += f"\nGoal: {path[-1]}"
        self.result_text.insert(1.0, result)
        
        # Update step label
        self.step_label.config(text=f"Step: 0 / {len(path)-1}")
    
    def populate_search_tree(self, tree):
        """Populate the search tree view"""
        # Clear existing tree
        for item in self.tree_view.get_children():
            self.tree_view.delete(item)
        
        if not tree:
            return
        
        # Add nodes to tree (limit to first 100 for performance)
        for i, node in enumerate(tree[:100]):
            node_id = str(node.get('id', i))
            state = str(node.get('state', []))
            parent = str(node.get('parent', ''))
            actions = node.get('actions', [])
            action = actions[-1] if actions else ''
            
            # Insert into treeview
            self.tree_view.insert('', 'end', text=node_id, 
                                 values=(state, parent, action))
        
        if len(tree) > 100:
            self.tree_view.insert('', 'end', text='...', 
                                 values=(f'{len(tree)-100} more nodes', '', ''))
    
    def update_statistics(self, tree, path, solve_time):
        """Update statistics display"""
        self.stats_text.delete(1.0, tk.END)
        
        stats = f"{'='*30}\n"
        stats += f"ALGORITHM STATISTICS\n"
        stats += f"{'='*30}\n\n"
        
        stats += f"Algorithm: {self.algorithm_var.get().upper()}\n"
        stats += f"Solution Length: {len(path)-1} steps\n"
        stats += f"Nodes Explored: {len(tree)}\n"
        stats += f"Solve Time: {solve_time:.3f} seconds\n"
        stats += f"Nodes/Second: {len(tree)/solve_time:.1f}\n\n"
        
        # Calculate branching factor
        if tree:
            children_counts = [len(node.get('children', [])) for node in tree]
            avg_branching = sum(children_counts) / len(children_counts) if children_counts else 0
            stats += f"Avg Branching Factor: {avg_branching:.2f}\n"
        
        # Memory estimate
        memory_kb = len(str(tree)) / 1024
        stats += f"Memory Used: ~{memory_kb:.2f} KB\n\n"
        
        # Optimality check
        if self.algorithm_var.get() in ['bfs', 'astar']:
            stats += "✓ Optimal solution guaranteed\n"
        else:
            stats += "⚠ Solution may not be optimal\n"
        
        self.stats_text.insert(1.0, stats)
    
    def play_solution(self):
        """Animate the solution"""
        if not self.current_solution:
            return
        
        self.is_animating = True
        self.play_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.step_button.config(state=tk.DISABLED)
        
        def animate():
            while self.is_animating and self.current_step < len(self.current_solution):
                self.show_step(self.current_step)
                self.current_step += 1
                
                if self.current_step >= len(self.current_solution):
                    self.is_animating = False
                    break
                
                time.sleep(self.animation_speed.get() / 1000.0)
            
            self.play_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.step_button.config(state=tk.NORMAL)
        
        self.animation_thread = threading.Thread(target=animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()
    
    def pause_animation(self):
        """Pause the animation"""
        self.is_animating = False
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.step_button.config(state=tk.NORMAL)
    
    def reset_visualization(self):
        """Reset to initial state"""
        self.current_step = 0
        self.is_animating = False
        
        if self.current_solution:
            self.show_step(0)
            self.play_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.step_button.config(state=tk.NORMAL)
    
    def step_forward(self):
        """Move one step forward"""
        if self.current_solution and self.current_step < len(self.current_solution) - 1:
            self.current_step += 1
            self.show_step(self.current_step)
    
    def show_step(self, step):
        """Show a specific step of the solution"""
        if not self.current_solution or step >= len(self.current_solution):
            return
        
        state = self.current_solution[step]
        sizes = self.current_problem['size']
        
        # Update canvas
        self.draw_jugs(state, sizes)
        
        # Highlight current step in path listbox
        self.path_listbox.selection_clear(0, tk.END)
        self.path_listbox.selection_set(step)
        self.path_listbox.see(step)
        
        # Update step label
        self.step_label.config(text=f"Step: {step} / {len(self.current_solution)-1}")
        
        # Check if goal reached
        if any(bucket == self.current_problem['target'] for bucket in state):
            # Draw success indicator
            canvas_width = self.canvas.winfo_width()
            self.canvas.create_text(
                canvas_width // 2, 60,
                text="✓ GOAL REACHED!", 
                font=("Arial", 16, "bold"), 
                fill=self.colors['success']
            )
    
    def compare_algorithms(self):
        """Compare all algorithms on the current problem"""
        if not all(entry.get() for entry in self.jug_entries):
            messagebox.showerror("Error", "Please configure the problem first")
            return
        
        try:
            # Get problem parameters
            sizes = [int(entry.get()) for entry in self.jug_entries if entry.get()]
            initial = [int(entry.get()) for entry in self.initial_entries if entry.get()]
            target = int(self.target_entry.get())
            
            problem = {
                "size": sizes,
                "filled": initial,
                "source": self.source_var.get(),
                "sink": self.sink_var.get(),
                "target": target
            }
            
            # Compare window
            compare_window = tk.Toplevel(self.root)
            compare_window.title("Algorithm Comparison")
            compare_window.geometry("800x600")
            
            # Results text
            results_text = scrolledtext.ScrolledText(compare_window, wrap=tk.WORD, font=("Courier", 10))
            results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            results = "ALGORITHM COMPARISON RESULTS\n"
            results += "=" * 60 + "\n"
            results += f"Problem: Jugs {sizes}, Target: {target}\n"
            results += "=" * 60 + "\n\n"
            
            algorithms = ["bfs", "dfs", "astar", "bidirectional"]
            comparison_data = []
            
            for algo in algorithms:
                player = Player()
                player.set_algorithm(algo)
                
                start_time = time.time()
                path, tree = player.run(problem)
                solve_time = time.time() - start_time
                
                if path:
                    comparison_data.append({
                        'algorithm': algo.upper(),
                        'steps': len(path) - 1,
                        'nodes': len(tree),
                        'time': solve_time,
                        'optimal': algo in ['bfs', 'astar']
                    })
                    
                    results += f"{algo.upper()}:\n"
                    results += f"  Solution Steps: {len(path)-1}\n"
                    results += f"  Nodes Explored: {len(tree)}\n"
                    results += f"  Time: {solve_time:.4f} seconds\n"
                    results += f"  Optimal: {'Yes' if algo in ['bfs', 'astar'] else 'Not guaranteed'}\n\n"
                else:
                    results += f"{algo.upper()}: No solution found\n\n"
            
            # Find best performers
            if comparison_data:
                results += "-" * 60 + "\n"
                results += "ANALYSIS:\n"
                results += "-" * 60 + "\n\n"
                
                # Fastest
                fastest = min(comparison_data, key=lambda x: x['time'])
                results += f"⚡ Fastest: {fastest['algorithm']} ({fastest['time']:.4f}s)\n"
                
                # Most efficient (fewest nodes)
                efficient = min(comparison_data, key=lambda x: x['nodes'])
                results += f"🎯 Most Efficient: {efficient['algorithm']} ({efficient['nodes']} nodes)\n"
                
                # Shortest solution
                shortest = min(comparison_data, key=lambda x: x['steps'])
                results += f"📏 Shortest Path: {shortest['algorithm']} ({shortest['steps']} steps)\n"
            
            results_text.insert(1.0, results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {str(e)}")
    
    def new_problem(self):
        """Clear and create new problem"""
        for entry in self.jug_entries + self.initial_entries:
            entry.delete(0, tk.END)
        self.target_entry.delete(0, tk.END)
        
        self.current_solution = None
        self.current_tree = None
        self.current_step = 0
        
        self.canvas.delete("all")
        self.result_text.delete(1.0, tk.END)
        self.stats_text.delete(1.0, tk.END)
        self.path_listbox.delete(0, tk.END)
        
        for item in self.tree_view.get_children():
            self.tree_view.delete(item)
        
        self.status_bar.config(text="Ready for new problem")
    
    def export_solution(self):
        """Export current solution to file"""
        if not self.current_solution:
            messagebox.showwarning("No Solution", "No solution to export")
            return
        
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            export_data = {
                'problem': self.current_problem,
                'algorithm': self.algorithm_var.get(),
                'solution': self.current_solution,
                'tree_size': len(self.current_tree) if self.current_tree else 0
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Export Complete", f"Solution exported to {filename}")
    
    def show_search_tree(self):
        """Show detailed search tree in new window"""
        if not self.current_tree:
            messagebox.showinfo("No Tree", "No search tree to display. Solve a problem first.")
            return
        
        tree_window = tk.Toplevel(self.root)
        tree_window.title("Search Tree Details")
        tree_window.geometry("900x600")
        
        tree_text = scrolledtext.ScrolledText(tree_window, wrap=tk.WORD, font=("Courier", 9))
        tree_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tree_json = json.dumps(self.current_tree[:50], indent=2)  # Limit to first 50 nodes
        tree_text.insert(1.0, tree_json)
        
        if len(self.current_tree) > 50:
            tree_text.insert(tk.END, f"\n\n... and {len(self.current_tree)-50} more nodes")
    
    def show_statistics(self):
        """Show detailed statistics"""
        if not self.current_tree:
            messagebox.showinfo("No Data", "No data to analyze. Solve a problem first.")
            return
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Detailed Statistics")
        stats_window.geometry("600x400")
        
        # Create simple bar chart using Canvas
        canvas = tk.Canvas(stats_window, bg='white')
        canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # This would be where you'd add matplotlib integration for better charts
        # For now, using simple canvas drawing
        
    def show_algorithm_help(self):
        """Show help about algorithms"""
        help_text = """
SEARCH ALGORITHMS GUIDE

BFS (Breadth-First Search):
• Explores level by level
• Guarantees optimal solution
• Memory intensive
• Best for: Finding shortest path

DFS (Depth-First Search):
• Explores deeply before backtracking
• Memory efficient
• May find longer paths
• Best for: Quick solutions

A* Search:
• Uses heuristic to guide search
• Guarantees optimal solution
• More efficient than BFS
• Best for: Balanced performance

Bidirectional Search:
• Searches from both start and goal
• Meets in the middle
• Reduces search space
• Best for: Deep solutions
        """
        
        messagebox.showinfo("Algorithm Help", help_text)
    
    def show_usage_help(self):
        """Show usage help"""
        help_text = """
HOW TO USE:

1. Configure Problem:
   • Enter jug capacities
   • Set initial water levels
   • Specify target amount

2. Select Algorithm:
   • Choose from dropdown menu
   • Each has different strengths

3. Solve:
   • Click "Solve Problem"
   • View solution path

4. Visualize:
   • Use Play to animate
   • Step through manually
   • Adjust animation speed

5. Analyze:
   • View search tree
   • Check statistics
   • Compare algorithms

Tips:
• Try different algorithms
• Smaller jugs = faster solve
• Target should be achievable
        """
        
        messagebox.showinfo("Usage Help", help_text)


def main():
    root = tk.Tk()
    app = WaterJugGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()