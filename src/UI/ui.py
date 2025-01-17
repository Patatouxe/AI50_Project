#@Author : tmayer
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import time
import csv
import matplotlib.pyplot as plt
from src.CVRP import CVRP, Route
from src.Saving_Algo.saving_class import Savings
from src.ACO_MACS.aco_colony import MACS_CVRP
from src.Gen_Algo.AG import GeneticAlgorithmCVRP
from src.ACO.aco import Colony
    
class CVRPApp:
    """
    A GUI application for solving the Capacitated Vehicle Routing Problem (CVRP) using various algorithms.
    Allows users to select data files, choose algorithms, view results, and analyze performance.

    Attributes:
        root (tk.Tk): The root Tkinter window.
        files (List[str]): List of selected CVRP data files.
        selected_algorithms (Dict[str, tk.BooleanVar]): Selected algorithms for execution.
        solution_details (Dict): Stores solution details for each file and algorithm.
    """
    def __init__(self, root):
        """
        Initialize the CVRPApp GUI with necessary components.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title("CVRP Solver GUI")
        self.files = []
        self.selected_algorithms = {}
        self.solution_details = {}

        # Configure root grid
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Create a main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.rowconfigure(4, weight=1)  # Allow results to expand
        self.main_frame.columnconfigure(0, weight=1)

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        """
        Create and layout all GUI components for the application.
        """
        # Header
        header = tk.Label(self.main_frame, text="CVRP Solver GUI", font=("Arial", 18, "bold"), bg="#003366", fg="white")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        # File selection frame
        file_frame = tk.LabelFrame(self.main_frame, text="Select Data Files", font=("Arial", 12))
        file_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        file_frame.columnconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(file_frame, selectmode=tk.MULTIPLE, height=5, font=("Arial", 10))
        self.file_listbox.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        file_buttons_frame = tk.Frame(file_frame)
        file_buttons_frame.grid(row=1, column=0, sticky="ew", pady=5)
        file_buttons_frame.columnconfigure([0, 1], weight=1)

        tk.Button(file_buttons_frame, text="Add Files", command=self.add_files, bg="#007acc", fg="white").grid(row=0, column=0, sticky="ew", padx=5)
        tk.Button(file_buttons_frame, text="Remove Selected Files", command=self.remove_selected_files, bg="#007acc", fg="white").grid(row=0, column=1, sticky="ew", padx=5)

        # Algorithm selection frame
        algo_frame = tk.LabelFrame(self.main_frame, text="Select Algorithms to Run", font=("Arial", 12))
        algo_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        self.selected_algorithms = {
            "Savings": tk.BooleanVar(),
            "MACS": tk.BooleanVar(),
            "Genetic Algorithm": tk.BooleanVar(),
            "Classical ACO": tk.BooleanVar()
        }
        for i, (algo, var) in enumerate(self.selected_algorithms.items()):
            tk.Checkbutton(algo_frame, text=algo, variable=var, font=("Arial", 10)).grid(row=i, column=0, sticky="w", padx=10)

        # Run button
        tk.Button(self.main_frame, text="Run Selected Algorithms", command=self.run_algorithms, bg="#28a745", fg="white").grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        # Results filtering frame
        filter_frame = tk.LabelFrame(self.main_frame, text="Select Data to View Results", font=("Arial", 12))
        filter_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        filter_frame.columnconfigure(0, weight=1)

        self.data_filter_combo = ttk.Combobox(filter_frame, state="readonly")
        self.data_filter_combo.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Results table
        results_frame = tk.LabelFrame(self.main_frame, text="Results", font=("Arial", 12))
        results_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(results_frame, columns=("File", "Algorithm", "Distance", "Time"), show="headings")
        self.results_tree.heading("File", text="File")
        self.results_tree.heading("Algorithm", text="Algorithm")
        self.results_tree.heading("Distance", text="Distance")
        self.results_tree.heading("Time", text="Time (s)")
        self.results_tree.bind("<Double-1>", self.view_details)
        self.results_tree.bind("<Motion>", self.on_mouse_over_tree)
        self.results_tree.bind("<Leave>", self.on_mouse_leave_tree)
        self.results_tree.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Buttons for analysis and export
        buttons_frame = tk.Frame(self.main_frame)
        buttons_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=10)
        buttons_frame.columnconfigure([0, 1], weight=1)

        tk.Button(buttons_frame, text="Comparison Analysis", command=self.comparison_analysis, bg="#007acc", fg="white").grid(row=0, column=0, sticky="ew", padx=10)
        tk.Button(buttons_frame, text="Export to CSV", command=self.export_to_csv, bg="#ffc107", fg="black").grid(row=0, column=1, sticky="ew", padx=10)
   
    def add_files(self):
        """
        Add selected data files to the list.
        """
        selected_files = filedialog.askopenfilenames(title="Select Data Files")
        for file in selected_files:
            if file not in self.files:
                self.files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))

    def remove_selected_files(self):
        """
        Remove the selected files from the list.
        """
        selected_indices = self.file_listbox.curselection()
        for index in reversed(selected_indices):
            self.file_listbox.delete(index)
            del self.files[index]

    def on_mouse_over(self, event):
        widget = event.widget
        try:
            index = widget.nearest(event.y)
            widget.itemconfig(index, background="#d3d3d3")
        except Exception:
            pass

    def on_mouse_leave(self, event):
        widget = event.widget
        for index in range(widget.size()):
            widget.itemconfig(index, background="white")

    def on_mouse_over_tree(self, event):
        region = self.results_tree.identify('region', event.x, event.y)
        if region == 'cell':
            row_id = self.results_tree.identify_row(event.y)
            self.results_tree.tag_configure(row_id, background="#d3d3d3")

    def on_mouse_leave_tree(self, event):
        for row in self.results_tree.get_children():
            self.results_tree.tag_configure(row, background="white")

    def run_algorithms(self):
        """
        Run the selected algorithms on the chosen files and display results.
        """
        if not self.files:
            messagebox.showerror("Error", "No files selected!")
            return

        if not any(var.get() for var in self.selected_algorithms.values()):
            messagebox.showerror("Error", "No algorithms selected!")
            return

        # Clear previous results
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)

        self.solution_details = {}

        for file in self.files:
            try:
                cvrp_instance = CVRP(file)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file {file}: {str(e)}")
                continue

            # Run selected algorithms
            if self.selected_algorithms["Savings"].get():
                self.run_savings(cvrp_instance, file)

            if self.selected_algorithms["MACS"].get():
                self.run_aco(cvrp_instance, file)

            if self.selected_algorithms["Genetic Algorithm"].get():
                self.run_genetic(cvrp_instance, file)

            if self.selected_algorithms["Classical ACO"].get():
                self.run_classical_aco(cvrp_instance, file)

        # Populate the data filter combo
        self.data_filter_combo["values"] = [os.path.basename(f) for f in self.files]
        if self.files:
            self.data_filter_combo.current(0)
            self.filter_results()

    def run_savings(self, cvrp_instance, file):
        """
        Run the Savings algorithm on the given CVRP instance and record the results.

        Args:
            cvrp_instance (CVRP): The CVRP instance to solve.
            file (str): The file name associated with the CVRP instance.
        """
        start_time = time.time()
        savings_solver = Savings(cvrp_instance)
        try:
            solution, distance = savings_solver.run()
            end_time = time.time()
            self.add_result(file, "Savings", distance, end_time - start_time, solution)
        except Exception as e:
            messagebox.showerror("Error", f"Error running Savings algorithm on {file}: {str(e)}")

    def run_aco(self, cvrp_instance, file):
        """
        Run the MACS (Ant Colony Optimization) algorithm on the given CVRP instance.

        Args:
            cvrp_instance (CVRP): The CVRP instance to solve.
            file (str): The file name associated with the CVRP instance.
        """
        start_time = time.time()
        macs_solver = MACS_CVRP(cvrp_instance)
        try:
            solution, distance, _ = macs_solver.run(200)
            end_time = time.time()
            self.add_result(file, "MACS", distance, end_time - start_time, solution)
        except Exception as e:
            messagebox.showerror("Error", f"Error running MACS algorithm on {file}: {str(e)}")

    def run_genetic(self, cvrp_instance, file):
        """
        Run the Genetic Algorithm on the given CVRP instance.

        Args:
            cvrp_instance (CVRP): The CVRP instance to solve.
            file (str): The file name associated with the CVRP instance.
        """
        start_time = time.time()
        ga_solver = GeneticAlgorithmCVRP(cvrp_instance, generations=200, population_size=100, mutation_rate=0.05)
        try:
            solution, distance = ga_solver.run()
            end_time = time.time()
            self.add_result(file, "Genetic Algorithm", distance, end_time - start_time, solution)
        except Exception as e:
            messagebox.showerror("Error", f"Error running Genetic Algorithm on {file}: {str(e)}")

    def run_classical_aco(self, cvrp_instance, file):
        """
        Run the Classical Ant Colony Optimization (ACO) algorithm on the given CVRP instance.

        Args:
            cvrp_instance (CVRP): The CVRP instance to solve.
            file (str): The file name associated with the CVRP instance.
        """
        start_time = time.time()
        colony = Colony(cvrp_instance, nbrIter=200, alpha=1.0, beta=2.0, gamma=1.0, evapRate=0.5, theta=0.1, Q=100)
        try:
            solution, distance = colony.solve()
            end_time = time.time()
            self.add_result(file, "Classical ACO", distance, end_time - start_time, solution)
        except Exception as e:
            messagebox.showerror("Error", f"Error running Classical ACO algorithm on {file}: {str(e)}")

    def export_to_csv(self):
        """
        Export the displayed results to a CSV file.
        """
        if not self.results_tree.get_children():
            messagebox.showerror("Error", "No results to export!")
            return

        # Open a file dialog to choose the export location
        file_path = filedialog.asksaveasfilename(
            title="Save Results as CSV",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not file_path:
            return  # User cancelled the save dialog

        # Extract data from the results_tree
        data = []
        for row_id in self.results_tree.get_children():
            row_data = self.results_tree.item(row_id)["values"]
            data.append(row_data)

        # Write the data to the CSV file
        try:
            with open(file_path, mode="w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                # Write headers
                writer.writerow(["File", "Algorithm", "Distance", "Time (s)"])
                # Write rows
                writer.writerows(data)
            messagebox.showinfo("Success", f"Results successfully exported to {file_path}!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def add_result(self, file, algorithm, distance, execution_time, solution):
        """
        Add a result entry to the results table and store details for analysis.

        Args:
            file (str): The file name of the CVRP instance.
            algorithm (str): The name of the algorithm used.
            distance (float): The total distance of the solution.
            execution_time (float): The execution time in seconds.
            solution (List[Route]): The solution routes.
        """
        file_basename = os.path.basename(file)
        self.results_tree.insert(
            "", tk.END,
            values=(file_basename, algorithm, f"{distance:.2f}", f"{execution_time:.2f}")
        )
        if file_basename not in self.solution_details:
            self.solution_details[file_basename] = {}
        self.solution_details[file_basename][algorithm] = {
            "routes": solution,
            "execution_time": execution_time,
        }

    def filter_results(self, event=None):
        """
        Filter and display results for the selected file in the dropdown menu.

        Args:
            event: Optional event parameter for binding to UI interactions.
        """
        selected_file = self.data_filter_combo.get()
        if not selected_file:
            return

        # Clear the results tree
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)

        # Add results for the selected file
        if selected_file in self.solution_details:
            for algorithm, data in self.solution_details[selected_file].items():
                routes = data["routes"]
                execution_time = data["execution_time"]

                # Ensure routes are Route objects and sum their distances
                distance = sum(route.distance for route in routes if isinstance(route, Route))

                self.results_tree.insert(
                    "", tk.END,
                    values=(selected_file, algorithm, f"{distance:.2f}", f"{execution_time:.2f}")
                )

    def view_details(self, event):
        """
        Display detailed solution routes in a separate window.

        Args:
            event: The event triggered by double-clicking a result.
        """
        selected_item = self.results_tree.selection()
        if not selected_item:
            return

        selected_values = self.results_tree.item(selected_item[0], "values")
        file, algorithm = selected_values[0], selected_values[1]

        if file in self.solution_details and algorithm in self.solution_details[file]:
            solution = self.solution_details[file][algorithm]["routes"]
            solution_str = "\n".join([f"Route {i+1}: {route.customers} (Distance: {route.distance:.2f}, Capacity : {route.capacity})" for i, route in enumerate(solution)])

            details_window = tk.Toplevel(self.root)
            details_window.title(f"Solution Details - {file} - {algorithm}")
            tk.Label(details_window, text=f"Solution Details for {algorithm} on {file}", font=("Arial", 12, "bold"), bg="#003366", fg="white").pack(fill="x")
            text_widget = tk.Text(details_window, wrap="word", width=80, height=20, font=("Arial", 10))
            text_widget.insert("1.0", solution_str)
            text_widget.config(state="disabled")
            text_widget.pack(padx=10, pady=10)

    def comparison_analysis(self):
        """
        Perform a comparative analysis of algorithms for the selected file, visualizing distance and time metrics.
        """
        selected_file = self.data_filter_combo.get()
        if not selected_file or selected_file not in self.solution_details:
            messagebox.showerror("Error", "No data available for comparison analysis!")
            return

        algorithms = []
        distances = []
        times = []

        for algorithm, data in self.solution_details[selected_file].items():
            routes = data["routes"]
            execution_time = data["execution_time"]
            distance = sum(route.distance for route in routes if isinstance(route, Route))

            algorithms.append(algorithm)
            distances.append(distance)
            times.append(execution_time)

        # Plot comparison charts
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        # Distance comparison
        axs[0].bar(algorithms, distances, color="skyblue")
        axs[0].set_title("Algorithm Comparison by Distance")
        axs[0].set_ylabel("Distance")

        # Time comparison
        axs[1].bar(algorithms, times, color="salmon")
        axs[1].set_title("Algorithm Comparison by Execution Time")
        axs[1].set_ylabel("Time (s)")

        plt.tight_layout()
        plt.show()