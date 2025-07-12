import os
import subprocess
import ltspice
import numpy as np
from glob import glob
import itertools
import csv
import random
import math
import time as timer
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

'''
Class to run netlists and analyze the output
'''

class Run_Netlists:
    
    def __init__(self, n, dir):
        # EDIT: Path to LTspice executable
        self.LTSPICE_EXE = r"C:\Users\stanm\AppData\Local\Programs\ADI\LTspice\ltspice.exe"
        self.VTHRESH = 4.0          # Threshold voltage
        self.HOLD_TIME = 50e-6      # Time to hold above threshold (50us)
        self.n = n
        self.dir = dir
        self.output_nodes = [f"V(x{i:02d}_{j:02d})" for i in range(1, n + 1) for j in range(1, n + 1)]

    """
    Run LTspice on the given .cir file
    """
    def run_ltspice(self, netlist_path):
        subprocess.run([self.LTSPICE_EXE, "-Run", "-b", netlist_path], check=True)

    """
    Load LTspice raw file and return time vector and output voltages
    """
    def parse_raw_file(self, raw_path):
        l = ltspice.Ltspice(raw_path)
        l.parse()
        time = l.get_time()
        voltages = {node: l.get_data(node) for node in self.output_nodes}
        return time, voltages
    
    """
    Return the time when the nth signal crosses and stays above 0.95*Vthresh
    for at least hold_time duration. Also returns the names of nodes that satisfied this.
    """
    def find_steady_state_time(self, time, voltages, n_required=None):
        threshold_val = 0.95 * self.VTHRESH
        sample_interval = time[1] - time[0]
        min_samples = int(self.HOLD_TIME / sample_interval)

        node_names = list(voltages.keys())
        voltage_matrix = np.array([voltages[name] for name in node_names])  # shape: (n_signals, len(time))

        if n_required is None:
            n_required = int(len(node_names) ** 0.5)

        # For each time index, check if each signal stays above threshold for the next min_samples
        for t_idx in range(len(time) - min_samples):
            # Boolean mask of signals that are above threshold for entire duration window
            sustained = (voltage_matrix[:, t_idx:t_idx + min_samples] >= threshold_val).all(axis=1)

            if np.count_nonzero(sustained) >= n_required:
                steady_nodes = [node_names[i] for i, ok in enumerate(sustained) if ok]
                return ((time[t_idx]*1e6)-50), steady_nodes

        return None, []
    
    """
    Run, parse, and analyze one netlist
    """
    def analyze_netlist(self, netlist_path, subdir_name):
        base = os.path.splitext(netlist_path)[0]
        raw_path = base + ".raw"
        if (os.path.exists(os.path.join(os.path.dirname(netlist_path), f"generated_{self.n}_node.raw"))):
            print(f"\n--- {subdir_name} has already been analyzed... ---")
        else:
            print(f"Running simulation for: {netlist_path}")
            time_start = timer.time()
            self.run_ltspice(netlist_path)
            time_end = timer.time()
            print(f"### Simulation time: {(time_end-time_start):.4f} s ###")

        print(f"Parsing: {raw_path}")
        time, voltages = self.parse_raw_file(raw_path)

        print(f"Analyzing output...")
        t_steady, steady_nodes = self.find_steady_state_time(time, voltages)
        if t_steady is not None:
            print(f"--> Steady state reached at {t_steady:.2f} µs <--")
            print(f"Nodes above threshold: {steady_nodes}")
        else:
            print("--> No steady state reached <--")
        return t_steady, steady_nodes

    """
    Recursively analyzes all 'generated_n_node.cir' netlists in subfolders of self.dir,
    and writes a results CSV to 'nNode/results/results.csv'.
    """     
    def analyze_all_netlists(self):
        # Prepare output directory
        results_dir = os.path.join(self.dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_csv = os.path.join(results_dir, "results.csv")

        # Find all matching netlist paths
        netlist_paths = glob(os.path.join(self.dir, f"{self.n}Node*", f"generated_{self.n}_node.cir"))
        results = []
        
        for netlist_path in sorted(netlist_paths):
            subdir_name = os.path.basename(os.path.dirname(netlist_path))
            print(f"\n--- Analyzing {subdir_name} ---")

            try:
                time, nodes = self.analyze_netlist(netlist_path, subdir_name)
                time_us = time if time is not None else None
            except Exception as e:
                print(f"Error analyzing {netlist_path}: {e}")
                time_us = None
                nodes = []

            results.append({
                "Simulation": subdir_name,
                "Time_us": time_us,
                "Nodes": ";".join(nodes) if nodes else "None"
            })

        # Write to CSV
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Simulation", "Time_us", "Nodes"])
            writer.writeheader()
            writer.writerows(results)

            # Compute and append average time
            valid_times = [r["Time_us"] for r in results if r["Time_us"] is not None]
            avg_time = sum(valid_times) / len(valid_times) if valid_times else None

            writer.writerow({
                "Simulation": "Average",
                "Time_us": f"{avg_time:.2f}" if avg_time else "N/A",
                "Nodes": ""
            })

        print(f"\nResults written to: {results_csv}")

        self.analyze_solution_quality()
        
    def read_distance_matrix_from_csv(self, filename):
        with open(filename, newline='') as f:
            reader = list(csv.reader(f))
            n = int(reader[0][1])  # First row: ["nodes", n]
            matrix_rows = reader[9:9+n]
            matrix = []
            for row in matrix_rows:
                matrix.append([float(x) for x in row[1:]])
            avg_len = reader[2][1]
            min_len = reader[3][1]
            return matrix, float(avg_len), float(min_len)

    def compute_tour_length(self, tour, dist_matrix):
        return sum(dist_matrix[tour[i - 1]][tour[i]] for i in range(len(tour)))

    def parse_tour_from_nodes(self, nodes):
        tour_pairs = []

        for node in nodes:
            label = node[3:-1].split("_")
            v = int(label[0])-1  # city
            k = int(label[1])  # position
            tour_pairs.append((k, v))

        # Sort by position (k), then extract city (v)
        tour = [v for k, v in sorted(tour_pairs)]
        return tour
        
    def sample_tour_lengths(self, adj_matrix, return_samples, length=0):
        count = 0
        samples = 0
        if self.n > 9:
            samples = int(1e5)
        elif self.n > 9 and return_samples:
            samples = int(1e4)
        else:
            samples = int(math.ceil(math.factorial(self.n-1)/2))

        if return_samples:
            lengths = list()
            for _ in range(samples):
                tour = random.sample(range(self.n), self.n)
                lengths.append(self.compute_tour_length(tour, adj_matrix))
            return lengths

        else:
            for _ in range(samples):
                tour = random.sample(range(self.n), self.n)
                len = self.compute_tour_length(tour, adj_matrix)
                if len < length:
                    count += 1     
            return round( 1 - (count/samples), 2)

    """
    Analyze the quality of each analog TSP solution and save normalized tour costs.
    """
    def analyze_solution_quality(self):
        results_csv_path = self.dir + r"\results\results.csv"
        result_dir = os.path.dirname(results_csv_path)
        output_csv = os.path.join(result_dir, "quality_summary.csv")
        avg_ratios = []
        min_ratios = []
        len_percents = []

        print("\nAnalyzing solution quality...")

        with open(results_csv_path, newline="") as f_in, open(output_csv, "w", newline="") as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.writer(f_out)
            writer.writerow(["Instance", "Tour", "Tour_Length", "Average_Tour_Length", "Normalized_Cost_Against_Avg", "Minimum_Tour_Length", "Normalized_Cost_Against_Min", "%_of_tours_greater_than_tour"])

            for row in reader:
                instance = row["Simulation"]
                if instance == "Average" or row["Time_us"] == "None" or row["Nodes"] == "None":
                    continue
                nodes = f"{row["Nodes"]}".split(";")
                tour = self.parse_tour_from_nodes(nodes)

                tsp_csv_path = os.path.join(self.dir, instance, f"generated_{self.n}_node.csv")
                if not os.path.exists(tsp_csv_path):
                    print(f"TSP instance missing: {tsp_csv_path}")
                    continue

                distance_matrix, avg_len, min_len = self.read_distance_matrix_from_csv(tsp_csv_path)
                length = self.compute_tour_length(tour, distance_matrix)
                avg_ratio = length / avg_len
                min_ratio = length / min_len
                len_percent = self.sample_tour_lengths(distance_matrix, False, length)
                avg_ratios.append(avg_ratio)
                min_ratios.append(min_ratio)
                len_percents.append(len_percent)

                writer.writerow([instance, tour, f"{length:.2f}", f"{avg_len:.2f}", f"{avg_ratio:.4f}", f"{min_len:.2f}", f"{min_ratio:.4f}", f"{len_percent:.4f}"])

            if avg_ratios and min_ratios and len_percents:
                avg_ratio = sum(avg_ratios) / len(avg_ratios)
                min_ratio = sum(min_ratios) / len(min_ratios)
                len_ratio = sum(len_percents) / len(len_percents)
                writer.writerow(["Average", "", "", "", f"{avg_ratio:.4f}", "", f"{min_ratio:.4f}", f"{len_ratio:.4f}"])


    def read_results_csv(self, filename, results, quality):
        with open(filename, newline='') as f:
            reader = list(csv.reader(f))

            if results == True:
                return int(reader[1][0].split("Node")[0]), float(reader[-1][1])

            elif quality == True:
                return int(reader[1][0].split("Node")[0]), float(reader[-1][4]), float(reader[-1][6]), float(reader[-1][7])


    def plot_quality_data(self, data, line, path, ylabel, title):
        print(f"Generating {path} plot...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        x = np.array(list(data.keys()))
        y = np.array(list(data.values()))

        # Fit a line (1st-degree polynomial)
        if line:
            slope, intercept = np.polyfit(x, y, 1)
            best_fit = slope * x + intercept
            plt.plot(x, best_fit, color='red')

        plt.scatter(x, y)
        plt.xlabel("Nodes")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.clf()

    '''
    plot solution quality for all instances in root_dir
    or, plot solution quality for only nNode instances in root_dir
    '''
    def plot_soltuion_quality(self, root_dir, all=True):
        if all:
            quality_paths = glob(os.path.join(root_dir, f"[0-9]*Node", "results", "quality_summary.csv"))
        else:
            quality_paths = glob(os.path.join(root_dir, "results", "quality_summary.csv"))
        time_paths = glob(os.path.join(root_dir, f"[0-9]*Node", "results", "results.csv"))

        avg = dict()
        p_times_min = dict()
        length = dict()
        time = dict()

        for path in quality_paths:
            result = self.read_results_csv(path, False, True)
            avg[result[0]] = result[1]
            p_times_min[result[0]] = result[2]
            length[result[0]] = result[3]

        for path in time_paths:
            result = self.read_results_csv(path, True, False)
            time[result[0]] = result[1]

        self.plot_quality_data(avg, False, os.path.join(root_dir, "Plots", "AverageLength.png"), "Route Length of Solution / Avg Route Length", "(Route Length of Solution / Avg Route Length) vs. Nodes")
        self.plot_quality_data(p_times_min, False, os.path.join(root_dir, "Plots", "xTimesMinLength.png"), "Route Length of Solution / Min Route Length", "(Route Length of Solution / Min Route Length) vs. Nodes")
        self.plot_quality_data(length, False, os.path.join(root_dir, "Plots", "SolutionQuality.png"), f"% of Tours Longer Than Solution", f"% of Tours Longer Than Solution vs. Nodes")
        self.plot_quality_data(time, True, os.path.join(root_dir, "Plots", "Time.png"), "Solution search time [μs]", "Solution search time vs. Nodes")


    '''
    Plot a histogram of the tour lengths and approximate solution lengths
    for TSP instances in root_dir
    '''
    def plot_tour_length_histogram(self, root_dir):
        csv_paths = glob(os.path.join(root_dir, f"{self.n}Node*", f"generated_{self.n}_node.csv"))
        quality_csv_path = os.path.join(root_dir, "results", "quality_summary.csv")
        lengths = list()
        approx_lengths = list()

        with open(quality_csv_path, newline="") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                instance = row["Instance"]
                if instance == "Average":
                    continue
                approx_lengths.append(float(row["Tour_Length"]))

        for path in csv_paths:
            adj_matrix, avg_len, min_len = self.read_distance_matrix_from_csv(path)    
            lengths.extend(self.sample_tour_lengths(adj_matrix, True))

        min_approx_len = min(approx_lengths)
        max_approx_len = max(approx_lengths)
        avg_approx_len = sum(approx_lengths) / len(approx_lengths)

        count = 0
        for i in lengths:
            if i < avg_approx_len:
                count += 1
        pct_above_avg = 100*(1-(count/len(lengths)))

        count = 0
        for i in lengths:
            if i < max_approx_len:
                count += 1
        pct_above_max = 100*(1-(count/len(lengths)))

        bin_edges = np.histogram_bin_edges(lengths, "fd")
        plt.figure(figsize=(10, 5))
        counts, bins, patches = plt.hist(lengths, bins=bin_edges, color="blue", alpha=0.5)
        for bin_start, bin_end, patch in zip(bins[:-1], bins[1:], patches):
            if (bin_start > avg_approx_len) and (bin_start < max_approx_len):
                patch.set_facecolor('blue')
                patch.set_alpha(0.7)
            elif bin_start > max_approx_len:
                patch.set_facecolor('blue')
                patch.set_alpha(0.9)
        plt.text(plt.xlim()[1]*0.79, plt.ylim()[1]*0.7, f"{pct_above_avg:.1f}% Tours > Mean approx", color=to_rgba('blue', alpha=0.7), fontsize=10)
        plt.text(plt.xlim()[1]*0.79, plt.ylim()[1]*0.65, f"{pct_above_max:.1f}% Tours > Max approx", color=to_rgba('blue', alpha=0.9), fontsize=10)
        plt.axvline(min_approx_len, color='green', linestyle='--', linewidth=2, label=f'Min approx: {min_approx_len:.2f}')
        plt.axvline(avg_approx_len, color='purple', linestyle='-', linewidth=2, label=f'Mean approx: {avg_approx_len:.2f}')
        plt.axvline(max_approx_len, color='red', linestyle='--', linewidth=2, label=f'Max approx: {max_approx_len:.2f}')
        plt.title(f"{os.path.basename(root_dir)} TSP Instance Tour lengths")
        plt.xlabel("Tour Length")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        png_path = os.path.join(root_dir, "Plots", "TourLengthHistogram.png")
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.clf()