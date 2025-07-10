from collections import defaultdict
import numpy as np
import itertools
import random
import csv
import os

class TSP_Instance:

    '''
    Init TSP instance
    n: number of nodes in graph
    mean and variance define the normal distribution of edge weights
    '''
    def __init__(self, n, mean, var):
        self.n = n
        self.mean = mean
        self.var = var
        self.is_metric = True
        self.adj_matrix = np.zeros((self.n, self.n))
        self.non_metric_edges = defaultdict(list)

    '''
    Generate n 2D points with coordinates drawn from a normal distribution
    '''
    def generate_coordinates(self):
        return np.random.normal(loc=self.mean, scale=np.sqrt(self.var), size=(self.n, 2))
    
    '''
    Compute an nxn adjacency matrix using points drawn from 
    a normal distribution to ensure the graph is metric
    '''
    def compute_metric_adj_matrix(self, using_coords):
        self.adj_matrix = np.zeros((self.n, self.n))
        if using_coords:
            coords = self.generate_coordinates()
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    self.adj_matrix[i][j] = self.adj_matrix[j][i] = dist
            # graph from paper
            #self.adj_matrix = np.array([[0, 5, 65, 35], [5, 0, 35, 65], [65, 35, 0, 25], [35, 65, 25, 0]])
        else:
            violations = 1
            weights = np.random.normal(loc=self.mean, scale=np.sqrt(self.var), size=(self.n, 1))
            while violations > 0:
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        self.adj_matrix[i][j] = self.adj_matrix[j][i] = weights[j][0]
                violations = self.count_violations()
    
    '''
    Generate a metric TSP instance
    '''
    def gen_metric_TSP(self, using_coords):
        self.compute_metric_adj_matrix(using_coords)

    def existing_edges_in_triplet(self, i, j, k):
            edges = [(i, j), (j, k), (i, k)]
            return {e for e in edges if self.adj_matrix[e[0]][e[1]] > 0}

    '''
    DOES NOT WORK
    '''
    def gen_non_metric_TSP(self):
        self.is_metric = False
        broken_triangles = set()
        edges_to_triangles = defaultdict(list)

        # map edges to the triangles they belong to
        for a, b in itertools.combinations(range(self.n), 2):
            for x in range(self.n):
                if x != a and x!= b:
                    edges_to_triangles[(a,b)].append(tuple(sorted((a,b,x))))

        
        for a, b, c in itertools.combinations(range(self.n), 3):
            if (a,b,c) in broken_triangles:
                continue
            else:
                broken_triangles.add((a,b,c))
                edges_to_add = list({(a,b), (b, c), (a, c)} - self.existing_edges_in_triplet(a, b, c)) # edges that need to be added
                for edge in edges_to_add:
                    u, v = edge
                    if edge != edges_to_add[-1]:
                        self.adj_matrix[u][v] = self.adj_matrix[v][u] = np.random.normal(loc=self.mean, scale=np.sqrt(self.var))
                    else:
                        # break the triangle
                        #self.adj_matrix[u][v] = self.adj_matrix[v][u] = ( (2*self.mean*(1+self.var)) / (1-self.var) ) + (self.var*self.mean)
                        self.adj_matrix[u][v] = self.adj_matrix[v][u] = 500
                        # add all other triangles broken by u, v edge
                        other_triangles = edges_to_triangles[(u,v)]
                        for i, j, k in other_triangles:
                            if (i,j,k) not in broken_triangles:
                                edges_to_add2 = list({(i,j), (i, k), (j, k)} - self.existing_edges_in_triplet(i, j, k))
                                for edge2 in edges_to_add2:
                                    x, y = edge2
                                    self.adj_matrix[x][y] = self.adj_matrix[x][y] = np.random.normal(loc=self.mean, scale=np.sqrt(self.var))
                                broken_triangles.add((i, j, k))
    
    '''
    determine if a triplet of nodes from a metric triangle
    '''
    def is_metric_triangle(self, a, b, c):
        ab, bc, ac = self.adj_matrix[a][b], self.adj_matrix[b][c], self.adj_matrix[a][c]
        return (ab + bc >= ac) and (ab + ac >= bc) and (ac + bc >= ab)

    '''
    Count the number of node triplets that violate the triangle inequality in a graph
    '''
    def count_violations(self):
        count = 0
        for a, b, c in itertools.combinations(range(self.n), 3):
            if not self.is_metric_triangle(a, b, c):
                count += 1
        return count
    
    '''
    Break the triangle inequality on >= num_violations of node triplets in the graph
    Currently set to break every triangle in a graph
    '''
    def break_triangle_inequality(self, num_violations=0, margin=1.1):
        self.is_metric = False
        triplets = list(itertools.combinations(range(self.n), 3))
        num_violations = len(triplets)
        random.shuffle(triplets)
        current_violations = set()

        def triplet_key(i, j, k):
            return tuple(sorted([i, j, k]))

        total_violations = 0

        while total_violations < num_violations:
            for (i, j, k) in triplets:
                if total_violations >= num_violations:
                    break

                # Skip if already non-metric
                if not self.is_metric_triangle(i, j, k):
                    continue

                # Inflate side ik
                a, b, c = i, k, j
                ac, cb = self.adj_matrix[a][c], self.adj_matrix[c][b]
                inflated = margin * (ac + cb)
                self.adj_matrix[a][b] = self.adj_matrix[b][a] = inflated

                # Check all affected triangles containing edge (a, b) = (i, k)
                new_violations = 0
                for x in range(self.n):
                    if x != a and x != b:
                        key = triplet_key(a, b, x)
                        if not self.is_metric_triangle(a, b, x):
                            if key not in current_violations:
                                current_violations.add(key)
                                new_violations += 1
                        else:
                            if key in current_violations:
                                current_violations.remove(key)
                                new_violations -= 1

                total_violations += new_violations
                
    '''
    Compute the length of a given tour, in list format [0,2,3, ...] = tour 0->2->3 ...
    '''
    def compute_tour_length(self, tour):
        return sum(self.adj_matrix[tour[i - 1], tour[i]] for i in range(len(tour)))

    '''
    Compute average tour length through exhaustive search
    '''
    def average_tour_length_exhaustive(self):
        nodes = list(range(self.n))
        total_length = 0.0
        count = 0
        min_length = float("inf")
        for perm in itertools.permutations(nodes[1:]):
            tour = [0] + list(perm)
            len = self.compute_tour_length(tour)
            if len < min_length:
                min_length = len
            total_length += len
            count += 1
        return round(total_length / count, 2), round(min_length, 2)

    '''
    Compute average tour length using sampling
    '''
    def average_tour_length_sampled(self, samples=20000):
        total_length = 0.0
        min_length = float("inf")
        for _ in range(samples):
            tour = random.sample(range(self.n), self.n)
            len = self.compute_tour_length(tour)
            if len < min_length:
                min_length = len      
            total_length += len
        return round(total_length / samples, 2), round(min_length, 2)

    '''
    Function to computer tour length based on input graph size
    '''
    def compute_average_tour_length(self):
        if self.n <= 8:
            #print("Using exhaustive search...")
            return self.average_tour_length_exhaustive()
        else:
            #print("Using random sampling...")
            return self.average_tour_length_sampled()
            
    '''
    Compute max(dist(V', V'')+dist(V'', V^3)) for use in normalization factor
    '''
    def max_two_hop_distance(self):
        max_dist = 0.0
        for v1 in range(self.n):
            for v2 in range(self.n):
                if v2 == v1:
                    continue
                for v3 in range(self.n):
                    if v3 == v1 or v3 == v2:
                        continue
                    two_hop = self.adj_matrix[v1][v2] + self.adj_matrix[v2][v3]
                    if two_hop > max_dist:
                        max_dist = two_hop
        return max_dist
    
    '''
    Find the edges in a graph that break the triangle ineq, and what triangles they are associated with
    '''
    def edges_breaking_triangles(self):
        for a, b, c in itertools.combinations(range(self.n), 3):
            distances = {(a,b): self.adj_matrix[a][b], (a,c): self.adj_matrix[a][c], (b,c): self.adj_matrix[b][c]}
            violating_edge = max(distances, key=distances.get)
            # Check if edge breaks the triangle inequality
            longest = distances[violating_edge]
            other_edges = [d for e, d in distances.items() if e != violating_edge]
            if longest >= sum(other_edges):
                self.non_metric_edges[violating_edge].append((a, b, c))

        for edge, triangles in self.non_metric_edges.items():
            print(f"{edge}: {triangles}")
    
    '''
    Read a TSP from a csv into .adj_matrix
    '''
    def read_distance_matrix_from_csv(self, filename):
        with open(filename, newline='') as f:
            reader = list(csv.reader(f))
            self.n = int(reader[0][1])
            self.mean = float(reader[4][1])
            self.var = float(reader[5][1])
            self.is_metric = bool(reader[6][1])
            matrix_rows = reader[9:9+self.n]
            self.adj_matrix = np.zeros((self.n, self.n))
            for i, row in enumerate(matrix_rows):
                for j, dist in enumerate(row[1:]):
                    self.adj_matrix[i][j] = float(dist)
    
    '''
    Save TSP instance to CSV file
    '''
    def save_TSP_to_csv(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["nodes", self.n])
            writer.writerow(["lambda", self.max_two_hop_distance()])
            lengths = self.compute_average_tour_length()
            avg = lengths[0]
            min_length = lengths[1]
            writer.writerow(["average_tour_length", f"{avg}"])
            writer.writerow(["minimum_tour_length", f"{min_length}"])
            writer.writerow(["Mean", f"{self.mean}"])
            writer.writerow(["Variance", f"{self.var}"])
            writer.writerow(["Is_Metric", f"{self.is_metric}"])
            writer.writerow(["k", f"{self.count_violations()}"])
            writer.writerow([""] + [f"Node_{i}" for i in range(self.n)])
            for i, row in enumerate(self.adj_matrix):
                writer.writerow([f"Node_{i}"] + [f"{dist:.2f}" for dist in row])
            print(f"Generated TSP instance with {self.n} nodes and saved to {path}\n")
