#Author : Daniel_JK


from matplotlib import cm , pyplot as plt
import random
import numpy as np
from math import sqrt
from typing import List, Tuple, Dict
from dataclasses import dataclass
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)

@dataclass
class Route:
    customers: List[int]
    capacity: float
    distance: float


class MACS_CVRP:


    def __init__(self, cvrp_instance, rho=0.1, local_rho=0.1, min_pheromon=0.1, max_pheromon=2.0):
        self.solution_dist = None
        self.solution_vei = None
        self.instance = cvrp_instance
        self.num_customers = cvrp_instance.dimension
        self.customer_coords = cvrp_instance.node_coord
        self.customer_demands = cvrp_instance.demand
        self.vehicle_capacity = cvrp_instance.capacity
        self.depot = cvrp_instance.depot[0]

        #Pheromone parameters
        self.local_rho = local_rho                        # Local pheromone update parameter(evaporation rate)
        self.initial_pheromone = 1.0                  # Initial pheromone level
        self.rho = rho     # Global pheromone update parameter(evaporation rate)
        self.min_pheromon = min_pheromon
        self.max_pheromon = max_pheromon

        #Pheromone trails
        self.pheromone = np.full((self.num_customers+1, self.num_customers+1), self.initial_pheromone)
        self.inactive_count = np.zeros((self.num_customers+1, self.num_customers+1))

        # Best solution found by the algorithm
        self.best_solution= None
        self.best_distance = float('inf')
        self.best_vehicles = float('inf')

        #Colony specific parameters
        self.vei_alpha = 1.0
        self.dist_alpha = 1.0   # Pheromone influence of VEI and DIST colonies
        self.init_vei_alpha = 1.0
        self.init_dist_beta = 3.0
        self.vei_beta = 2.0        # Heuristic  influence of VEI colonies
        self.dist_beta = 3.0        # Heuristic  influence of DIST colonies

        #performance tracker
        self.history = {
            'distances': [],
            'vehicles': [],
            'improvements': [],
            'colony_wins': {'vei': [], 'dist': []}
        }
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)



    def construct_solution_vei(self) -> List[Route]:
        """
            Constructs a solution for the CVRP using the VEI (Vehicle Efficiency Index) colony.

            This method attempts to build a feasible solution by iteratively selecting the next customer to visit
            based on pheromone levels and the remaining vehicle capacity. The process continues until all customers
            are visited or the maximum number of attempts is reached.

            Returns:
                List[Route]: A list of routes representing the solution found by the VEI colony.
            """

        solution = []
        unvisited = set(range(1, self.num_customers + 1))
        max_attempts = 100


        while unvisited and max_attempts > 0:
            route = []
            remaining_capacity = self.vehicle_capacity
            current = self.depot
            local_attempts = 50 #prevent routing from getting stuck in local minima

            while unvisited and local_attempts > 0:
                feasible = [(j, self.customer_demands[j])
                            for j in unvisited
                            if self.customer_demands[j] <= remaining_capacity]
                if not feasible:
                    break

                #Add logger for debugging
                self.logger.debug(f'Feasible nodes: {feasible}, Current node: {current}, Remaining capacity: {remaining_capacity}')

                try:
                    probs = []
                    nodes = []

                    # KnapSack heuristic selection
                    for j,  demand in feasible:
                        pheromone = max(float(self.pheromone[current][j]), self.min_pheromon)
                        efficiency = demand/remaining_capacity if remaining_capacity > 0 else 0
                        prob = (pheromone ** self.vei_alpha) * (efficiency ** self.vei_beta)

                        if prob > 0:
                            probs.append(prob)
                            nodes.append(j)
                    if not nodes:
                        next_node = random.choice([j for j, _  in feasible])
                    else:
                        total = sum(probs)
                        probs = [p / total for p in probs]
                        next_node = np.random.choice(nodes, p=probs)

                    route.append(next_node)
                    unvisited.remove(next_node)
                    remaining_capacity -= self.customer_demands[next_node]
                    current = next_node
                    self.local_pheromone_update(current, next_node)

                except Exception as e:
                    self.logger.error(f"Error in VEI route construction: {str(e)}")
                    local_attempts -= 1
                    continue

                local_attempts-= 1

            if route:
                route = self. optimize_route_sequence(route)
                distance = self.calculate_route_distance(route)
                solution.append(Route(route, self.vehicle_capacity - remaining_capacity, distance))

            max_attempts -= 1

        return solution

    def construct_solution_dist(self) -> List[Route]:
        """
            Constructs a solution for the CVRP using the DIST (Distance) colony.

            This method attempts to build a feasible solution by iteratively selecting the next customer to visit
            based on pheromone levels and the distance to the next customer. The process continues until all customers
            are visited or the maximum number of attempts is reached.

            Returns:
                List[Route]: A list of routes representing the solution found by the DIST colony.
            """
        solution = []
        unvisited = set(range(1, self.num_customers + 1))
        max_attempts = 100

        while unvisited and max_attempts > 0:
            route = []
            remaining_capacity = self.vehicle_capacity
            current = self.depot
            local_attempts = 50

            while unvisited and local_attempts > 0:
                feasible = [(j, self.distance(current, j))
                            for j in unvisited
                            if self.customer_demands[j] <= remaining_capacity]
                if not feasible:
                    break

                try:
                    probs = []
                    nodes = []        #track feasiblible nodes with valid probabilities

                    for j, dist  in feasible:
                        if dist == 0:
                            continue
                        pheromone = max(float(self.pheromone[current][j]), self.min_pheromon)
                        prob = (pheromone ** self.dist_alpha) * ((1 / dist) ** self.dist_beta)

                        if prob > 0:
                            probs.append(prob)
                            nodes.append(j)
                    if not nodes :
                        next_node = random.choice([j for j, _ in feasible])
                    else :
                        total = sum(probs)
                        probs = [p / total for p in probs]
                        next_node = np.random.choice(nodes, p=probs)

                    route.append(next_node)
                    unvisited.remove(next_node)
                    remaining_capacity -= self.customer_demands[next_node]
                    current = next_node

                except Exception as e:
                    self.logger.error(f"Error in DIST route construction: {str(e)}")
                    local_attempts -= 1
                    continue

                local_attempts -= 1

            if route:
                route = self.optimize_route_sequence(route)
                distance = self.calculate_route_distance(route)
                solution.append(Route(route, self.vehicle_capacity - remaining_capacity, distance))

            max_attempts -= 1

        if not solution:
            self.logger.warning("DIST colony failed to find a solution.")
            return [Route([1], self.customer_demands[1], 0)]

        return solution

    def construct_initial_solution(self):
        solution = [[]]
        remaining_capacity = self.vehicle_capacity
        current_node = self.depot[0]

        print(f'there are {len(solution)} customers')
        while len(solution) < self.num_customers:

            print(f'there are {self.num_customers} customers')
            next_node = self.choose_next_node(current_node, remaining_capacity)
            if next_node is None:
                solution.append([])
                remaining_capacity = self.vehicle_capacity
                current_node = self.depot[0]
            else:
                solution[-1].append(next_node)
                remaining_capacity -= self.customer_demands[next_node]
                current_node = next_node

            # Log the current state of the solution
            logging.info(f"Current solution: {solution[-1]}")
            #logging.info(f"Current node: {current_node}")

        return solution

    def optimize_route_sequence(self, route: List[int]) -> List[int]:
        if len(route) <= 2:
            return route

        # Step 1: Initial optimization using nearest neighbor
        optimized_route = self.nearest_neighbor_optimization(route)

        # Step 2: Further improvement using k-means clustering and local search
        #optimized_route = self.cluster_based_optimization(optimized_route)

        # Step 3: Final refinement with 3-opt
        #optimized_route = self.three_opt_optimization(optimized_route)

        return optimized_route

    def nearest_neighbor_optimization(self, route: List[int]) -> List[int]:
        """Optimize route using nearest neighbor approach"""
        if not route:
            return route

        unvisited = set(route)
        current = route[0]
        optimized_route = [current]
        unvisited.remove(current)

        while unvisited:
            next_node = min(unvisited,
                            key=lambda x: self.distance(current, x))
            optimized_route.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        return optimized_route

    def cluster_based_optimization(self, route: List[int]) -> List[int]:
        """Optimize route using k-means clustering"""
        if len(route) <= 3:
            return route

        # Get coordinates for route points
        coords = np.array([self.customer_coords[i] for i in route])

        # Determine number of clusters (square root of points)
        n_clusters = max(2, int(np.sqrt(len(route))))

        # Perform k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        clusters = kmeans.fit_predict(coords)

        # Optimize within each cluster
        cluster_routes = []
        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_route = [route[idx] for idx in cluster_indices]
            if cluster_route:
                # Optimize cluster using nearest neighbor
                optimized_cluster = self.nearest_neighbor_optimization(cluster_route)
                cluster_routes.append(optimized_cluster)

        # Connect clusters optimally
        final_route = []
        unvisited_clusters = set(range(len(cluster_routes)))
        current_cluster = 0

        while unvisited_clusters:
            final_route.extend(cluster_routes[current_cluster])
            unvisited_clusters.remove(current_cluster)

            if unvisited_clusters:
                # Find closest cluster
                last_node = cluster_routes[current_cluster][-1]
                current_cluster = min(unvisited_clusters,
                                      key=lambda c: min(self.distance(last_node, node)
                                                        for node in cluster_routes[c]))

        return final_route

    def three_opt_optimization(self, route: List[int]) -> List[int]:
        """Refine route using 3-opt local search"""

        def reverse_segment(tour, i, j):
            return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]

        improved = True
        best_distance = self.calculate_route_distance(route)
        best_route = route[:]

        while improved:
            improved = False
            for i in range(len(route) - 3):
                for j in range(i + 2, len(route) - 1):
                    for k in range(j + 2, len(route)):
                        # Consider all possible 3-opt moves
                        new_routes = [
                            reverse_segment(best_route, i, j - 1),
                            reverse_segment(best_route, j, k - 1),
                            reverse_segment(best_route, i, k - 1)
                        ]

                        for new_route in new_routes:
                            new_distance = self.calculate_route_distance(new_route)
                            if new_distance < best_distance:
                                best_route = new_route[:]
                                best_distance = new_distance
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

        return best_route

    def choose_next_node(self, feasible_nodes, current_node):
        """
        Chooses the next node to visit based on the current node and remaining capacity.

        Parameters:
        feasible_nodes (list): A list of nodes that can be visited next based on the remaining capacity.
        current_node (int): The current node being visited.

        Returns:
        int: The next node to visit, or None if no feasible node is found.
        """
        # Check if there are any feasible nodes to choose from
        if not feasible_nodes:
            print("Warning: No feasible nodes available.")
            return None

        # Ensure current_node index is within bounds
        if current_node >= len(self.pheromone):
            print(
                f"Error: current_node index {current_node} is out of bounds for pheromone matrix with size {len(self.pheromone)}.")
            return None

        # Calculate total attractiveness, ensuring indices are within bounds
        total_attractiveness = 0
        for node in feasible_nodes:
            # Check if node is within bounds of pheromone and distance arrays
            if node < len(self.pheromone[current_node]) and self.distance(current_node, node) != 0 and node != \
                    self.depot[0]:
                try:
                    total_attractiveness += self.pheromone[current_node][node] / self.distance(current_node, node)
                except IndexError:
                    print(
                        f"Error: Node index {node} is out of bounds for pheromone matrix row with size {len(self.pheromone[current_node])}.")
                    continue

        # If the total attractiveness is 0, return None to indicate no feasible node
        if total_attractiveness == 0:
            return None

        # Calculate probabilities, handling edge cases
        probabilities = []
        for node in feasible_nodes:
            if node < len(self.pheromone[current_node]) and node != self.depot[0]:
                distance = self.distance(current_node, node)
                if distance > 0:
                    probabilities.append(
                        round(self.pheromone[current_node][node] / (total_attractiveness * distance), 6))
                else:
                    probabilities.append(0)
            else:
                probabilities.append(0)  # Set probability to 0 if node is out of bounds or is the depot

        # Choose the next node based on probabilities
        try:
            if len(feasible_nodes) == len(probabilities):
                chosen_node = random.choices(feasible_nodes, weights=probabilities, k=1)[0]
            else:
                print("Warning: Number of weights does not match the population. Choosing randomly.")
                chosen_node = random.choice(feasible_nodes)
        except IndexError:
            print("Error: No valid customer node could be chosen due to index error.")
            return None

        return chosen_node


    def calculate_total_distance(self, solution):
        total_distance = 0
        for route in solution:
            if not route:
                continue
            total_distance += self.distance(self.depot[0], route[0])  # Go to first customer
            for i in range(1, len(route)):
                total_distance += self.distance(route[i-1], route[i])
            total_distance += self.distance(route[-1], self.depot[0])  # Return to depot
        return total_distance

    def distance(self, i, j):
        xi, yi = self.customer_coords[i]
        xj, yj = self.customer_coords[j]
        return sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    def local_pheromone_update(self, i: int, j: int):
        decay = 1.0 - self.local_rho
        deposit = self.local_rho * self.min_pheromon

        self.pheromone[i][j] = max(
            self.min_pheromon,
            min(self.max_pheromon, decay * float(self.pheromone[i][j]) + deposit)
        )
        self.pheromone[j][i] = self.pheromone[i][j]

        self.inactive_count[i][j] = 0
        self.inactive_count[j][i] = 0

    def global_pheromone_update(self, best_solution: List[Route]):
        self.pheromone *= (1.0 - self.rho)

        total_distance = sum(route.distance for route in best_solution)
        deposit = 1.0 / total_distance if total_distance > 0 else 0

        for route in best_solution:
            prev = self.depot
            for current in route.customers:
                self.pheromone[prev][current] += deposit
                self.pheromone[current][prev] = self.pheromone[prev][current]
                prev = current

            self.pheromone[prev][self.depot] += deposit
            self.pheromone[self.depot][prev] = self.pheromone[prev][self.depot]

    def handle_inactivity(self):
        self.inactive_count += 1
        evaporation_rate = np.clip(0.1 * (self.inactive_count / 10), 0.1, 0.9)
        inactive_mask = self.inactive_count > 10

        #Apply local evaporation to inactive pheromones
        self.pheromone[inactive_mask] *= (1 - evaporation_rate[inactive_mask])

        #Reinforce strong trails
        strong_trails = self.pheromone > np.mean(self.pheromone)
        self.pheromone[strong_trails] *= 1.1

        self.pheromone = np.clip(self.pheromone, self.min_pheromon, self.max_pheromon)

    def calculate_route_distance(self, route: List[int]) -> float:
        if not route:
            return 0

        distance = self.distance(self.depot, route[0])
        for i in range(len(route) - 1):
            distance += self.distance(route[i], route[i + 1])
        distance += self.distance(route[-1], self.depot)

        return distance

    def evaluate_solution (self, solution: List[Route]) -> Tuple[float, float]:
        """
        Evaluate the solution found by the algorithm.
        :param solution:
        :return:
        """

        distance_score = sum(route.distance for route in solution)
        capacity_score = sum(route.capacity for route in solution)/(len(solution) * self.vehicle_capacity)
        return distance_score, capacity_score

    def weighted_solution_score(self, solution: List[Route]) -> float:
        """
        Calculate the weighted score of the solution based on distance and capacity.
        :param solution:"""
        dist_score, cap_score = self.evaluate_solution(solution)
        #higher weight to distance minimization
        weighted_score = 0.7 * (1/dist_score)+ 0.3 * cap_score
        return weighted_score

    def update_colony_parameters(self):
        #Adjust based on solution quality
        if self.best_distance< float('inf'):
            improvement_rate = self.history['distances'][-1] / self.best_distance

            self.vei_alpha = self.init_vei_alpha * (2 - improvement_rate)
            self.dist_beta = self.init_dist_beta * (1 + improvement_rate)

            self.vei_alpha = np.clip(self.vei_alpha, 0.5, 2.0)
            self.dist_beta = np.clip(self.dist_beta, 2.0, 5.0)

    def eval_colony_performance(self):
        if not hasattr(self, 'colony_stats'):
            self.colony_stats = {'vei_wins': 0, 'dist_wins': 0}

        def score_solution(solution):
            return self.weighted_solution_score(solution)

        vei_score = score_solution(self.solution_vei)
        dist_score = score_solution(self.solution_dist)

        if vei_score> dist_score:
            self.colony_stats['vei_wins'] += 1
        else:
            self.colony_stats['dist_wins'] += 1
        return vei_score, dist_score

    def maintain_solution_diversity(self, solution: List[Route]) -> List[Route]:
        print("Maintaining solution diversity...")
        if not solution:
            return solution

        modified_solution = []
        used_edges = set()

        for route in solution:
            new_route = []
            current = self.depot
            unvisited = set(route.customers)

            while unvisited:
                next_nodes = sorted(unvisited,
                                    key=lambda x: (
                                        0 if (current, x) in used_edges else 1,
                                        self.distance(current, x)
                                    ))
                next_node = next_nodes[0]
                new_route.append(next_node)
                used_edges.add((current, next_node))
                unvisited.remove(next_node)
                current = next_node

            distance = self.calculate_route_distance(new_route)
            capacity = sum(self.customer_demands[i] for i in new_route)
            modified_solution.append(Route(new_route, capacity, distance))
        print("Solution diversity maintained.")
        return modified_solution

    def balance_vehicle_capacity(self, solution: List[Route]) -> List[Route]:
        self.logger.info("Starting vehicle capacity balancing...")
        avg_capacity = sum(route.capacity for route in solution) / len(solution)

        unbalanced = [i for i, route in enumerate(solution)
                      if abs(route.capacity - avg_capacity) > 0.2 * self.vehicle_capacity]
        self.logger.info(f"Number of unbalanced routes: {len(unbalanced)}")

        if not unbalanced:
            return solution

        balanced_solution = solution.copy()
        for idx in unbalanced:
            route = balanced_solution[idx]
            self.logger.info(f"Balancing route {idx} with capacity {route.capacity}:.2f")

            if route.capacity > avg_capacity:
                # Find underutilized route
                target_idx = min(range(len(balanced_solution)),
                                 key=lambda i: balanced_solution[i].capacity
                                 if i != idx else float('inf'))

                # Transfer customers
                while route.capacity > avg_capacity:
                    customer = min(route.customers,
                                   key=lambda x: self.customer_demands[x])
                    if self.customer_demands[customer] + balanced_solution[
                        target_idx].capacity <= self.vehicle_capacity:
                        route.customers.remove(customer)
                        balanced_solution[target_idx].customers.append(customer)
                        route.capacity -= self.customer_demands[customer]
                        balanced_solution[target_idx].capacity += self.customer_demands[customer]
                    else:
                        break
        self.logger.info(f" routes in solution are all balanced f")

        return balanced_solution

    def run(self, max_iterations: int) -> Tuple[List[Route], float, int]:
        update_plot =self.visualize_process()

        try:
            for iteration in range(max_iterations):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_dist = executor.submit(self.construct_solution_dist)
                    future_vei = executor.submit(self.construct_solution_vei)
                    solution_dist = future_dist.result()
                    solution_vei = future_vei.result()

                #Track metrics
                distance_vei = sum(route.distance for route in solution_vei)
                distance_dist = sum(route.distance for route in solution_dist)

                self.history['distances'].append(min(distance_vei, distance_dist))

                #Colony competition
                self.solution_vei = solution_vei
                self.solution_dist = solution_dist
                vei_score, dist_score = self.eval_colony_performance()

                winnin_solution, winning_distance = (solution_vei, distance_vei )\
                    if vei_score > dist_score \
                    else (solution_dist, distance_dist)


                #Update the  best solution and distance based on winning coloony
                if len(winnin_solution) < self.best_vehicles or \
                        (len(winnin_solution) == self.best_vehicles and
                         winning_distance < self.best_distance):
                    self.best_solution = winnin_solution
                    self.best_distance = winning_distance
                    self.best_vehicles = len(winnin_solution)
                    self.global_pheromone_update(winnin_solution)

                self.eval_colony_performance()
                self.update_colony_parameters()
                self.handle_inactivity()
                self.best_solution = self.maintain_solution_diversity(self.best_solution)
                self.best_solution = self.balance_vehicle_capacity(self.best_solution)

                #UPdate visualization
                update_plot(iteration, solution_vei if distance_vei < distance_dist else solution_dist, self.pheromone)
                self.history['distances'].append(winning_distance)
                self.history['vehicles'].append(len(winnin_solution))

                self.logger.info(f"Iteration {iteration}: Best distance = {self.best_distance:.2f}, "
                                 f"Vehicles = {self.best_vehicles}")

        except Exception as e:
            self.logger.error(f"Error in running MACS-CVRP: {str(e)}")
            raise

        plt.ioff()  # Turn off interactive mode

        return self.best_solution, self.best_distance, self.best_vehicles

    def visualize_solution(self):
        plt.figure(figsize=(12, 8))

        # Plot depot
        plt.plot(self.customer_coords[self.depot][0], self.customer_coords[self.depot][1],
                 'ks', markersize=10, label='Depot')

        # Plot routes with pheromone levels as line thickness
        max_pheromone = np.max(self.pheromone)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.best_solution)))

        for idx, (route, color) in enumerate(zip(self.best_solution, colors)):
            customer_coords = [self.customer_coords[self.depot]]
            customer_coords.extend(self.customer_coords[i] for i in route.customers)
            customer_coords.append(self.customer_coords[self.depot])

            for i in range(len(customer_coords)-1):
                start = customer_coords[i]
                end = customer_coords[i+1]

                if i < len(route.customers):
                    node1 = route.customers[i-1] if i > 0 else self.depot
                    node2 = route.customers[i]
                    width = 1 + 5 * float(round((self.pheromone[node1][node2] / max_pheromone), 3))
                else:
                    width = 1

                plt.plot([start[0], end[0]], [start[1], end[1]],
                         '-o', color=color, alpha=0.7,
                         linewidth=width,
                         label=f'Route {idx + 1} (Cap: {route.capacity:.0f})' if i == 0 else "")

        plt.title(f'Best Solution: {self.best_distance:.2f} units, {self.best_vehicles} vehicles')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def visualize_process(self):
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        def update_plot(iteration, current_solution, pheromone_levels):
            for ax in (ax1, ax2, ax3):
                ax.clear()

            # Network visualization (ax1)
            G = nx.Graph()
            pos = {}

            # Add depot
            G.add_node(self.depot, node_type='depot')
            pos[self.depot] = self.customer_coords[self.depot]


            #Add nodess
            for i in range(1, self.num_customers + 1):
                G.add_node(i, node_type='customer')
                pos[i] = self.customer_coords[i]


            # Add edges with pheromone weights
            max_pheromone = np.max(pheromone_levels)
            edges = []
            weights = []

            for i in G.nodes():
               for j in G.nodes():
                   if i < j and pheromone_levels[i][j] > self.min_pheromon:
                       edges.append((i, j))
                       weights.append(pheromone_levels[i][j])

            G.add_weighted_edges_from([(i, j, w) for (i, j), w in zip(edges, weights)])

            # Draw network
            node_colors = ['red' if node == self.depot else 'lightblue'
                           for node in G.nodes()]
            edge_weights = [G[u][v]['weight'] / max_pheromone for u, v in G.edges()]

            nx.draw_networkx_nodes(G,
                                   pos,
                                   node_color=node_colors,
                                   node_size=100,
                                   node_shape='h',
                                   label='Depot' if self.depot  else 'cust',
                                   ax=ax1)
            nx.draw_networkx_edges(G,
                                   pos,
                                   edge_color=edge_weights,
                                   edge_cmap=plt.cm.get_cmap('YlOrRd'),
                                   width=[1 + 2 * w  for w in edge_weights],
                                   ax=ax1)

            # Current solution in ax2
            self._plot_solution(ax2, current_solution, pheromone_levels, max_pheromone)
            ax2.set_title(f'Current Solution (Iteration {iteration}) Cap : {sum(route.capacity for route in current_solution)}')

            # Pheromone distribution histogram in ax3
            ax3.hist(pheromone_levels.flatten(), bins=50)
            ax3.set_title('Pheromone Distribution')
            ax3.set_xlabel('Pheromone Level')
            ax3.set_ylabel('Frequency')

            # Performance metrics in ax4
            if hasattr(self, 'history'):
                ax4.clear()
                ax4.plot(self.history.get('distances', []), label='Distance')
                ax4.plot(self.history.get('vehicles', []), label='Vehicles')
                ax4.set_title('Optimization Progress')
                if hasattr(self, 'colony_stats'):
                    ax4.clear()
                    ax4.bar(['VEI Colony', 'DIST Colony'],
                            [self.colony_stats['vei_wins'], self.colony_stats['dist_wins']])
                    ax4.set_title('Colony Performance')
                    total_wins = sum(self.colony_stats.values())
                    if total_wins > 0:
                        for i, v in enumerate(self.colony_stats.values()):
                            ax4.text(i, v, f'{(v / total_wins) * 100:.1f}%', ha='center')
                ax4.legend()

            plt.tight_layout()
            plt.pause(0.1)

        return update_plot

    def _plot_route(self, ax, route, color, pheromone_levels, max_pheromone):
        """Plot individual route with pheromone-weighted edges"""
        coords = [self.customer_coords[self.depot]]
        coords.extend(self.customer_coords[i] for i in route.customers)
        coords.append(self.customer_coords[self.depot])

        # Plot edges with varying thickness based on pheromone levels
        for i in range(len(coords) - 1):
            start = coords[i]
            end = coords[i + 1]

            if i < len(route.customers):
                node1 = route.customers[i - 1] if i > 0 else self.depot
                node2 = route.customers[i]
                width = 1 + 5 * (pheromone_levels[node1][node2] / max_pheromone)
            else:
                width = 1

            ax.plot([start[0], end[0]], [start[1], end[1]],
                    '-o', color=color, alpha=0.7,
                    linewidth=width,
                    label=f'Route {route.customers[0]} (Cap: {route.capacity:.0f})' if i == 0 else "")

        # Plot customer points
        for coord in coords[1:-1]:
            ax.plot(coord[0], coord[1], 'o', color=color, markersize=6)

    def  _plot_solution(self, ax, solution, pheromone_levels, max_pheromone):
        ax.plot(self.customer_coords[self.depot][0], self.customer_coords[self.depot][1],
                'ks', markersize=10, label='Depot')

        colors = plt.cm.rainbow(np.linspace(0, 1, len(solution)))
        for route, color in zip(solution, colors):
            self._plot_route(ax, route, color, pheromone_levels, max_pheromone)

        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1))

