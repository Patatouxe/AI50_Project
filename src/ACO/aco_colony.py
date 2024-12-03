#Author : Daniel_JK


from matplotlib import cm , pyplot as plt
import random
import numpy as np
from math import sqrt
from typing import List, Tuple, Dict
from dataclasses import dataclass
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
        self.vei_alpha = self.dist_alpha = 1.0   # Pheromone influence of VEI and DIST colonies
        self.vei_beta = 2.0        # Heuristic  influence of VEI colonies
        self.dist_beta = 3.0        # Heuristic  influence of DIST colonies

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)



    def construct_solution_vei(self) -> List[Route]:
        solution = []
        unvisited = set(range(1, self.num_customers + 1))
        max_attempts = 100


        while unvisited and max_attempts > 0:
            route = []
            remaining_capacity = self.vehicle_capacity
            current = self.depot
            local_attempts = 50 #prevent routing from getting stuck in local minima

            while unvisited and local_attempts > 0:
                feasible = [j for j in unvisited if self.customer_demands[j] <= remaining_capacity]
                if not feasible:
                    break

                #Add logger for debugging
                self.logger.debug(f'Feasible nodes: {feasible}, Current node: {current}, Remaining capacity: {remaining_capacity}')

                try:
                    probs = []
                    for j in feasible:
                        pheromone = self.pheromone[current][j]

                        if remaining_capacity > 0:
                            capacity_score = self.customer_demands[j] / remaining_capacity
                            prob = (pheromone ** self.vei_alpha) * (capacity_score ** self.vei_beta)
                            probs.append(prob)
                        else:
                            probs.append(0)

                    total = sum(probs)
                    if total == 0:
                        break

                    probs = [p / total for p in probs]
                    next_node = np.random.choice(feasible, p=probs)

                    route.append(next_node)
                    unvisited.remove(next_node)
                    remaining_capacity -= self.customer_demands[next_node]
                    current = next_node

                    self.local_pheromone_update(current, next_node)

                except Exception as e:
                    self.logger.error(f"Error in VEI colony: {str(e)}")
                    local_attempts -= 1
                    continue

                local_attempts -= 1

            if route:
                distance = self.calculate_route_distance(route)
                solution.append(Route(route, self.vehicle_capacity - remaining_capacity, distance))

            max_attempts -= 1

        if not solution:
            self.logger.warning("VEI colony failed to find a solution.")
            return [Route([1], self.customer_demands[1], 0)]

        return solution

    def construct_solution_dist(self) -> List[Route]:
        solution = []
        unvisited = set(range(1, self.num_customers + 1))
        max_attempts = 100

        while unvisited and max_attempts > 0:
            route = []
            remaining_capacity = self.vehicle_capacity
            current = self.depot
            local_attempts = 50

            while unvisited and local_attempts > 0:
                feasible = [j for j in unvisited if self.customer_demands[j] <= remaining_capacity]
                if not feasible:
                    break

                try:

                    probs = []
                    feasible_list = []          #track feasiblible nodes with valid probabilities

                    for j in feasible:
                        pheromone = self.pheromone[current][j]
                        distance = self.distance(current, j)
                        if distance == 0:
                            continue
                        prob = (pheromone ** self.dist_alpha) * ((1 / distance) ** self.dist_beta)

                        if prob > 0 :
                            probs.append(prob)
                            feasible_list.append(j)

                    if not feasible_list:
                        next_node = random.choice(feasible)
                    else :
                        total = sum(probs)
                        probs = [p / total for p in probs]
                        next_node = np.random.choice(feasible_list, p=probs)


                    route.append(next_node)
                    unvisited.remove(next_node)
                    remaining_capacity -= self.customer_demands[next_node]
                    current = next_node

                except Exception as e:
                    self.logger.error(f"Error in DIST colony: {str(e)}")
                    local_attempts -= 1
                    continue

            if route:
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
            min(self.max_pheromon, decay * self.pheromone[i][j] + deposit)
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
        inactive_mask = self.inactive_count > 10
        self.pheromone[inactive_mask] *= 0.95
        self.pheromone = np.clip(self.pheromone, self.min_pheromon, self.max_pheromon)

    def calculate_route_distance(self, route: List[int]) -> float:
        if not route:
            return 0

        distance = self.distance(self.depot, route[0])
        for i in range(len(route) - 1):
            distance += self.distance(route[i], route[i + 1])
        distance += self.distance(route[-1], self.depot)

        return distance

    def run(self, max_iterations: int) -> Tuple[List[Route], float, int]:
        update_plot =self.visualize_process()

        try:
            for iteration in range(max_iterations):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_vei = executor.submit(self.construct_solution_vei)
                    future_dist = executor.submit(self.construct_solution_dist)

                    solution_vei = future_vei.result()
                    solution_dist = future_dist.result()

                #update real-time visualization
                update_plot(iteration, solution_vei, self.pheromone)


                distance_vei = sum(route.distance for route in solution_vei)
                distance_dist = sum(route.distance for route in solution_dist)

                if len(solution_vei) < self.best_vehicles or \
                        (len(solution_vei) == self.best_vehicles and distance_vei < self.best_distance):
                    self.best_solution = solution_vei
                    self.best_distance = distance_vei
                    self.best_vehicles = len(solution_vei)
                    self.global_pheromone_update(solution_vei)

                if len(solution_dist) == self.best_vehicles and distance_dist < self.best_distance:
                    self.best_solution = solution_dist
                    self.best_distance = distance_dist
                    self.global_pheromone_update(solution_dist)

                self.handle_inactivity()

                self.logger.info(f"Iteration {iteration}: Best distance = {self.best_distance:.2f}, "
                                 f"Vehicles = {self.best_vehicles}")

        except Exception as e:
            self.logger.error(f"Error in MACS-CVRP: {str(e)}")
            raise

        plt.ioff()  # Turn off interactive mode

        return self.best_solution, self.best_distance, self.best_vehicles

    def visualize_solution(self):
        plt.figure(figsize=(12, 8))

        # Plot depot
        plt.plot(self.customer_coords[self.depot][0], self.customer_coords[self.depot][1],
                 'ks', markersize=10, label='Depot')

        # Plot routes with different colors and add to legend
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.best_solution)))
        for idx, (route, color) in enumerate(zip(self.best_solution, colors)):
            customer_coords = [self.customer_coords[self.depot]]
            customer_coords.extend(self.customer_coords[i] for i in route.customers)
            customer_coords.append(self.customer_coords[self.depot])

            xs, ys = zip(*customer_coords)
            plt.plot(xs, ys, '-o', color=color, alpha=0.7,
                     label=f'Route {idx + 1} (Cap: {route.capacity:.0f})')

        plt.title(f'Best Solution: {self.best_distance:.2f} units, {self.best_vehicles} vehicles')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


    def visualize_process(self):
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        def update_plot(iteration, current_solution, pheromone_levels):
            ax1.clear()
            ax2.clear()

            # Plot current solution
            ax1.plot(self.customer_coords[self.depot][0], self.customer_coords[self.depot][1],
                     'ks', markersize=10, label='Depot')

            colors = plt.cm.rainbow(np.linspace(0, 1, len(current_solution)))
            for route, color in zip(current_solution, colors):
                coords = [self.customer_coords[self.depot]]
                coords.extend(self.customer_coords[i] for i in route.customers)
                coords.append(self.customer_coords[self.depot])

                xs, ys = zip(*coords)
                ax1.plot(xs, ys, '-o', color=color, alpha=0.7)

            ax1.set_title(f'Iteration {iteration}')
            ax1.grid(True)

            # Plot pheromone levels
            im = ax2.imshow(pheromone_levels, cmap='viridis')
            ax2.set_title('Pheromone Levels')
            plt.colorbar(im, ax=ax2)

            plt.pause(0.1)

        return update_plot

