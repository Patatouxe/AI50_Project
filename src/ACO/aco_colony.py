#Author : Daniel_JK
import random
from math import sqrt
from tqdm import tqdm
from matplotlib import pyplot as plt

import logging

from requests_toolbelt.multipart.encoder import total_len

# Configure logging
logging.basicConfig(level=logging.INFO)

class MACS_CVRP:
    def __init__(self, cvrp_instance, rho=0.1, local_rho=0.1):
        self.num_customers = cvrp_instance.dimension
        self.customer_coords = cvrp_instance.node_coord
        self.customer_demands = cvrp_instance.demand
        self.vehicle_capacity = cvrp_instance.capacity
        self.depot = cvrp_instance.depot
        self.local_rho = local_rho                        # Local pheromone update parameter(evaporation rate)
        self.initial_pheromone = 1                  # Initial pheromone level
        self.rho = rho     # Global pheromone update parameter(evaporation rate)
        self.pheromone = [[1 for _ in range(self.num_customers)] for _ in range(self.num_customers)]
        self.best_solution = [[]]
        #self.best_distance = self.calculate_total_distance(self.best_solution)
        self.best_distance = float('inf')
        self.best_vehicles = float('inf')
        #self.best_vehicles = len(self.best_solution)



    def construct_initial_solution(self):
        solution = [[]]
        remaining_capacity = self.vehicle_capacity
        current_node = self.depot[0]

        while len([node for route in solution for node in route]) < self.num_customers:
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
            logging.info(f"Current solution: {solution}")
            logging.info(f"Current node: {current_node}")

        return solution

    def choose_next_node(self, current_node, remaining_capacity):
        """
            Chooses the next node to visit based on the current node and remaining capacity.

            Parameters:
            current_node (int): The current node being visited.
            remaining_capacity (int): The remaining capacity of the vehicle.

            Returns:
            int: The next node to visit, or None if no feasible node is found.
            """
        if not self.best_solution or not any(self.best_solution):
            return None

        feasible_nodes = [node for node in range(self.num_customers) if node not in [route[-1] for route in self.best_solution] and remaining_capacity >= self.customer_demands[node]]
        if not feasible_nodes:
            return None

        total_attractiveness = sum(self.pheromone[current_node][node] / self.distance(current_node, node) for node in feasible_nodes)
        probabilities = [self.pheromone[current_node][node] / (total_attractiveness * self.distance(current_node, node)) for node in feasible_nodes]
        chosen_node = random.choices(feasible_nodes, weights=probabilities, k=1)[0]
        return chosen_node

    def calculate_total_distance(self, solution):
        total_distance = 0
        for route in solution:
            if not route:
                continue
            for i in range(len(route)):
                if i > 0:
                    total_distance += self.distance(route[i-1], route[i])
            total_distance += self.distance(route[-1], self.depot[0])  # Return to depot
        return total_distance

    def distance(self, i, j):
        xi, yi = self.customer_coords[i]
        xj, yj = self.customer_coords[j]
        return sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    def run(self, max_iterations):
        with tqdm(total=max_iterations, desc="MACS Progress" ) as pbar:
            for _ in range(max_iterations):
                # ACS-VEI colony (maximize vehicle load)
                new_solution_vei = self.construct_solution(maximize_load=True)
                new_distance_vei = self.calculate_total_distance(new_solution_vei)
                new_vehicles_vei = len(new_solution_vei)

                if new_vehicles_vei < self.best_vehicles or (new_vehicles_vei == self.best_vehicles and new_distance_vei < self.best_distance):
                    self.best_solution = new_solution_vei
                    self.best_distance = new_distance_vei
                    self.best_vehicles = new_vehicles_vei

                # ACS-DIST colony (minimize distance)
                new_solution_dist = self.construct_solution(maximize_load=False)
                new_distance_dist = self.calculate_total_distance(new_solution_dist)
                new_vehicles_dist = len(new_solution_dist)

                if new_vehicles_dist == self.best_vehicles and new_distance_dist < self.best_distance:
                    self.best_solution = new_solution_dist
                    self.best_distance = new_distance_dist

                self.update_pheromones(new_solution_vei, new_solution_dist)

                #visual log
                self.visual_log(_)

                pbar.update(1)
        return self.best_solution, self.best_distance, self.best_vehicles

    def construct_solution(self, maximize_load):
        solution = [[]]
        remaining_capacity = self.vehicle_capacity
        current_node = self.depot[0]

        while len([node for route in solution for node in route]) < self.num_customers:
            next_node = self.choose_next_node(current_node, remaining_capacity)
            if next_node is None:
                if solution[-1]:    # Only append the new route if the current route is not empty
                    solution.append([])
                remaining_capacity = self.vehicle_capacity
                current_node = self.depot[0]
            else:
                solution[-1].append(next_node)
                remaining_capacity -= self.customer_demands[next_node]
                current_node = next_node

        return solution

    def update_pheromones(self, solution_vei, solution_dist):

        """
            Updates the pheromone levels on the paths based on the solutions provided by the ACS-VEI and ACS-DIST colonies.

            Parameters:
            solution_vei (list of lists): The solution provided by the ACS-VEI colony.
            solution_dist (list of lists): The solution provided by the ACS-DIST colony.
            rho : global pheromone level
            local_rho : local pheromone level
            """

        # Global pheromone update
        for i in range(self.num_customers):
            for j in range(self.num_customers):
                self.pheromone[i][j] = (1 - self.rho) * self.pheromone[i][j] + self.rho * (1 / self.best_distance)

        # Local pheromone update (optional)
        for route in solution_vei:
            for i in range(len(route)):
                if i > 0:
                    self.pheromone[route[i-1]][route[i]] = (1 - self.local_rho) * self.pheromone[route[i - 1]][route[i]] + self.local_rho * self.initial_pheromone
        for route in solution_dist:
            for i in range(len(route)):
                if i > 0:
                    self.pheromone[route[i-1]][route[i]] = (1 - self.local_rho) * self.pheromone[route[i - 1]][route[i]] + self.local_rho * self.initial_pheromone

    def visual_log(self, iteration):
        plt.figure(figsize=(10, 6))
        for route in self.best_solution:
            if route:
                x_coords = [self.customer_coords[node][0] for node in route]
                y_coords = [self.customer_coords[node][1] for node in route]
                plt.plot(x_coords, y_coords, marker='o')
                plt.plot([self.customer_coords[self.depot[0]][0], x_coords[0]], [self.customer_coords[self.depot[0]][1], y_coords[0]], 'r--')
                plt.plot([x_coords[-1], self.customer_coords[self.depot[0]][0]], [y_coords[-1], self.customer_coords[self.depot[0]][1]], 'r--')
        plt.scatter(self.customer_coords[self.depot[0]][0], self.customer_coords[self.depot[0]][1], c='red', marker='s', label='Depot')
        plt.title(f'Iteration {iteration}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()