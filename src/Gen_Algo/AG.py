import random
import math
from src.CVRP import CVRP, Route


class GeneticAlgorithmCVRP:
    def __init__(self, cvrp_instance, generations=100, population_size=50, mutation_rate=0.1):
        self.cvrp = cvrp_instance
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []

    @staticmethod
    def calculate_distance(coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def calculate_route_cost(self, route):
        cost = 0
        depot = self.cvrp.depot[0]
        current_location = depot
        for node in route:
            cost += self.calculate_distance(self.cvrp.node_coord[current_location], self.cvrp.node_coord[node])
            current_location = node
        cost += self.calculate_distance(self.cvrp.node_coord[current_location], self.cvrp.node_coord[depot])
        return cost

    def fitness(self, solution):
        total_cost = sum(self.calculate_route_cost(route.customers) for route in solution)
        return 1 / (1 + total_cost)

    def initialize_population(self):
        nodes = list(self.cvrp.node_coord.keys())
        nodes.remove(self.cvrp.depot[0])
        for _ in range(self.population_size):
            random.shuffle(nodes)
            solution = []
            current_route = []
            current_capacity = 0
            for node in nodes:
                if current_capacity + self.cvrp.demand[node] <= self.cvrp.capacity:
                    current_route.append(node)
                    current_capacity += self.cvrp.demand[node]
                else:
                    distance = self.calculate_route_cost(current_route)
                    solution.append(Route(customers=current_route, capacity=current_capacity, distance=distance))
                    current_route = [node]
                    current_capacity = self.cvrp.demand[node]
            if current_route:
                distance = self.calculate_route_cost(current_route)
                solution.append(Route(customers=current_route, capacity=current_capacity, distance=distance))
            self.population.append(solution)

    def repair_solution(self, solution):
        nodes = set(self.cvrp.node_coord.keys()) - {self.cvrp.depot[0]}
        used_nodes = set(node for route in solution for node in route.customers)
        missing_nodes = list(nodes - used_nodes)
        excess_nodes = list(used_nodes - nodes)

        for route in solution:
            for node in excess_nodes:
                if node in route.customers:
                    route.customers.remove(node)

        for node in missing_nodes:
            for route in solution:
                route_capacity = sum(self.cvrp.demand[n] for n in route.customers)
                if route_capacity + self.cvrp.demand[node] <= self.cvrp.capacity:
                    route.customers.append(node)
                    route.capacity = route_capacity + self.cvrp.demand[node]
                    break
            else:
                distance = self.calculate_route_cost([node])
                solution.append(Route(customers=[node], capacity=self.cvrp.demand[node], distance=distance))

        return solution

    def crossover(self, parent1, parent2):
        """Order Crossover (OX)"""
        flat_parent1 = [node for route in parent1 for node in route.customers]
        flat_parent2 = [node for route in parent2 for node in route.customers]

        # Select a random subsequence
        start, end = sorted(random.sample(range(len(flat_parent1)), 2))
        child = [None] * len(flat_parent1)
        child[start:end] = flat_parent1[start:end]

        # Fill the rest from parent2 in order
        current_pos = end
        for node in flat_parent2:
            if node not in child:
                if current_pos >= len(flat_parent1):
                    current_pos = 0
                child[current_pos] = node
                current_pos += 1

        return self.split_into_routes(child, parent1)

    def mutate(self, solution):
        """Improved Mutation: Inversion Mutation + Route Reassignment"""
        for route in solution:
            if random.random() < self.mutation_rate and len(route.customers) > 2:
                # Inversion Mutation
                i, j = sorted(random.sample(range(len(route.customers)), 2))
                route.customers[i:j] = reversed(route.customers[i:j])

        # Route Reassignment Mutation
        if random.random() < self.mutation_rate:
            # Select a random node and move it to another route
            non_empty_routes = [route for route in solution if len(route.customers) > 1]
            if len(non_empty_routes) >= 2:
                source_route = random.choice(non_empty_routes)
                target_route = random.choice(solution)

                if source_route != target_route:
                    node = random.choice(source_route.customers)
                    source_route.customers.remove(node)

                    if sum(self.cvrp.demand[n] for n in target_route.customers) + self.cvrp.demand[node] <= self.cvrp.capacity:
                        target_route.customers.append(node)

        self.repair_solution(solution)

    def split_into_routes(self, flat_solution, parent_solution):
        routes = []
        current_route = []
        current_capacity = 0
        for node in flat_solution:
            demand = self.cvrp.demand[node]
            if current_capacity + demand <= self.cvrp.capacity:
                current_route.append(node)
                current_capacity += demand
            else:
                distance = self.calculate_route_cost(current_route)
                routes.append(Route(customers=current_route, capacity=current_capacity, distance=distance))
                current_route = [node]
                current_capacity = demand
        if current_route:
            distance = self.calculate_route_cost(current_route)
            routes.append(Route(customers=current_route, capacity=current_capacity, distance=distance))
        return routes

    def run(self):
        self.initialize_population()

        for generation in range(self.generations):
            fitness_scores = [(solution, self.fitness(solution)) for solution in self.population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            parents = [solution for solution, _ in fitness_scores[:self.population_size // 2]]
            offspring = []

            for _ in range(len(parents) // 2):
                parent1, parent2 = random.sample(parents, 2)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                offspring.append(self.repair_solution(child1))
                offspring.append(self.repair_solution(child2))

            for child in offspring:
                self.mutate(child)

            self.population = parents + offspring

        best_solution = max(self.population, key=lambda solution: self.fitness(solution))
        self.cvrp.solution = best_solution
        total_cost = 1 / self.fitness(best_solution) - 1
        return best_solution, total_cost