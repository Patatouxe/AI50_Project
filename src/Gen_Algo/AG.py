#@Author : tmayer
import random
import math
from src.CVRP import CVRP, Route


class GeneticAlgorithmCVRP:
    """
    Implements a Genetic Algorithm (GA) for solving the Capacitated Vehicle Routing Problem (CVRP).

    Attributes:
        cvrp (CVRP): An instance of the CVRP class containing the problem data.
        generations (int): The number of generations for the algorithm to run.
        population_size (int): The number of individuals in the population.
        mutation_rate (float): The probability of mutation for each individual.
        population (List[List[Route]]): The current population of solutions.
    """
    def __init__(self, cvrp_instance, generations=100, population_size=50, mutation_rate=0.1):
        """
        Initialize the Genetic Algorithm with parameters and a CVRP instance.

        Args:
            cvrp_instance (CVRP): The CVRP instance with problem details.
            generations (int): Number of generations for the algorithm to evolve.
            population_size (int): Size of the population.
            mutation_rate (float): Mutation probability for each solution.
        """
        self.cvrp = cvrp_instance
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []

    @staticmethod
    def calculate_distance(coord1, coord2):
        """
        Calculate the Euclidean distance between two coordinates.

        Args:
            coord1 (Tuple[float, float]): The (x, y) coordinates of the first point.
            coord2 (Tuple[float, float]): The (x, y) coordinates of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def calculate_route_cost(self, route):
        """
        Calculate the total distance of a route, including returning to the depot.

        Args:
            route (List[int]): A list of customer nodes in the route.

        Returns:
            float: The total distance of the route.
        """
        cost = 0
        depot = self.cvrp.depot[0]
        current_location = depot
        for node in route:
            cost += self.calculate_distance(self.cvrp.node_coord[current_location], self.cvrp.node_coord[node])
            current_location = node
        cost += self.calculate_distance(self.cvrp.node_coord[current_location], self.cvrp.node_coord[depot])
        return cost

    def fitness(self, solution):
        """
        Calculate the fitness of a solution. Higher fitness corresponds to lower total route cost.

        Args:
            solution (List[Route]): A solution consisting of a list of routes.

        Returns:
            float: The fitness value of the solution.
        """
        total_cost = sum(self.calculate_route_cost(route.customers) for route in solution)
        return 1 / (1 + total_cost)

    def initialize_population(self):
        """
        Initialize the population with random solutions by randomly shuffling customers.
        """
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
        """
        Repair a solution to ensure feasibility by adding missing nodes and removing excess nodes.

        Args:
            solution (List[Route]): The solution to be repaired.

        Returns:
            List[Route]: The repaired solution.
        """
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
        """
        Perform Order Crossover (OX) between two parents to produce a child solution.
        Steps in OX Crossover : 
        1. Input:
         - Two parent solutions (parent1 and parent2) represented as flat sequences of nodes (customers).
         - These solutions are derived from routes by flattening them into a single list of customer nodes (ignoring depot representation).
        2. Random Subsequence Selection:
         - Randomly select two indices (start and end) in the range of the length of the parent solution.
         - Extract the subsequence between these indices from parent1. This subsequence will remain fixed in the child.
        Example:
            parent1 = [A, B, C, D, E, F, G, H]
            parent2 = [E, G, A, F, H, B, D, C]
            Assume start = 2, end = 5.
            Subsequence from parent1: [C, D, E]
        3. Create an Empty Child:
         - Create an empty child solution of the same length as the parents.
         - Copy the subsequence from parent1 into the child at the same positions.
        Example:
            child = [None, None, C, D, E, None, None, None]
        4. Fill Remaining Slots from parent2
         - Traverse parent2 in order, skipping nodes that are already in the child's subsequence.
            Final child: [G, A, C, D, E, F, H, B].


        Args:
            parent1 (List[Route]): The first parent solution.
            parent2 (List[Route]): The second parent solution.

        Returns:
            List[Route]: The child solution.
        """
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
        """
        Mutate a solution by applying inversion mutation and route reassignment.

        Args:
            solution (List[Route]): The solution to mutate.
        """
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
        """
        Split a flat list of customers into valid routes.

        Args:
            flat_solution (List[int]): The flat list of customer nodes.
            parent_solution (List[Route]): A reference parent solution for structure.

        Returns:
            List[Route]: A list of valid routes.
        """
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
        """
        Execute the Genetic Algorithm to solve the CVRP.

        Returns:
            Tuple[List[Route], float]: The best solution and its total cost.
        """
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