import numpy as np
from math import sqrt
from typing import List, Tuple
from src.CVRP import CVRP

class Savings:
    def __init__(self, cvrp_instance: CVRP):
        self.cvrp = cvrp_instance
        self.routes = self.initialize_routes()

    def initialize_routes(self) -> List[List[int]]:
        """
        Initialize routes with each customer as a separate route.
        """
        return [[self.cvrp.depot[0]] + [i] + [self.cvrp.depot[0]] for i in range(2, self.cvrp.dimension + 1)]

    def calculate_savings(self) -> List[Tuple[float, List[int], List[int]]]:
        """
        Calculate savings for each pair of routes.
        """
        savings = []
        for route_i in self.routes:
            route_i_trimmed = route_i[:-1]
            for route_j in self.routes:
                route_j_trimmed = route_j[:-1]
                if route_i_trimmed == route_j_trimmed:
                    continue
                saving = (self.calc_dist_route(route_i_trimmed) +
                          self.calc_dist_route(route_j_trimmed) -
                          self.calc_dist_2nodes(route_i_trimmed[-1], route_j_trimmed[0]))
                savings.append((saving, route_i, route_j))
        return sorted(savings, reverse=True)

    def calc_dist_2nodes(self, node1: int, node2: int) -> float:
        """
        Calculate distance between two nodes.
        """
        x = self.cvrp.node_coord[node1][0] - self.cvrp.node_coord[node2][0]
        y = self.cvrp.node_coord[node1][1] - self.cvrp.node_coord[node2][1]
        return sqrt(x**2 + y**2)

    def merge_routes(self, i: int, j: int) -> None:
        """
        Merge two routes.
        """
        new_route = self.routes[i][:-1] + self.routes[j][1:]
        self.routes.pop(j)
        self.routes.pop(i)
        self.routes.append(new_route)

    def calc_dist_route(self, route: List[int]) -> float:
        """
        Calculate total distance for a route.
        """
        distance = 0
        for i in range(len(route) - 1):
            distance += self.calc_dist_2nodes(route[i], route[i + 1])
        return distance

    def calc_demand_route(self, route: List[int]) -> int:
        """
        Calculate total demand for a route.
        """
        return sum(self.cvrp.demand[node] for node in route)

    def run(self) -> Tuple[List[List[int]], float]:
        """
        Execute the savings algorithm.
        """
        while len(self.routes) > self.cvrp.num_trucks:
            savings = self.calculate_savings()
            for save, route_i, route_j in savings:
                if len(self.routes) <= self.cvrp.num_trucks:
                    break

                if route_i not in self.routes or route_j not in self.routes:
                    continue

                if (self.calc_demand_route(route_i) + self.calc_demand_route(route_j) > self.cvrp.capacity):
                    continue

                i, j = self.routes.index(route_i), self.routes.index(route_j)
                if save > 0:
                    self.merge_routes(i, j)

        total_distance = sum(self.calc_dist_route(route) for route in self.routes)
        return self.routes, total_distance
