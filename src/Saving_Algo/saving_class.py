import numpy as np
from math import sqrt
from typing import List, Tuple
from src.CVRP import CVRP, Route

class Savings:
    """
    Implements the Savings Algorithm for solving the Capacitated Vehicle Routing Problem (CVRP).

    Attributes:
        cvrp (CVRP): An instance of the CVRP class containing problem data.
        routes (List[Route]): A list of initial routes, where each route starts with a single customer.
    """
    def __init__(self, cvrp_instance):
        """
        Initialize the Savings algorithm with a CVRP instance.

        Args:
            cvrp_instance (CVRP): An instance of the CVRP class.
        """
        self.cvrp = cvrp_instance
        self.routes = self.initialize_routes()

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
        return sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def calculate_route_cost(self, route: List[int]) -> float:
        """
        Calculate the total distance of a route, including the return to the depot.

        Args:
            route (List[int]): A list of nodes representing the route.

        Returns:
            float: The total distance of the route.
        """
        depot = self.cvrp.depot[0]
        cost = 0
        current_location = depot

        for node in route:
            cost += self.calculate_distance(
                self.cvrp.node_coord[current_location], self.cvrp.node_coord[node]
            )
            current_location = node

        # Add the distance back to the depot
        cost += self.calculate_distance(
            self.cvrp.node_coord[current_location], self.cvrp.node_coord[depot]
        )
        return cost

    def initialize_routes(self) -> List[Route]:
        """
        Initialize the routes by assigning each customer to a separate route.

        Returns:
            List[Route]: A list of initial routes.
        """
        routes = []
        for customer in range(1, self.cvrp.dimension + 1):
            distance = self.calculate_route_cost([customer])
            routes.append(
                Route(
                    customers=[self.cvrp.depot[0], customer, self.cvrp.depot[0]],
                    capacity=self.cvrp.demand[customer],
                    distance=distance,
                )
            )
        return routes

    def calculate_savings(self) -> List[Tuple[float, int, int]]:
        """
        Calculate savings for each pair of customers based on the distance.

        Returns:
            List[Tuple[float, int, int]]: A sorted list of savings in the format (saving, customer_i, customer_j),
                                          sorted in descending order of savings.
        """
        savings = []
        depot = self.cvrp.depot[0]

        for i in range(1, self.cvrp.dimension + 1):
            for j in range(i + 1, self.cvrp.dimension + 1):
                # Calculate distances
                dist_depot_i = self.calculate_distance(self.cvrp.node_coord[depot], self.cvrp.node_coord[i])
                dist_depot_j = self.calculate_distance(self.cvrp.node_coord[depot], self.cvrp.node_coord[j])
                dist_i_j = self.calculate_distance(self.cvrp.node_coord[i], self.cvrp.node_coord[j])

                # Compute savings
                saving = dist_depot_i + dist_depot_j - dist_i_j

                # Only consider valid savings (savings must be positive)
                if saving > 0:
                    savings.append((saving, i, j))

        # Sort savings in descending order
        savings.sort(reverse=True, key=lambda x: x[0])
        return savings

    def merge_routes(self, route_i: Route, route_j: Route) -> Route:
        """
        Merge two routes if feasible.

        Args:
            route_i (Route): The first route.
            route_j (Route): The second route.

        Returns:
            Route: The merged route.
        """
        # Remove the depot from the ends of both routes
        merged_customers = route_i.customers[:-1] + route_j.customers[1:]
        merged_capacity = route_i.capacity + route_j.capacity

        # Recalculate the distance for the merged route
        merged_distance = self.calculate_route_cost(merged_customers[1:-1])

        return Route(customers=merged_customers, capacity=merged_capacity, distance=merged_distance)

    def run(self) -> Tuple[List[Route], float]:
        """
        Execute the Savings Algorithm to solve the CVRP.

        Returns:
            Tuple[List[Route], float]: A tuple containing the final list of routes and the total distance.
        """
        savings = self.calculate_savings()

        for saving, i, j in savings:
            # Find the routes containing customers i and j
            route_i = None
            route_j = None
            for route in self.routes:
                if i in route.customers[1:-1]:  # Exclude the depot
                    route_i = route
                if j in route.customers[1:-1]:  # Exclude the depot
                    route_j = route

            # If i and j are in the same route, skip
            if route_i is None or route_j is None or route_i == route_j:
                continue

            # Check if merging is feasible
            if route_i.capacity + route_j.capacity <= self.cvrp.capacity:
                # Merge the two routes
                merged_route = self.merge_routes(route_i, route_j)
                self.routes.remove(route_i)
                self.routes.remove(route_j)
                self.routes.append(merged_route)

        # Calculate total distance
        total_distance = sum(route.distance for route in self.routes)
        return self.routes, total_distance
