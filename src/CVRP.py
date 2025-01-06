#@Author : tmayer
import re
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class Route:
    """
    Represents a route for the CVRP solution.

    Attributes:
        customers (List[int]): The list of customer nodes included in the route.
        capacity (float): The total demand served by the route.
        distance (float): The total distance traveled in the route.
    """
    customers: List[int]
    capacity: float
    distance: float

class CVRP:
    """
    A class for modeling and solving the Capacitated Vehicle Routing Problem (CVRP).

    Attributes:
        name (str): The name of the problem instance.
        dimension (int): The number of nodes (customers + depot).
        capacity (int): The capacity of each vehicle.
        num_trucks (int): The number of trucks available.
        node_coord (dict): Coordinates of each node as {node_id: (x, y)}.
        demand (dict): The demand for each node as {node_id: demand}.
        depot (List[int]): List of depot node IDs.
        cout (float): Cost metric (optional, may depend on implementation).
        solution (dict): Placeholder for storing solutions (optional).
    """
    def __init__(self, path):
        """
        Initialize the CVRP instance by loading data from a given file.

        Args:
            path (str): Path to the problem instance file.
        """
        self.name = ""
        self.dimension = 0
        self.capacity = 0
        self.num_trucks = 0
        self.node_coord = {}
        self.demand = {}
        self.depot = []
        self.cout = 0
        self.solution = {}
        
        self.load_data(path)

    def load_data(self, path):
        """
        Parse and load the problem instance data from the given file.

        Args:
            path (str): Path to the problem instance file.
        """
        with open(path, 'r') as file:
            section = None
            for line in file:
                line = line.strip()
                if line.startswith("NAME"):
                    self.name = line.split(":")[1].strip()
                elif line.startswith("COMMENT"):
                    # Capture du commentaire sur une seule ligne
                    #self.comment = line.split(":", 1)[1].strip()   #On ne capture plus comment car la variable est inutile
                    #print(f"DEBUG - Comment line: {self.comment}")
                    # Extraction du nombre de camions
                    match = re.search(r"No of trucks:\s*(\d+)", line.split(":", 1)[1].strip())
                    if match:
                        self.num_trucks = int(match.group(1))
                        #print(f"DEBUG - Number of trucks found: {self.num_trucks}")
                elif line.startswith("DIMENSION"):
                    self.dimension = int(line.split(":")[1].strip())
                elif line.startswith("CAPACITY"):
                    self.capacity = int(line.split(":")[1].strip())
                elif line.startswith("NODE_COORD_SECTION"):
                    section = "NODE_COORD_SECTION"
                elif line.startswith("DEMAND_SECTION"):
                    section = "DEMAND_SECTION"
                elif line.startswith("DEPOT_SECTION"):
                    section = "DEPOT_SECTION"
                elif line.startswith("EOF"):
                    break
                elif section == "NODE_COORD_SECTION":
                    parts = line.split()
                    node_id = int(parts[0])
                    x, y = int(parts[1]), int(parts[2])
                    self.node_coord[node_id] = (x, y)
                elif section == "DEMAND_SECTION":
                    parts = line.split()
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    self.demand[node_id] = demand
                elif section == "DEPOT_SECTION":
                    node_id = int(line)
                    if node_id == -1:
                        break
                    self.depot.append(node_id)

    def __repr__(self):
        """
        Represent the CVRP instance in a human-readable format.

        Returns:
            str: A formatted string with the details of the CVRP instance.
        """
        details = [
            f"Name: {self.name}",
            f"Dimension: {self.dimension}",
            f"Capacity: {self.capacity}",
            f"Number of Trucks: {self.num_trucks}",
            f"Depot: {self.depot}"
        ]

        node_coords = "\n".join([f"  Node {node}: (x={x}, y={y})" for node, (x, y) in self.node_coord.items()])
        node_coords_section = f"Node Coordinates:\n{node_coords}"

        demands = "\n".join([f"  Node {node}: Demand={demand}" for node, demand in self.demand.items()])
        demands_section = f"Demands:\n{demands}"

        return "\n".join(details + [node_coords_section, demands_section])

    def format_solution(self, routes: List[Route], total_distance: float) -> str:
        """
        Format the solution for presentation.

        Args:
            routes (List[Route]): The list of routes in the solution.
            total_distance (float): The total distance of the solution.

        Returns:
            str: A formatted string representing the solution.
        """
        output_lines = []

        for i , route in enumerate(routes, 1):
            route_capacity= sum(self.demand[customer] for customer in route.customers)
            route_str = f"Route #{i}: {'  '.join(map(str, route.customers))}  (Total demand answered: {route_capacity})"
            output_lines.append(route_str)

        output_lines.append(f"Total distance: {total_distance:.2f}")
        return "\n ".join(output_lines)



    def validate_solution(self, routes: List[Route]) -> Tuple[bool, str]:
        """
        Validate the feasibility of a given solution.

        Args:
            routes (List[Route]): The list of routes to validate.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating feasibility and a message.
        """
        visited_solution = set()

        # Check if the solution is feasible
        for route in routes:
            route_capacity = sum(self.demand[customer] for customer in route.customers)
            if route_capacity > self.capacity:
                return False, "Route exceeds capacity :  {route_capacity} > {self.capacity}"
            for customer in route.customers:
                if customer in visited_solution:
                    return False, f"Customer {customer} visited more than once"
                visited_solution.add(customer)

        all_customers =set(range(1, self.dimension + 1))
        if visited_solution != all_customers:
            missing = all_customers - visited_solution
            return False, f"Unvisited customers: {missing}"

        return True, "Solution is Feasible"