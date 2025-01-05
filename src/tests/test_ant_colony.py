
import unittest
from unittest.mock import MagicMock

from src.ACO_MACS.aco_colony import MACS_CVRP, Route


class TestMACS_CVRP(unittest.TestCase):

    def setUp(self):
        self.mock_instance = MagicMock()
        self.mock_instance.dimension = 5
        self.mock_instance.node_coord = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (3, 3), 4: (4, 4), 5: (5, 5)}
        self.mock_instance.demand = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        self.mock_instance.capacity = 10
        self.mock_instance.depot = [0]
        self.macs_cvrp = MACS_CVRP(self.mock_instance)

    def test_constructs_solution_vei_within_capacity(self):
        self.macs_cvrp.customer_demands = {1: 1, 2: 2, 3: 3, 4: 4}
        self.macs_cvrp.vehicle_capacity = 10
        solution = self.macs_cvrp.construct_solution_vei()
        self.assertTrue(all(route.capacity <= self.macs_cvrp.vehicle_capacity for route in solution))

    def test_constructs_solution_dist_within_capacity(self):
        self.macs_cvrp.customer_demands = {1: 1, 2: 2, 3: 3, 4: 4}
        self.macs_cvrp.vehicle_capacity = 10
        solution = self.macs_cvrp.construct_solution_dist()
        self.assertTrue(all(route.capacity <= self.macs_cvrp.vehicle_capacity for route in solution))

    def test_constructs_solution_vei_all_customers_visited(self):
        self.macs_cvrp.customer_demands = {1: 1, 2: 2, 3: 3, 4: 4}
        self.macs_cvrp.vehicle_capacity = 10
        solution = self.macs_cvrp.construct_solution_vei()
        visited_customers = {customer for route in solution for customer in route.customers}
        self.assertEqual(visited_customers, set(self.macs_cvrp.customer_demands.keys()))

    def test_constructs_solution_dist_all_customers_visited(self):
        self.macs_cvrp.customer_demands = {1: 1, 2: 2, 3: 3, 4: 4}
        self.macs_cvrp.vehicle_capacity = 10
        solution = self.macs_cvrp.construct_solution_dist()
        visited_customers = {customer for route in solution for customer in route.customers}
        self.assertEqual(visited_customers, set(self.macs_cvrp.customer_demands.keys()))

    def test_constructs_solution_vei_no_feasible_solution(self):
        self.macs_cvrp.customer_demands = {1: 11, 2: 12, 3: 13, 4: 14}
        self.macs_cvrp.vehicle_capacity = 10
        solution = self.macs_cvrp.construct_solution_vei()
        self.assertEqual(solution, [Route([1], 11, 0)])

    def test_constructs_solution_dist_no_feasible_solution(self):
        self.macs_cvrp.customer_demands = {1: 11, 2: 12, 3: 13, 4: 14}
        self.macs_cvrp.vehicle_capacity = 10
        solution = self.macs_cvrp.construct_solution_dist()
        self.assertEqual(solution, [Route([1], 11, 0)])

    def test_calculates_total_distance_correctly(self):
        solution = [Route([1, 2, 3], 6, 10), Route([4, 5], 9, 8)]
        total_distance = self.macs_cvrp.calculate_total_distance(solution)
        self.assertEqual(total_distance, 18)


if __name__ == '__main__':
    unittest.main()
