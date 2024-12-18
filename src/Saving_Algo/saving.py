import numpy as np
import pandas as pd
from math import sqrt
from ..CVRP import CVRP

# Calcul des économies pour chaque paire de clients (i, j)
def calculate_savings(cvrp, routes):
    savings = []
    for routei in routes:
        route_i = routei[:-1]
        for routej in routes:
            route_j = routej[:-1]
            if route_i == route_j:
                continue
            # Changer les routes pour retirer le dernier noeud qui est le dépôt
            
            # Calculer l'économie de distance en fusionnant les routes i et j
            saving = calc_dist_route(cvrp, route_i) + calc_dist_route(cvrp, route_j) - calc_dist_2nodes(cvrp, route_i[-2], route_j[1])
            savings.append((saving, routei, routej))
    return sorted(savings, reverse=True)
            

def calc_dist_2nodes(cvrp, node1, node2):
    x = cvrp.node_coord[node1][0] - cvrp.node_coord[node2][0]
    y = cvrp.node_coord[node1][1] - cvrp.node_coord[node2][1]
    return sqrt(x**2 + y**2)

def merge_routes(routes, i, j):
    # Fusionner les routes aux indices i et j
    new_route = routes[i][:-1] + routes[j][1:]
    routes.pop(i)
    routes.pop(j)
    routes.append(new_route)
    # print(f"New route: {new_route}")
    return routes

def calc_dist_route(cvrp, route):
    distance = 0
    for i in range(len(route) - 1):
        distance += calc_dist_2nodes(cvrp, route[i], route[i + 1])
    return distance

def calc_demand_route(cvrp, route):
    demand = 0
    for node in route:
        demand += cvrp.demand[node]
    return demand


# Initialisation des routes (chaque client est une route séparée au début)
def initialize_routes(cvrp):
    # routes est de type [[1, 2, 3], [4, 5, 6], ...]
    routes = [[cvrp.depot[0]] + [i] + [cvrp.depot[0]] for i in range(2, cvrp.dimension + 1)]
    return routes

# Fusion des routes en utilisant les économies maximales
def savings_algorithm(cvrp):
    routes = initialize_routes(cvrp)

    while len(routes) > cvrp.num_trucks:
        savings = calculate_savings(cvrp, routes)
        for save, ri, rj in savings:  # Itérer sur une copie en utilisant slicing [:]

            # Vérifier si les routes i et j existent toujours
            if ri not in routes or rj not in routes:
                continue  # Passer à l'économie suivante

            i = routes.index(ri)
            j = routes.index(rj)
            # Vérifier que la fusion de i et j ne dépasse pas la capacité du camion
            if calc_demand_route(cvrp, ri) + calc_demand_route(cvrp, rj) > cvrp.capacity:
                continue  # Passer à la prochaine économie

            # Fusionner les routes i et j si elles existent
            if ri in routes and rj in routes and save > 0:
                # Localiser les positions de i et j dans routes
                routes = merge_routes(routes, i, j)
    return routes
                

savings = CVRP("src\data\A\A-n32-k5.vrp")
solution = savings_algorithm(savings)