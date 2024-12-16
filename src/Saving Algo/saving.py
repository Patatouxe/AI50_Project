import numpy as np
import pandas as pd

# Calcul des économies pour chaque paire de clients (i, j)
def calculate_savings(distance_matrix, depot):
    savings = []
    num_locations = len(distance_matrix)
    for i in range(1, num_locations):
        for j in range(i + 1, num_locations):
            saving = distance_matrix[depot][i] + distance_matrix[depot][j] - distance_matrix[i][j]
            savings.append((saving, i, j))
    return sorted(savings, reverse=True)

# Initialisation des routes (chaque client est une route séparée au début)
def initialize_routes(num_clients):
    return {i: [i] for i in range(1, num_clients)}

# Fusion des routes en utilisant les économies maximales
def savings_algorithm(distance_matrix, depot):
    savings = calculate_savings(distance_matrix, depot)
    routes = initialize_routes(len(distance_matrix))
    for saving, i, j in savings:
        route_i = next((route for route in routes.values() if route[-1] == i), None)
        route_j = next((route for route in routes.values() if route[0] == j), None)
        
        # Fusionner les routes si elles existent et ne sont pas encore fusionnées
        if route_i and route_j and route_i != route_j:
            route_i.extend(route_j)
            del routes[route_j[0]]
    
    # Ajouter le dépôt au début et à la fin de chaque route
    for route in routes.values():
        route.insert(0, depot)
        route.append(depot)
    
    return list(routes.values())

print(initialize_routes(5))