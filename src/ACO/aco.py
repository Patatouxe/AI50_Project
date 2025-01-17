#Classical ACO with base formula + mu + pheromon evaporation formula (+ 

import math
import random as rand
from src.CVRP import Route

class Colony:
    
    def __init__(self, cvrpInstance, nbrIter, alpha, beta, gamma, evapRate,theta, Q):
        #print(cvrpInstance.node_coord)
        self.cvrpInstance = cvrpInstance
        self.computeMatrixD()
        self.matrixP = [[1.0 for j in range(cvrpInstance.dimension)] for i in range(cvrpInstance.dimension)]
        self.evapRate = evapRate
        self.theta = theta
        self.nbrAnts = cvrpInstance.dimension
        self.nbrIter = nbrIter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.Q = Q

    def computeMatrixD(self):
        dimension = self.cvrpInstance.dimension
        node_coord = self.cvrpInstance.node_coord
        self.matrixD = []
        temp = []
        for i in range(dimension):
            for j in range(dimension):
                temp.append(math.sqrt((node_coord[i+1][0]-node_coord[j+1][0])**2+(node_coord[i+1][1]-node_coord[j+1][1])**2))
            self.matrixD.append(temp.copy())
            temp.clear()

    def pheromonUpdate(self, solutions, costs):
        averageCost = sum(costs)/self.nbrAnts
        for i in range(len(self.matrixP)):
            for j in range(len(self.matrixP[i])):
                self.matrixP[i][j] *= (1-self.evapRate+self.theta/averageCost)# simulates the evaporation process of the pheromone trail in nature which depends on the length of the path travelled by an ant
                #self.matrixP[i][j] *= (1-self.evapRate)

        for solution, cost in zip(solutions,costs):
            pheromone = self.Q/cost
            for route in solution:
                routeArray = route.customers
                for i in range(len(routeArray)):
                    current_node = routeArray[i]
                    next_node = routeArray[i+1] if i+1<len(routeArray) else 1 # Return to depot
                    self.matrixP[current_node-1][next_node-1] += pheromone
                    self.matrixP[next_node-1][current_node-1] += pheromone

    def solve(self):
        bestSol = None
        bestCost = float('inf')

        #lock = Lock()

        for _ in range(self.nbrIter):
            antsSolutions = []
            antsCosts = []
            
            for _ in range(self.nbrAnts):
                solution, cost = self.antSolution()
                antsSolutions.append(solution)
                antsCosts.append(cost)

                # Update best solution
                if cost < bestCost:
                    bestSol = solution
                    bestCost = cost
            
            self.pheromonUpdate(antsSolutions,antsCosts)
        
        return bestSol, bestCost
                

    def antSolution(self):
        dimension = self.cvrpInstance.dimension
        demands = self.cvrpInstance.demand
        capacity = self.cvrpInstance.capacity

        unvisited = set(range(2,dimension+1)) #Depot being number 1 we exclude it (so going from 2 to dimension (included) )
        solution = []
        total_cost = 0

        while unvisited:
                route = [1] #We Begin the route at the depot (used to update the pheromones from the depot to location #1)
                remainingCap = capacity
                current_node = 1 #Beginning from Depot
                route_cost = 0
                while True:
                    probabilities = []
                    
                    #Compute the probability to move at a destination for each destination having less demand than the remaining capacity in the vehicle
                    for next_node in unvisited:
                        if demands[next_node] <= remainingCap:
                            tau = self.matrixP[current_node-1][next_node-1] ** self.alpha #The -1 is because the nodes range from 1 to dimension so we put it back to a 0 to dimension-1 to go through the matrix
                            eta = (1/self.matrixD[current_node-1][next_node-1]) ** self.beta if self.matrixD[current_node-1][next_node-1] != 0 else 1
                            mu = (self.matrixD[current_node-1][0] + self.matrixD[0][next_node-1] - self.matrixD[current_node-1][next_node-1])**self.gamma #Savings of combining 2 cities i and j on one tour instead of visiting them on 2 different tours
                            mu = mu if mu!=0 else 1 #This line is for the routes to or from the depot
                            probabilities.append((next_node,tau*eta*mu))
                        
                    if not probabilities:
                        break #No feasible next node (not enough capacity remaining), return to depot

                    #Dividing the probabilities
                    total_prob = sum(p[1] for p in probabilities)
                    probabilities = [(p[0],p[1]/total_prob) for p in probabilities]

                    #Select the next_node based on the probabilities
                    next_node = rand.choices([p[0] for p in probabilities],[p[1] for p in probabilities])[0]

                    #Move to next_node
                    route.append(next_node)
                    unvisited.remove(next_node)
                    remainingCap -= demands[next_node]
                    route_cost += self.matrixD[current_node-1][next_node-1]
                    current_node = next_node
                
                #Return to depot
                route_cost += self.matrixD[current_node-1][0]
                total_cost += route_cost
                solution.append(Route(route[1:],remainingCap,route_cost))
            
        return solution, total_cost