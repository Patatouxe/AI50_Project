 #### ***Ant Colony System*** 
 Using the version by Gambardella, Taillard and Agazzi 1999, we implement a MACS-CVRP (Multiple Ant colony System for Capacitated VRP). 
 The **objective function**s fo two ant colonies:
 -  one for minimizing the vehicle/vehicle tours by maximising the load of each vehicle ensuring it stays within capacity limits 
 - the other to minimize the total travel distances, given the fixed number of vehicles.
 These colonies interact with each other through pheromone updates as a shared memory system.
**Capacity Constraints** 
Each ant will ensure that when constructing routes, the total demand of customers assigned to a vehicle does not exceed its capacity. 
Incorporating the capacity and total demand of customers assigned to a vehicle to not exceed the vehicle capacity $Q: \sum_{i=1}^{m} q_{i} \leq Q$
Variables : 
- Choose the define the instances level we want to address. Small and large instances according to benchmark used to test

**Algorithm process** 

Solution construction: Using the MACS-CVRP, we will use the **distance between customers** and the **remaining vehicle capacity** as part of the attractiveness calculation, ensuring that ants prioritize closer customers while respecting capacity. Also while using the local search procedure to find the closer customer to improve the solution by capacity constraint.

Pheromones updates : Globally and updating by using best solution found by either colony. Prioritize solutions with fewer vehicles must capacity constraints must always be met. and locally updating of pheromones emphasizing feasible routes that obey objective functions

MACS - CVRP

``` MACS-CVRP()
1. /* Initialization */
    best_solution ← Initial solution using nearest neighbor heuristic with capacity constraint.
    
1. /* Main loop */
    Repeat
        active_vehicles ← Number of vehicles in the best_solution.
        Activate ACS-VEI (to minimize vehicles)
        Activate ACS-DIST (to minimize distance with fixed vehicles)

        While ACS-VEI and ACS-DIST are active:
            Wait for an improved solution from either ACS-VEI or ACS-DIST.
            If a better solution (fewer vehicles or shorter distance) is found, update the best_solution.
            If ACS-VEI finds a solution with fewer vehicles, kill ACS-DIST and restart both colonies.

    Until stopping criteria met.

```
This will allow the two-colony structure to efficiently solve the CVRP.

``` #ACS-VEI number of minimization of vehicles  
	**Initialisation**
		(( S // best feasible solutions with shortest travel distance
		active_vehicle(S) //commputes the vehicles on the feasible solutions))
		S_init  // initial solution with unlimited num of neighbors with K-means.
		customers //Array of all customers visited
		
	Main Loop
		// V : Active vehicles(S_init)
		While true 
			for each ant k
				// construct a solution S
				new_active_ant(k, local_search=False, IN)
				for j in customers
					if j NOT IN customers
						IN_j ++
			If k number_visited_customers(for k, feasible solution) > number_visited_customers(for unfeasible solution) 
				unfeasible solution = feasible 
				for j in customers:
					IN_j = 0   // reset IN
					if unfeasible solution IS feasible
						send to colony
			// perform global updating of pheromones for both solution sets
				  
					 	
```
This colony to minimise the number of vehicle

For the distance minimisation the colony will the following ;
- Solution construction with capacity constraints: 
	- Each ant will constructs a solution (i.e., a set of vehicle routes) where the total demand of customers assigned to a vehicle does not exceed the vehicle capacity.
	- The focus will be on minimizing the total travel distance for each route while ensuring that all customers are served and the capacity constraint is respected.
- Global Pheromone Update:
	- In ACS-DIST, the pheromone update favors routes with shorter distances.
	- Only the best solution found in each iteration will update the pheromone levels.
- Pheromone and Heuristic Calculation:
	•	Use both pheromone trails and a heuristic based on distance (closer customers are more attractive) ants ensures they do not violate the capacity constraint during route construction.
	•	The attractiveness  $\eta_{ij}$  in is inversely proportional to the distance between customer  $i$  and customer  $j$ .
```ACS-DIST-Capacity 

/* Solution construction for ACS-DIST with capacity constraint */
Procedure construct_solution_with_capacity_constraint(ant k)

1. /* Initialization */
    Put ant k at the depot.
    Initialize current_route ← empty route, load ← 0, current_location ← depot.

2. /* Solution construction loop */
    Repeat
        /* Compute the set of feasible customers (i.e., those whose demand does not exceed remaining capacity) */
        feasible_customers ← { customers that can be added without exceeding capacity }

        /* For each feasible customer, compute the attractiveness (inverse of distance) */
        for each customer j in feasible_customers:
            attractiveness[j] ← 1 / distance(current_location, j) /*(inverse of distance) */

        /* Select the next customer probabilistically based on the pheromone and attractiveness */
        next_customer ← select_customer_based_on_pheromone_and_attractiveness(feasible_customers)

        /* Add the customer to the current route */
        current_route ← current_route + next_customer
        load ← load + demand[next_customer]
        current_location ← next_customer

        /* Update local pheromone (optional, same as ACS-TIME) */
        local_pheromone_update(current_location, next_customer)

    Until no more feasible customers are available or route is complete.

3. /* Return the constructed route as part of the solution */

Return current_route.


/* Global pheromone update for ACS-DIST */

Procedure global_pheromone_update(best_solution)

1. /* For each edge (i, j) in the best solution */

	for each (i, j) in best_solution:
        /* Update the pheromone on edge (i, j) based on the distance */
        pheromone[i][j] ← (1 - ρ) * pheromone[i][j] + ρ * (1 / distance(best_solution))

```