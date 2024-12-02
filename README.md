# AI50_Project


Capacitated VRP (C-VRP)

CVRP is a Vehicle Routing Problem (VRP) in which a fixed fleet of delivery vehicles of uniform capacity must service known customer demands for a single commodity from a common depot at minimum transit cost. That is, CVRP is like VRP with the additional constraint that **every vehicles must have uniform capacity** of a single commodity. 

We can find below a formal description for the CVRP:

- **Objective:** The objective is to minimize the vehicle fleet and the sum of distance, and the total demand of commodities for each route may not exceed the capacity of the vehicle which serves that route.

- **Feasibility:** A solution is feasible if the total quantity assigned to each route does not exceed the capacity of the vehicle which services the route.

- **Formulation:** Let Q denote the capacity of a vehicle. Mathematically, a solution for the CVRP is the same that VRP's one, but with the additional restriction that the total demand of all customers supplied on a route $R_{i}$ does not exceed the vehicle capacity

  	 $$Q: \sum_{i=1}^{m} q_{i} \leq Q$$


#### Variables: 
-  $n_{customer}$ : number of customers to be served n a unique depot
-  $q_{i}$ : Quantity of goods demanded by the customer (i = 1, ..,n)
-  $Q_{vehicle}$ : Capacity of vehicle available to deliver the good. Vehicle has to return to reload.
- $R_{i}$ : Route used by a vehicle to supply customer
-  $d_{ij}$ : Distance from customer i to customer j
#### Solution: 
Our solution of CVRP will be represented by a collection(array) of tours on the routes where all customers are served, and the total tour demand is Q

Objectif function : 

			$$S_{x} = [[(R_{i}, T_{i}), \sum_{i=1} q_{i}];    [(R_{n}, T_{n}), \sum_{n=1} q_{n}]] 
					where Q = max(total q)$$
Graphical theoritical formulation, 
	For a graph  G = (C, L),
		we have $c_{0}$ as the depot and $c_{n}/{n≠0}$ each note has a fixed demanded quantity  $q_{n}$. With $G_{0}= (c_{0},(c_{i},c_{j}) i≠j≠0)$ and $q_{0} = 0$  
		Each arc $(c_{i},c_{j})$ has a distance $d_{ij}$


#### Solution techniques: 

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

[Benchmark](https://www.bernabe.dorronsoro.es/vrp/)
For Tabu Research:
- Rochat and Taillard, 1995
For Genetic Algorithm:
- Potvin and Bengio, 1996(PB)


# AI50 Project - Capacitated Vehicle Routing Problem (CVRP)

An implementation of Multiple Ant Colony System for solving the Capacitated Vehicle Routing Problem (CVRP).

## Requirements

- Python 3.10+
- Dependencies:
  - numpy
  - matplotlib
  - tqdm
  - requests_toolbelt

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install numpy matplotlib tqdm requests_toolbelt
```

## Project Structure

```
.
├── data/
│   └── tai/           # CVRP benchmark instances
├── src/
│   ├── ACO/
│   │   └── aco_colony.py  # MACS-CVRP implementation
│   └── CVRP.py       # CVRP problem definition
└── main.py           # Main execution script
```

## Usage

Run the main script to execute the MACS-CVRP algorithm:

```bash
python main.py
```

The script will:
1. Load a CVRP instance from the data directory
2. Initialize the MACS-CVRP solver
3. Run the optimization for 100 iterations
4. Display the best solution found, including:
   - Route assignments
   - Total distance
   - Number of vehicles used

## Algorithm Details

The implementation uses Multiple Ant Colony System (MACS) with two colonies:
- ACS-VEI: Minimizes the number of vehicles
- ACS-DIST: Minimizes the total travel distance

Key parameters:
- `rho`: Global pheromone evaporation rate (default: 0.1)
- `local_rho`: Local pheromone update rate (default: 0.1)

## Benchmarks

The project uses Taillard's CVRP benchmark instances. Solutions can be compared against:
- Rochat and Taillard, 1995
- Potvin and Bengio, 1996 (PB)

## Development

### Testing
Add unit tests in a `tests/` directory (to be implemented).

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

## License
[Add your license information here]